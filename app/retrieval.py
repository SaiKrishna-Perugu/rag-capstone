"""
Hybrid retrieval + reranking. Two JD requirements live here specifically:
"hybrid retrieval" and "rerankers", as distinct from plain vector search
(which is what app/rag.py did before this module existed).

Pipeline: hybrid_retrieve() -> rerank() -> top-k chunks handed to generation.

  1. hybrid_retrieve(): runs full-text (keyword/lexical via Postgres tsvector)
     and vector (semantic via pgvector) search in a SINGLE SQL query, fused
     with Reciprocal Rank Fusion (RRF). This catches both "the words match"
     queries (product codes, IDs, exact terms -- vector search alone is often
     weak here) and "the meaning matches" queries (paraphrases, synonyms --
     keyword search alone is weak here).

     The RRF fusion, vector search, and full-text search all happen inside
     Postgres rather than in Python -- eliminating the per-request BM25
     index rebuild that the Chroma-based version required, and giving us
     a shared, durable index across all Cloud Run instances.

  2. rerank(): takes the fused candidate pool and re-scores it with an
     LLM-based listwise reranker, narrowing down to the final top-k that
     actually gets passed to generation. A cross-encoder (e.g.
     sentence-transformers) is the more common production choice, but
     requires downloading model weights from HuggingFace Hub at runtime,
     which isn't guaranteed to be available in every deployment
     environment. An LLM-based reranker needs nothing beyond the same
     LLM already configured via app/providers.py -- a real, legitimate
     alternative (this is close to what Cohere's Rerank API and several
     production RAG systems do), not a placeholder. Swapping in a
     cross-encoder later is a drop-in replacement for rerank()'s body,
     not a redesign.
"""
import json
import logging

from langchain_core.documents import Document

from app import config, database
from app.providers import get_embeddings, get_llm

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF damping constant -- de-emphasizes rank-1 dominance
CANDIDATE_POOL_MULTIPLIER = 3  # retrieve 3x more candidates than final top_k before reranking

_RERANK_SYSTEM_PROMPT = """You are a search relevance judge. You will be given \
a QUESTION and a numbered list of CANDIDATE passages. Rank the candidates from \
MOST to LEAST relevant to answering the question.

Respond with ONLY a JSON array of the candidate numbers in ranked order, e.g. \
[3, 1, 4, 2]. Include every candidate number exactly once. No other text."""


def hybrid_retrieve(question: str, k: int | None = None, session_id: str | None = None) -> list:
    """
    Vector search + full-text search, fused via Reciprocal Rank Fusion
    inside a single Postgres query. Returns the top-k fused results (this
    is the CANDIDATE pool that rerank() will further narrow down, not the
    final answer's context -- see rerank() and app/rag.py's retrieve()).
    """
    k = k or config.TOP_K
    candidate_k = k * CANDIDATE_POOL_MULTIPLIER

    # Embed the question for vector similarity search
    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(question)

    # Single SQL query does vector + full-text + RRF fusion
    rows = database.hybrid_search(
        query_embedding=query_embedding,
        query_text=question,
        k=candidate_k,
        candidate_pool=candidate_k,
        # Scopes retrieval to the curated corpus plus this visitor's own
        # uploads. None means curated only.
        session_id=session_id,
    )

    # Convert database rows back to LangChain Document objects so the
    # downstream rerank() and rag.py code doesn't need to change.
    documents = []
    for row in rows:
        metadata = row.get("metadata") or {}
        metadata["source"] = row["source"]
        # Underscore-prefixed so it cannot collide with a document's own
        # metadata keys. Consumed by rag.py to decide whether an answer is
        # safe to put in the shared semantic cache -- an answer derived from
        # one visitor's private upload must not be replayed to another.
        metadata["_session_id"] = row.get("session_id")
        documents.append(
            Document(page_content=row["content"], metadata=metadata)
        )

    return documents


def rerank(question: str, candidates: list, top_k: int | None = None) -> list:
    """
    LLM-based listwise reranking: one call scores the whole candidate
    pool at once (cheaper and usually more consistent than N pointwise
    calls), returning the candidates reordered by actual relevance to
    the question -- catches cases where RRF's rank fusion still lets a
    tangentially-related chunk outrank a more relevant one.
    """
    top_k = top_k or config.TOP_K
    if len(candidates) <= top_k:
        return candidates  # nothing to narrow down

    numbered = "\n\n".join(
        f"[{i+1}] {doc.page_content[:500]}" for i, doc in enumerate(candidates)
    )
    llm = get_llm(temperature=0.0, stage="rerank")
    messages = [
        ("system", _RERANK_SYSTEM_PROMPT),
        ("human", f"QUESTION:\n{question}\n\nCANDIDATES:\n{numbered}"),
    ]

    try:
        # llm.invoke() must be INSIDE the try: it was previously outside,
        # so a transient provider failure (timeout, rate limit, connection
        # reset) propagated out of rerank() and took down the whole
        # retrieval call instead of degrading to RRF order. That defeats
        # the point of having a fallback at all, and it is not theoretical
        # -- a CI eval run logged ~11 TimeoutErrors against Groq in a
        # single pass.
        response = llm.invoke(messages).content.strip()
        order = json.loads(response)
        # The prompt asks for a JSON array, but nothing guarantees one:
        # a dict, bare string, or number all parse fine as JSON and then
        # blow up on `1 <= i` (TypeError) or iteration -- neither of which
        # the old except-tuple caught.
        if not isinstance(order, list):
            raise TypeError(f"expected a JSON array, got {type(order).__name__}")
        ranks = [i for i in order if isinstance(i, int) and 1 <= i <= len(candidates)]
        reranked = [candidates[i - 1] for i in ranks]
        # Guard against a malformed/partial response silently dropping
        # candidates -- fall back to the pre-rerank order for anything
        # the LLM didn't include.
        seen = set(ranks)
        reranked += [c for i, c in enumerate(candidates, start=1) if i not in seen]
        return reranked[:top_k]
    except Exception as e:
        # Reranking is a quality optimization, not a correctness
        # requirement -- on any failure fall back to the pre-rerank
        # (RRF-fused) order rather than failing the whole request. Same
        # fail-open posture as check_groundedness() in app/rag.py.
        logger.warning(f"rerank failed ({type(e).__name__}: {e}); using RRF order")
        return candidates[:top_k]


def retrieve_with_hybrid_and_rerank(
    question: str, k: int | None = None, session_id: str | None = None
) -> list:
    """The full pipeline: hybrid candidate retrieval -> LLM rerank -> top-k."""
    k = k or config.TOP_K
    candidates = hybrid_retrieve(question, k=k, session_id=session_id)
    return rerank(question, candidates, top_k=k)
