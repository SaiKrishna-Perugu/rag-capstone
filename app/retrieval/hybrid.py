"""
Hybrid retrieval + reranking, as distinct from plain vector search: keyword
and semantic matching fail on different queries, and fusing them covers both.

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
     LLM already configured via app/llm/providers.py -- a real, legitimate
     alternative (this is close to what Cohere's Rerank API and several
     production RAG systems do), not a placeholder. Swapping in a
     cross-encoder later is a drop-in replacement for rerank()'s body,
     not a redesign.
"""
import json
import logging
import re

from langchain_core.documents import Document

from app import config
from app.db import database
from app.llm.providers import get_embeddings, get_llm

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
    final answer's context -- see rerank() and app/retrieval/rag.py's retrieve()).
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


def session_documents(session_id: str | None) -> list | None:
    """This visitor's uploads as Documents in reading order, or None if there
    are none or they are too large to pass whole.

    The escape hatch from chunk selection. `hybrid_retrieve()` above answers
    "which passages look most relevant?", which is the right question for a
    corpus and the wrong one for a single short document the visitor just
    uploaded and is now asking about -- there, the honest answer is "all of
    it". Measured on an uploaded resume, ranking dropped 2 of 3 projects on
    one phrasing of a question and kept all 3 on another; see the note on
    config.WHOLE_DOC_MAX_CHARS.

    Returns None rather than falling back internally, so retrieve() owns the
    decision and the fallback stays visible at the call site.

    Fails open, matching cache.py: this is an optimisation over retrieval,
    not a correctness requirement, so a database error here must degrade to
    the normal hybrid path rather than costing the visitor an answer. It
    adds a query that runs *before* hybrid_search(), so without this the
    same outage would surface as a 500 from a new call site instead of the
    error the retrieval path already reports.
    """
    if not session_id:
        return None

    try:
        rows = database.get_session_chunks(session_id, config.WHOLE_DOC_MAX_CHARS)
    except Exception:
        logger.warning(
            "Whole-document lookup failed; falling back to hybrid retrieval.",
            exc_info=True,
        )
        return None
    if not rows:
        return None

    documents = []
    for row in rows:
        metadata = row.get("metadata") or {}
        metadata["source"] = row["source"]
        # Same underscore-prefixed marker hybrid_retrieve() sets: these are
        # by definition private uploads, so rag.py must keep the resulting
        # answer out of the shared semantic cache.
        metadata["_session_id"] = row.get("session_id")
        documents.append(Document(page_content=row["content"], metadata=metadata))
    return documents


def _parse_rank_order(response: str) -> list:
    """
    Extract the ranked array from a reranker reply.

    The prompt asks for a bare JSON array, but two other shapes arrive
    routinely and both fail json.loads() at character 0 with the same
    unhelpful "Expecting value: line 1 column 1 (char 0)": a markdown-fenced
    array (```json\\n[3,1,2]\\n```) and an empty string. That was not
    hypothetical -- the eval gate logged exactly three such failures per run,
    on every run, each one silently discarding the most expensive stage in
    the pipeline (~47% of per-request cost) and falling back to RRF order.
    A genuinely empty reply still raises, because there is nothing to rank
    and RRF order is the correct answer then.
    """
    text = response.strip()
    if not text:
        raise ValueError("reranker returned an empty response")

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        order = json.loads(text)
    except json.JSONDecodeError:
        # The model wrapped the array in prose despite being told not to.
        # Recovering the first bracketed group beats throwing away the call.
        match = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
        if match is None:
            raise
        order = json.loads(match.group(0))

    # A dict, bare string, or number all parse fine as JSON and then blow up
    # on `1 <= i` (TypeError) or on iteration further down.
    if not isinstance(order, list):
        raise TypeError(f"expected a JSON array, got {type(order).__name__}")
    return order


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
        f"[{i+1}] {doc.page_content[:config.CHUNK_SIZE]}" for i, doc in enumerate(candidates)
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
        response = llm.invoke(messages).content
        order = _parse_rank_order(response)
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
        # fail-open posture as check_groundedness() in app/retrieval/rag.py.
        logger.warning(f"rerank failed ({type(e).__name__}: {e}); using RRF order")
        return candidates[:top_k]


def retrieve_with_hybrid_and_rerank(
    question: str, k: int | None = None, session_id: str | None = None
) -> list:
    """The full pipeline: hybrid candidate retrieval -> LLM rerank -> top-k."""
    k = k or config.TOP_K
    candidates = hybrid_retrieve(question, k=k, session_id=session_id)
    return rerank(question, candidates, top_k=k)
