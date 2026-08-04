"""
Hybrid retrieval + reranking. Two JD requirements live here specifically:
"hybrid retrieval" and "rerankers", as distinct from plain vector search
(which is what app/rag.py did before this module existed).

Pipeline: hybrid_retrieve() -> rerank() -> top-k chunks handed to generation.

  1. hybrid_retrieve(): runs BM25 (keyword/lexical) and vector (semantic)
     search in parallel over a larger candidate pool, then fuses the two
     ranked lists with Reciprocal Rank Fusion (RRF). This catches both
     "the words match" queries (product codes, IDs, exact terms -- vector
     search alone is often weak here) and "the meaning matches" queries
     (paraphrases, synonyms -- BM25 alone is weak here).

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

SCOPE NOTE (bm25 index rebuild): the BM25 index is rebuilt from the full
Chroma collection on every call rather than persisted and incrementally
updated. At portfolio-project corpus sizes (tens to low thousands of
chunks) this costs milliseconds and guarantees the BM25 index can never
drift out of sync with what's actually in the vector store -- which
matters more here than the perf cost. At real production scale this is
exactly the kind of thing you'd move to a persisted, incrementally
updated BM25 index (or a hybrid-native vector DB) instead.
"""
import json

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app import config
from app.providers import get_embeddings, get_llm

RRF_K = 60  # standard RRF damping constant -- de-emphasizes rank-1 dominance
CANDIDATE_POOL_MULTIPLIER = 3  # retrieve 3x more candidates than final top_k before reranking

_RERANK_SYSTEM_PROMPT = """You are a search relevance judge. You will be given \
a QUESTION and a numbered list of CANDIDATE passages. Rank the candidates from \
MOST to LEAST relevant to answering the question.

Respond with ONLY a JSON array of the candidate numbers in ranked order, e.g. \
[3, 1, 4, 2]. Include every candidate number exactly once. No other text."""


def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def _build_bm25_retriever() -> BM25Retriever | None:
    """Fetch every chunk currently in the vector store and index it for
    BM25. Returns None if the collection is empty (nothing to index)."""
    vector_store = _get_vector_store()
    raw = vector_store.get(include=["documents", "metadatas"])

    texts = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    if not texts:
        return None

    documents = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(texts, metadatas)
    ]
    return BM25Retriever.from_documents(documents)


def _reciprocal_rank_fusion(ranked_lists: list, k: int = RRF_K) -> list:
    """
    Fuse multiple ranked lists of Documents into one ranking via RRF:
    score(d) = sum over lists of 1 / (k + rank_in_that_list(d))

    Documents are matched by (source, page_content) since Chroma/BM25
    Document objects from different retrievers aren't the same object
    instances even when they represent the same chunk.
    """
    scores = {}
    doc_by_key = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = (doc.metadata.get("source", ""), doc.page_content)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_by_key[key] = doc

    fused = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in fused]


def hybrid_retrieve(question: str, k: int | None = None) -> list:
    """
    Vector search + BM25 search, fused via Reciprocal Rank Fusion.
    Returns the top-k fused results (this is the CANDIDATE pool that
    rerank() will further narrow down, not the final answer's context --
    see rerank() and app/rag.py's retrieve()).
    """
    k = k or config.TOP_K
    candidate_k = k * CANDIDATE_POOL_MULTIPLIER

    vector_store = _get_vector_store()
    vector_results = vector_store.similarity_search(question, k=candidate_k)

    bm25 = _build_bm25_retriever()
    if bm25 is None:
        return vector_results[:k]  # empty index -- nothing for BM25 to contribute
    bm25.k = candidate_k
    bm25_results = bm25.invoke(question)

    fused = _reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:candidate_k]


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
    llm = get_llm(temperature=0.0)
    messages = [
        ("system", _RERANK_SYSTEM_PROMPT),
        ("human", f"QUESTION:\n{question}\n\nCANDIDATES:\n{numbered}"),
    ]
    response = llm.invoke(messages).content.strip()

    try:
        order = json.loads(response)
        reranked = [candidates[i - 1] for i in order if 1 <= i <= len(candidates)]
        # Guard against a malformed/partial response silently dropping
        # candidates -- fall back to the pre-rerank order for anything
        # the LLM didn't include.
        seen = set(order)
        reranked += [c for i, c in enumerate(candidates, start=1) if i not in seen]
        return reranked[:top_k]
    except (json.JSONDecodeError, ValueError, IndexError):
        # Reranking is a quality optimization, not a correctness
        # requirement -- if the LLM returns something unparseable, fall
        # back to the pre-rerank (RRF-fused) order rather than failing
        # the whole request.
        return candidates[:top_k]


def retrieve_with_hybrid_and_rerank(question: str, k: int | None = None) -> list:
    """The full pipeline: hybrid candidate retrieval -> LLM rerank -> top-k."""
    k = k or config.TOP_K
    candidates = hybrid_retrieve(question, k=k)
    return rerank(question, candidates, top_k=k)
