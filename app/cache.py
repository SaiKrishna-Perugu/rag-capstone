"""
Semantic caching for RAG.

Instead of just exact-match caching, we embed incoming questions and
compare against previously answered questions. If the semantic similarity
is above a high threshold (e.g., 0.95), we return the cached answer
immediately, bypassing retrieval and LLM generation entirely.
"""

import logging
import os

from langchain_chroma import Chroma

from app import config
from app.providers import get_embeddings

logger = logging.getLogger(__name__)

# Threshold for semantic similarity (1.0 = exact match, 0.0 = completely different).
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.95"))

_cache_store = None

def _get_cache_store() -> Chroma:
    """Lazy-load the cache vector store to avoid initialization on import."""
    global _cache_store
    if _cache_store is None:
        _cache_store = Chroma(
            collection_name="semantic_cache",
            embedding_function=get_embeddings(),
            persist_directory=config.CHROMA_DIR,
        )
    return _cache_store

def get_cached_answer(question: str) -> dict | None:
    """
    Search for a semantically similar question in the cache.
    Returns the cached answer payload if a match > CACHE_THRESHOLD is found.
    """
    try:
        store = _get_cache_store()
        # We use similarity_search_with_relevance_scores to get the score directly.
        # We only need the top 1 result.
        results = store.similarity_search_with_relevance_scores(question, k=1)
    except Exception:
        # The cache is a latency optimization, not a correctness requirement --
        # a transient embeddings/store failure should degrade to a cache miss,
        # not take down the whole request.
        logger.warning("Semantic cache read failed; treating as cache miss.", exc_info=True)
        return None

    if not results:
        return None
        
    doc, score = results[0]
    
    # If the score meets our high confidence threshold, it's a hit.
    if score >= CACHE_THRESHOLD:
        return {
            "answer": doc.metadata["answer"],
            "groundedness": doc.metadata.get("groundedness", "unknown"),
            "sources": [], # We don't reconstruct the full sources list for cached hits
            "cached": True,
            "similarity_score": score
        }
        
    return None

def set_cached_answer(question: str, answer: str, groundedness: str) -> None:
    """Store a successful Q&A pair in the semantic cache."""
    try:
        store = _get_cache_store()
        store.add_texts(
            texts=[question],
            metadatas=[{
                "answer": answer,
                "groundedness": groundedness
            }]
        )
    except Exception:
        logger.warning("Semantic cache write failed; answer was still returned to the caller.", exc_info=True)
