"""
Semantic caching for RAG.

Instead of just exact-match caching, we embed incoming questions and
compare against previously answered questions in the PostgreSQL database
using pgvector. If the semantic similarity is above a high threshold
(e.g., 0.95), we return the cached answer immediately, bypassing
retrieval and LLM generation entirely.
"""

import logging
import os

from app.db import database
from app.llm.providers import get_embeddings

logger = logging.getLogger(__name__)

# Threshold for semantic similarity (1.0 = exact match, 0.0 = completely different).
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.95"))


def get_cached_answer(question: str) -> dict | None:
    """
    Search for a semantically similar question in the cache.
    Returns the cached answer payload if a match > CACHE_THRESHOLD is found.
    """
    try:
        embeddings = get_embeddings()
        question_embedding = embeddings.embed_query(question)
        return database.cache_get(question_embedding, threshold=CACHE_THRESHOLD)
    except Exception:
        # The cache is a latency optimization, not a correctness requirement --
        # a transient embeddings/store failure should degrade to a cache miss,
        # not take down the whole request.
        logger.warning("Semantic cache read failed; treating as cache miss.", exc_info=True)
        return None


def set_cached_answer(question: str, answer: str, groundedness: str) -> None:
    """Store a successful Q&A pair in the semantic cache."""
    try:
        embeddings = get_embeddings()
        question_embedding = embeddings.embed_query(question)
        database.cache_set(question, answer, groundedness, question_embedding)
    except Exception:
        logger.warning("Semantic cache write failed; answer was still returned to the caller.", exc_info=True)


def session_has_uploads(session_id: str | None) -> bool:
    """Whether this visitor has documents of their own in the corpus.

    Gates the cache READ above. This cache is global and is consulted
    *before* retrieval, so a visitor who uploaded a document would otherwise
    be served a previously-cached answer built only from the curated corpus
    -- their upload silently ignored. Found by live testing: three requests
    in, an anonymous curated-only answer got cached, and every later session
    was handed it back regardless of what they had uploaded.

    Not a privacy problem -- answers grounded in private documents are never
    written here (see rag.RagResult.used_private_docs) -- but a correctness
    one, and the more visible of the two: "I uploaded a file and nothing
    changed" is what a visitor actually notices.

    Costs one COUNT against idx_chunks_session per request carrying a
    session. Fails open to "no uploads", matching this module's posture: the
    cache is a latency optimisation and a transient DB error should not cost
    an answer.
    """
    if not session_id:
        return False
    try:
        return database.get_chunk_count(session_id=session_id) > 0
    except Exception:
        logger.warning("Session upload check failed; using the shared cache.", exc_info=True)
        return False
