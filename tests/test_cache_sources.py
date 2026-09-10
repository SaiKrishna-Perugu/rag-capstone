"""Cached answers must carry their citations.

The semantic cache stored question, answer and groundedness -- but not the
sources -- so `cache_get` returned a hardcoded `"sources": []` and every cache
hit rendered as an answer with nothing behind it.

Measured against production before the fix: asking "What is the refund window
for digital products?" returned 2 citations; asking the same thing reworded
returned the identical answer with **zero**. From the UI that is
indistinguishable from retrieval having failed, which is exactly how it was
reported -- "the app is not that accurate at retrieving info". Retrieval was
correct both times; the citations were simply dropped on the way out of the
cache.

It also makes the two endpoints untestable against each other by hand: whichever
of /ask and /ask-agentic you call second hits the cache and reports no sources,
which is a trap when comparing their behaviour.
"""
from unittest.mock import patch

from app.retrieval import cache

_SOURCES = [
    {"source": "docs/sample_policy.txt", "page": None, "excerpt": "Digital products are non-refundable."},
    {"source": "docs/sample_returns_guide.md", "page": 2, "excerpt": "Returns and Exchanges Guide"},
]


def test_set_cached_answer_persists_sources():
    """The write side must hand the cards to the database, not drop them."""
    with patch("app.retrieval.cache.get_embeddings") as emb, \
         patch("app.db.database.cache_set") as db_set:
        emb.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        cache.set_cached_answer("q", "a", "GROUNDED", _SOURCES)

    assert db_set.called, "cache_set was never called"
    kwargs = db_set.call_args.kwargs
    passed = kwargs.get("sources", db_set.call_args.args[4] if len(db_set.call_args.args) > 4 else None)
    assert passed == _SOURCES, f"sources not persisted, got {passed!r}"


def test_get_cached_answer_returns_stored_sources():
    """The read side must return what was stored, not an empty list."""
    hit = {
        "answer": "Digital products are non-refundable once downloaded.",
        "groundedness": "GROUNDED",
        "sources": _SOURCES,
        "cached": True,
        "similarity_score": 0.97,
    }
    with patch("app.retrieval.cache.get_embeddings") as emb, \
         patch("app.db.database.cache_get", return_value=hit):
        emb.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        result = cache.get_cached_answer("q")

    assert result is not None
    assert result["sources"] == _SOURCES


def test_cache_hit_survives_a_missing_sources_column():
    """Entries written before the column existed have no sources.

    They must degrade to an uncited answer rather than raising -- the cache is
    fail-open everywhere else, and a KeyError here would turn every pre-existing
    row into a 500 on the first request after deploy.
    """
    legacy = {
        "answer": "a", "groundedness": "GROUNDED",
        "sources": None, "cached": True, "similarity_score": 0.99,
    }
    with patch("app.retrieval.cache.get_embeddings") as emb, \
         patch("app.db.database.cache_get", return_value=legacy):
        emb.return_value.embed_query.return_value = [0.1]
        result = cache.get_cached_answer("q")

    assert result is not None
    assert result["sources"] == []


def test_ask_returns_citations_on_a_cache_hit(client):
    """The end-to-end shape: a cache hit is a fully-formed answer, citations
    included. This is the behaviour a visitor sees."""
    hit = {
        "answer": "Digital products are non-refundable once downloaded.",
        "groundedness": "GROUNDED",
        "sources": _SOURCES,
        "cached": True,
        "similarity_score": 0.97,
    }
    with patch("app.retrieval.cache.session_has_uploads", return_value=False), \
         patch("app.retrieval.cache.get_cached_answer", return_value=hit), \
         patch("app.main.logger"):
        response = client.post("/ask", json={"question": "refund window for digital goods?"})

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is True
    assert len(body["sources"]) == 2, "a cache hit must still cite what it used"
    assert body["sources"][0]["source"] == "docs/sample_policy.txt"
    assert body["sources"][1]["page"] == 2
