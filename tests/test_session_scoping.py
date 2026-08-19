"""Session-scoped documents: one visitor's upload must be invisible to another.

Before this, every upload landed in one shared corpus, so any visitor could
change what every later visitor saw -- including leaving a prompt-injection
payload in place for the next person.

Two of these guard failures that are *silent* rather than loud, which is why
they are worth having:

* the SQL visibility predicate -- a filter dropped from one of the two CTEs
  still returns plausible results, just leaky ones;
* the shared semantic cache -- it is consulted BEFORE retrieval, so an answer
  grounded in a private upload would be replayed to the next visitor even
  though retrieval itself is correctly scoped.
"""
from unittest.mock import MagicMock, patch

import pytest

from app import config
from app.db import database
from app.ingestion import ingest


def _doc(session_id=None):
    d = MagicMock()
    d.page_content = "private content"
    d.metadata = {"source": "up.pdf", "_session_id": session_id}
    return d


# --- SQL visibility predicate --------------------------------------------

def test_visibility_predicate_is_applied_to_every_candidate_source():
    """It must appear in the vector CTE, the keyword CTE, and the final
    select. Dropping it from just one still returns results -- leaky ones."""
    captured = {}

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params
        def fetchall(self): return []

    conn = MagicMock()
    conn.cursor.return_value = Cur()
    ctx = MagicMock()
    ctx.__enter__ = lambda s: conn
    ctx.__exit__ = lambda s, *a: False

    with patch("app.db.database.get_conn", return_value=ctx):
        database.hybrid_search([0.1] * 4, "q", k=4, candidate_pool=12, session_id="sess-A")

    sql = captured["sql"]
    assert sql.count("session_id IS NULL OR session_id = %(session_id)s") == 3
    assert sql.count("expires_at IS NULL OR expires_at > now()") == 3
    assert captured["params"]["session_id"] == "sess-A"


def test_anonymous_callers_see_only_the_curated_corpus():
    """No session means session_id is None, which matches only the
    `session_id IS NULL` branch -- curated docs, nobody's uploads."""
    captured = {}

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None): captured["params"] = params
        def fetchall(self): return []

    conn = MagicMock()
    conn.cursor.return_value = Cur()
    ctx = MagicMock()
    ctx.__enter__ = lambda s: conn
    ctx.__exit__ = lambda s, *a: False

    with patch("app.db.database.get_conn", return_value=ctx):
        database.hybrid_search([0.1] * 4, "q")

    assert captured["params"]["session_id"] is None


# --- Ingestion tagging ----------------------------------------------------

def test_only_uploads_are_session_tagged_never_the_curated_corpus(tmp_path, monkeypatch):
    """ingest.run() rescans the whole docs tree on every upload. Tagging
    indiscriminately would convert the shared sample corpus into one
    visitor's private documents and hide it from everyone else."""
    docs = tmp_path / "docs"
    (docs / "uploads" / "sess-A").mkdir(parents=True)
    (docs / "curated.txt").write_text("curated corpus content", encoding="utf-8")
    (docs / "uploads" / "sess-A" / "mine.txt").write_text("private content", encoding="utf-8")

    monkeypatch.setattr(config, "DOCS_DIR", str(docs))
    calls = []

    def fake_upsert(**kwargs):
        calls.append(kwargs)
        return len(kwargs["contents"])

    embeddings = MagicMock()
    embeddings.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

    with patch("app.ingestion.ingest.database.init_db"), \
         patch("app.ingestion.ingest.database.get_manifest", return_value={}), \
         patch("app.ingestion.ingest.database.upsert_manifest_entry"), \
         patch("app.ingestion.ingest.database.upsert_chunks", side_effect=fake_upsert), \
         patch("app.ingestion.ingest.get_embeddings", return_value=embeddings):
        ingest.run(session_id="sess-A", expires_at="2099-01-01T00:00:00Z")

    by_name = {c["source"]: c for c in calls}
    curated = next(v for k, v in by_name.items() if "curated" in k)
    uploaded = next(v for k, v in by_name.items() if "mine" in k)

    assert curated["session_id"] is None and curated["expires_at"] is None
    assert uploaded["session_id"] == "sess-A"
    assert uploaded["expires_at"] == "2099-01-01T00:00:00Z"


# --- The semantic cache leak ---------------------------------------------

def test_private_answers_are_not_written_to_the_shared_cache(
    client, mock_cache, mock_llm_answer, mock_groundedness
):
    """The cache is global and is checked BEFORE retrieval, so caching an
    answer grounded in a private upload replays it to the next visitor and
    undoes the isolation entirely -- while retrieval itself still looks
    correct."""
    _, mock_set = mock_cache
    with patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_doc("sess-A")]), \
         patch("app.main.logger"):
        resp = client.post(
            "/ask", json={"question": "what is in my document?"},
            headers={"X-Session-Id": "sess-A"},
        )

    assert resp.status_code == 200
    mock_set.assert_not_called()


def test_curated_only_answers_are_still_cached(
    client, mock_cache, mock_llm_answer, mock_groundedness
):
    """The guard must not disable caching wholesale -- answers from the
    curated corpus are the common case and still belong in the cache."""
    _, mock_set = mock_cache
    with patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_doc(None)]), \
         patch("app.main.logger"):
        resp = client.post(
            "/ask", json={"question": "what is the refund policy?"},
            headers={"X-Session-Id": "sess-A"},
        )

    assert resp.status_code == 200
    mock_set.assert_called_once()


# --- Job ownership --------------------------------------------------------

def test_jobs_are_not_readable_across_sessions(client):
    """A job records which files someone uploaded. A mismatched session gets
    404, not 403 -- a 403 would confirm the ID is real."""
    job = {"status": "done", "files": ["secret.pdf"], "session_id": "sess-A"}
    with patch("app.ingestion.jobs.get_job", return_value=job):
        mine = client.get("/jobs/abc", headers={"X-Session-Id": "sess-A"})
        theirs = client.get("/jobs/abc", headers={"X-Session-Id": "sess-B"})
        anon = client.get("/jobs/abc")

    assert mine.status_code == 200
    assert theirs.status_code == 404
    assert anon.status_code == 404


# --- Cache read gating (found by live testing, not by these mocks) --------

def test_a_visitor_with_uploads_bypasses_the_shared_cache(
    client, mock_llm_answer, mock_groundedness
):
    """The cache is consulted BEFORE retrieval and is not session-aware, so a
    previously-cached curated-only answer would be handed to a visitor who had
    uploaded a document -- their upload silently ignored. Caught live: three
    requests in, an anonymous answer got cached and every later session was
    served it back."""
    hit = {"answer": "stale curated answer", "groundedness": "GROUNDED", "similarity_score": 0.99}
    with patch("app.db.database.get_chunk_count", return_value=3),          patch("app.retrieval.cache.get_cached_answer", return_value=hit) as mock_get,          patch("app.retrieval.cache.set_cached_answer"),          patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_doc("sess-A")]),          patch("app.main.logger"):
        resp = client.post(
            "/ask", json={"question": "what is in my document?"},
            headers={"X-Session-Id": "sess-A"},
        )

    assert resp.status_code == 200
    mock_get.assert_not_called()
    assert resp.json()["answer"] != "stale curated answer"


def test_visitors_without_uploads_still_get_cache_hits(
    client, mock_llm_answer, mock_groundedness
):
    """The gate must not disable the cache for everyone -- a visitor who has
    uploaded nothing can safely be served a curated-corpus answer."""
    hit = {"answer": "cached curated answer", "groundedness": "GROUNDED", "similarity_score": 0.99}
    with patch("app.db.database.get_chunk_count", return_value=0),          patch("app.retrieval.cache.get_cached_answer", return_value=hit) as mock_get,          patch("app.main.logger"):
        resp = client.post(
            "/ask", json={"question": "what is the refund policy?"},
            headers={"X-Session-Id": "sess-new"},
        )

    assert resp.status_code == 200
    mock_get.assert_called_once()
    assert resp.json()["answer"] == "cached curated answer"


def test_cache_gate_fails_open(client, mock_llm_answer, mock_groundedness):
    """A DB error checking for uploads must not cost an answer -- same
    fail-open posture as the rest of app/cache.py."""
    hit = {"answer": "cached", "groundedness": "GROUNDED", "similarity_score": 0.99}
    with patch("app.db.database.get_chunk_count", side_effect=RuntimeError("db down")),          patch("app.retrieval.cache.get_cached_answer", return_value=hit) as mock_get,          patch("app.main.logger"):
        resp = client.post(
            "/ask", json={"question": "q"}, headers={"X-Session-Id": "sess-A"},
        )

    assert resp.status_code == 200
    mock_get.assert_called_once()


# --- Document removal -----------------------------------------------------

_DOCS_A = [
    {"source": "docs/uploads/sess-A/report.pdf", "chunks": 4,
     "ingested_at": None, "expires_at": None},
    {"source": "docs/uploads/sess-A/notes.txt", "chunks": 2,
     "ingested_at": None, "expires_at": None},
]


def test_documents_lists_only_the_callers_own_uploads(client):
    with patch("app.db.database.list_session_documents", return_value=_DOCS_A) as listed:
        resp = client.get("/documents", headers={"X-Session-Id": "sess-A"})

    assert resp.status_code == 200
    assert [d["name"] for d in resp.json()["documents"]] == ["report.pdf", "notes.txt"]
    # Scoped at the query, so curated docs/ files can never appear.
    listed.assert_called_once_with("sess-A")


def test_documents_is_empty_without_a_session(client):
    """One code path for the UI: a brand-new visitor gets [], not a 404."""
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


def test_removing_a_document_also_clears_the_manifest(client):
    """The regression that would otherwise ship silently broken. ingest.run()
    skips files whose manifest hash is unchanged, so leaving the row behind
    makes re-uploading the SAME file a no-op -- the document would be
    permanently unrecoverable with no error raised anywhere."""
    with patch("app.db.database.list_session_documents", return_value=_DOCS_A), \
         patch("app.db.database.delete_session_document", return_value=4) as del_chunks, \
         patch("app.db.database.delete_manifest_entry", return_value=1) as del_manifest, \
         patch("app.main.logger"):
        resp = client.delete("/documents/report.pdf", headers={"X-Session-Id": "sess-A"})

    assert resp.status_code == 200
    assert resp.json()["chunks_deleted"] == 4
    del_chunks.assert_called_once_with("sess-A", "docs/uploads/sess-A/report.pdf")
    del_manifest.assert_called_once_with("docs/uploads/sess-A/report.pdf")


def test_cannot_remove_another_sessions_document(client):
    """B asks for a file it does not own. The lookup is scoped to B's own
    documents, so nothing matches and nothing is deleted -- 404, not 403,
    which would confirm the file exists."""
    with patch("app.db.database.list_session_documents", return_value=[]), \
         patch("app.db.database.delete_session_document") as del_chunks, \
         patch("app.db.database.delete_manifest_entry") as del_manifest:
        resp = client.delete("/documents/report.pdf", headers={"X-Session-Id": "sess-B"})

    assert resp.status_code == 404
    del_chunks.assert_not_called()
    del_manifest.assert_not_called()


@pytest.mark.parametrize("name", ["../curated.txt", "../../etc/passwd", "sub/dir.txt"])
def test_traversal_attempts_are_rejected(client, name):
    """The source used for deletion is only ever one the database already
    associated with this session, so this is belt-and-braces -- but a crafted
    filename must not even reach the lookup."""
    with patch("app.db.database.list_session_documents") as listed, \
         patch("app.db.database.delete_session_document") as del_chunks:
        resp = client.delete(f"/documents/{name}", headers={"X-Session-Id": "sess-A"})

    assert resp.status_code == 404
    del_chunks.assert_not_called()
    listed.assert_not_called()


def test_removal_requires_a_session(client):
    with patch("app.db.database.delete_session_document") as del_chunks:
        resp = client.delete("/documents/report.pdf")
    assert resp.status_code == 404
    del_chunks.assert_not_called()


# --- Corpus cap counts live chunks, not dead ones ------------------------

def test_chunk_count_excludes_expired_by_default():
    """The cap must not be consumed by documents that expired days ago and
    nobody can retrieve. Retrieval already filters on expires_at, but the
    rows persist until a sweep deletes them -- and no sweep is guaranteed to
    be configured, so counting them meant /upload could start returning 507
    over invisible data."""
    captured = {}

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None): captured["sql"] = sql
        def fetchone(self): return (0,)

    conn = MagicMock()
    conn.cursor.return_value = Cur()
    ctx = MagicMock()
    ctx.__enter__ = lambda s: conn
    ctx.__exit__ = lambda s, *a: False

    with patch("app.db.database.get_conn", return_value=ctx):
        database.get_chunk_count()
    assert "expires_at IS NULL OR expires_at > now()" in captured["sql"]

    with patch("app.db.database.get_conn", return_value=ctx):
        database.get_chunk_count(session_id="sess-A")
    assert "expires_at" in captured["sql"] and "session_id = %s" in captured["sql"]

    with patch("app.db.database.get_conn", return_value=ctx):
        database.get_chunk_count(include_expired=True)
    assert "expires_at" not in captured["sql"]
