"""Wave 1: the three critical upload defects found by the /autoplan review.

Every one of these defects survived a fully green suite, which is the point of
this file. `tests/test_jobs.py` mocks `app.ingestion.jobs.ingest.run` -- the exact
function containing the cross-session leak -- so the leak was invisible to the
tests that looked like they covered it.
"""
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app import config

# --- T3: X-Session-Id must not be able to escape the uploads directory ------

@pytest.mark.parametrize(
    "header,expected",
    [
        pytest.param("../../app", None, id="traversal_up_two"),
        pytest.param("../..", None, id="traversal_bare"),
        pytest.param("a/b", None, id="forward_slash"),
        pytest.param("a\\b", None, id="backslash"),
        pytest.param(".", None, id="single_dot"),
        pytest.param("x" * 65, None, id="too_long"),
        pytest.param("", None, id="empty"),
        # Both shapes ui.html actually emits must survive. docSessionId() uses
        # crypto.randomUUID() in a secure context and 'sid-<base36>' otherwise,
        # and existing visitors carry whichever one their browser produced.
        pytest.param("2f1c9a4e-0b7d-4c3a-9e21-8f5a6d7c1b2e", "keep", id="uuid_v4"),
        pytest.param("sid-lz4k9x2mq1", "keep", id="ui_html_fallback"),
        pytest.param("sess-A", "keep", id="short_alnum"),
    ],
)
def test_doc_session_rejects_anything_that_could_become_a_path(header, expected):
    from app.main import _doc_session

    request = MagicMock()
    request.headers = {"X-Session-Id": header}
    result = _doc_session(request)

    if expected is None:
        assert result is None, f"{header!r} should be refused, got {result!r}"
    else:
        assert result == header, f"{header!r} is legitimate and must be preserved"


def test_upload_with_traversal_header_writes_nothing_outside_uploads(
    client, tmp_path, monkeypatch
):
    """The end-to-end version of T3.

    Before the fix this wrote into the application source directory: an anonymous
    POST with `X-Session-Id: ../../app` and a file named `ui.html` overwrote
    `app/ui.html`, which `serve_ui()` re-reads from disk on every `GET /` --
    persistent stored XSS on the public landing page.
    """
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    victim = tmp_path.parent / "victim.md"
    victim.write_text("ORIGINAL", encoding="utf-8")

    # process_job is stubbed as well as enqueue_cloud_task, because /upload
    # picks between them on config.GCP_PROJECT_ID: set (a .env locally) sends it
    # to Cloud Tasks, unset (CI) sends it to BackgroundTasks -- which TestClient
    # runs synchronously after the response, so the real job would execute and
    # reach Firestore. Patching only one branch makes the test pass or fail
    # depending on whose machine it runs on, which is exactly what happened.
    with patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"), \
         patch("app.main.database.get_chunk_count", return_value=0):
        client.post(
            "/upload",
            headers={"X-Session-Id": "../../"},
            files={"files": ("victim.md", b"PWNED", "text/markdown")},
        )

    assert victim.read_text(encoding="utf-8") == "ORIGINAL"
    # And nothing was created outside the uploads tree.
    assert not (tmp_path.parent / "PWNED").exists()


# --- T1: session tagging must never cross sessions --------------------------

def _seed(tmp_path: Path, session: str, name: str, body: str) -> Path:
    d = tmp_path / "uploads" / session
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text(body, encoding="utf-8")
    return f


def test_ingest_never_tags_another_sessions_upload(tmp_path, monkeypatch):
    """T1 -- the cross-session ownership leak.

    Two visitors' files sit in the tree at once, which `containerConcurrency: 10`
    makes ordinary rather than exotic. Running ingestion for session A must not
    touch session B's document.
    """
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    a, b = "sess-aaa", "sess-bbb"
    _seed(tmp_path, a, "alice.md", "alice private")
    _seed(tmp_path, b, "bob.md", "bob private")

    upserted = {}

    def fake_upsert(source, contents, embeddings, content_hashes, metadatas,
                    session_id=None, expires_at=None):
        upserted[Path(source).name] = session_id

    from app.ingestion import ingest

    with patch("app.ingestion.ingest.database.init_db"), \
         patch("app.ingestion.ingest.database.get_manifest", return_value={}), \
         patch("app.ingestion.ingest.database.upsert_chunks", side_effect=fake_upsert), \
         patch("app.ingestion.ingest.database.upsert_manifest_entry"), \
         patch("app.ingestion.ingest.get_embeddings") as emb:
        emb.return_value.embed_documents.side_effect = lambda cs: [[0.0] * 4 for _ in cs]
        ingest.run(session_id=a)

    assert upserted.get("alice.md") == a, "the owning session must be tagged"
    assert "bob.md" not in upserted, (
        f"bob.md was ingested by session {a}'s run -- this is the ownership leak"
    )


def test_ingest_refuses_an_upload_with_no_owning_session(tmp_path, monkeypatch):
    """The DX-voice path: a failed upload becomes a permanent public document.

    /upload writes files before creating the job. When job creation fails
    (Firestore unreachable) the files stay on disk. A later plain CLI ingest --
    which passes no session -- used to write them with session_id NULL and
    expires_at NULL: globally visible to every visitor, never expiring.
    """
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    _seed(tmp_path, "sess-orphan", "leaked.md", "private content")

    upserted = {}

    def fake_upsert(source, contents, embeddings, content_hashes, metadatas,
                    session_id=None, expires_at=None):
        upserted[Path(source).name] = session_id

    from app.ingestion import ingest

    with patch("app.ingestion.ingest.database.init_db"), \
         patch("app.ingestion.ingest.database.get_manifest", return_value={}), \
         patch("app.ingestion.ingest.database.upsert_chunks", side_effect=fake_upsert), \
         patch("app.ingestion.ingest.database.upsert_manifest_entry"), \
         patch("app.ingestion.ingest.get_embeddings") as emb:
        emb.return_value.embed_documents.side_effect = lambda cs: [[0.0] * 4 for _ in cs]
        ingest.run(session_id=None)          # the plain CLI path

    assert "leaked.md" not in upserted, (
        "an orphaned upload was ingested as curated corpus -- globally visible, "
        "never expiring"
    )


def test_ownership_gate_survives_mixed_path_separators(tmp_path, monkeypatch):
    """Regression: the ownership gate must not depend on separator style.

    _discover_files() runs glob.glob(os.path.join(DOCS_DIR, ...)), which returns
    whatever separator mix DOCS_DIR was written with -- a forward-slash DOCS_DIR
    on Windows yields "C:/x/y\\uploads\\s\\f.md". Comparing that against
    str(Path(...)) (all backslashes) with startswith() matches nothing, so every
    upload silently fell through and was written as CURATED corpus: session_id
    NULL, never expiring, visible to everyone.

    The first version of this fix had exactly that bug. The other tests here
    passed anyway, because tmp_path produces consistent separators; only a run
    against a real database exposed it. Hence this test.
    """
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path).replace("\\", "/"))
    s = "sess-sep"
    _seed(tmp_path, s, "owned.md", "owned content")

    upserted = {}

    def fake_upsert(source, contents, embeddings, content_hashes, metadatas,
                    session_id=None, expires_at=None):
        upserted[Path(source).name] = session_id

    from app.ingestion import ingest

    with patch("app.ingestion.ingest.database.init_db"), \
         patch("app.ingestion.ingest.database.get_manifest", return_value={}), \
         patch("app.ingestion.ingest.database.upsert_chunks", side_effect=fake_upsert), \
         patch("app.ingestion.ingest.database.upsert_manifest_entry"), \
         patch("app.ingestion.ingest.get_embeddings") as emb:
        emb.return_value.embed_documents.side_effect = lambda cs: [[0.0] * 4 for _ in cs]
        ingest.run(session_id=s)

    assert upserted.get("owned.md") == s, (
        "an upload was written as curated corpus because the ownership gate "
        "compared paths as strings across separator styles"
    )


def test_only_restricts_ingestion_to_the_jobs_own_files(tmp_path, monkeypatch):
    """The job's file list is the allowlist, so a concurrent upload in the same
    session directory is not even considered."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    s = "sess-shared"
    _seed(tmp_path, s, "mine.md", "mine")
    _seed(tmp_path, s, "theirs.md", "theirs")

    upserted = {}

    def fake_upsert(source, contents, embeddings, content_hashes, metadatas,
                    session_id=None, expires_at=None):
        upserted[Path(source).name] = session_id

    from app.ingestion import ingest

    with patch("app.ingestion.ingest.database.init_db"), \
         patch("app.ingestion.ingest.database.get_manifest", return_value={}), \
         patch("app.ingestion.ingest.database.upsert_chunks", side_effect=fake_upsert), \
         patch("app.ingestion.ingest.database.upsert_manifest_entry"), \
         patch("app.ingestion.ingest.get_embeddings") as emb:
        emb.return_value.embed_documents.side_effect = lambda cs: [[0.0] * 4 for _ in cs]
        ingest.run(session_id=s, only=["mine.md"])

    assert "mine.md" in upserted
    assert "theirs.md" not in upserted


# --- T2: a job must never report success for work it did not do -------------

def test_job_fails_loudly_when_its_files_are_not_on_this_instance(
    tmp_path, monkeypatch
):
    """T2 -- Cloud Tasks targets the load-balanced service URL, so the task can
    land on an instance whose local disk never held the upload. Before the fix
    ingest.run() found nothing, returned added=[], and the job was marked `done`:
    silent data loss reported as success."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    session = str(uuid.uuid4())

    from app.ingestion import jobs

    statuses = []

    with patch("app.ingestion.jobs.get_job", return_value={
                   "session_id": session, "files": ["never_arrived.md"]}), \
         patch("app.ingestion.jobs.update_job_status",
               side_effect=lambda jid, st, **kw: statuses.append((st, kw))), \
         patch("app.ingestion.jobs.ingest.run") as ran:
        with pytest.raises(RuntimeError, match="not present on this instance"):
            jobs.process_job("job-xyz")

    ran.assert_not_called()
    assert statuses[-1][0] == "failed"
    assert "done" not in [s for s, _ in statuses]


# --- 1.4: durable staging (Cloud Storage) -----------------------------------

def test_storage_is_inert_when_no_bucket_is_configured(monkeypatch):
    """UPLOAD_BUCKET unset must leave the local-disk path untouched -- that is
    what local development and this suite run on."""
    from app.ingestion import storage

    monkeypatch.setattr(config, "UPLOAD_BUCKET", "")
    assert storage.enabled() is False
    # delete() is called unconditionally from _cleanup_files; it must no-op
    # rather than trying to build a client.
    storage.delete("sess-x", ["a.md"])


def test_upload_stages_to_the_bucket_instead_of_local_disk(
    client, tmp_path, monkeypatch
):
    """With a bucket configured the bytes must not touch this instance's disk --
    that dependency is the whole reason a Cloud Task could land on an instance
    that never had the file."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)
    monkeypatch.setattr(config, "UPLOAD_BUCKET", "test-bucket")

    with patch("app.main.storage.put", return_value="uploads/s/x.md") as put, \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"), \
         patch("app.main.database.get_chunk_count", return_value=0):
        r = client.post(
            "/upload",
            headers={"X-Session-Id": "sess-stage"},
            files={"files": ("x.md", b"hello", "text/markdown")},
        )

    assert r.status_code == 202
    put.assert_called_once()
    assert not (tmp_path / "uploads" / "sess-stage" / "x.md").exists(), (
        "bytes were written to instance-local disk despite a bucket being configured"
    )


def test_upload_returns_503_when_staging_fails(client, tmp_path, monkeypatch):
    """Not fail-open. Falling back to local disk on a bucket error would restore
    the exact failure mode staging exists to remove, and do it invisibly."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)
    monkeypatch.setattr(config, "UPLOAD_BUCKET", "test-bucket")

    with patch("app.main.storage.put", side_effect=RuntimeError("bucket down")), \
         patch("app.main.database.get_chunk_count", return_value=0):
        r = client.post(
            "/upload",
            headers={"X-Session-Id": "sess-stage"},
            files={"files": ("x.md", b"hello", "text/markdown")},
        )

    assert r.status_code == 503
    assert not (tmp_path / "uploads" / "sess-stage" / "x.md").exists()


def test_process_job_fetches_staged_files_onto_this_instance(tmp_path, monkeypatch):
    """The point of 1.4: the job materialises its own files wherever it lands."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "UPLOAD_BUCKET", "test-bucket")
    session = "sess-fetch"

    def fake_fetch(sid, names, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        for n in names:
            (Path(dest) / n).write_text("fetched", encoding="utf-8")
        return list(names)

    from app.ingestion import jobs

    with patch("app.ingestion.jobs.get_job", return_value={
                   "session_id": session, "files": ["doc.md"]}), \
         patch("app.ingestion.jobs.update_job_status"), \
         patch("app.ingestion.jobs.storage.enabled", return_value=True), \
         patch("app.ingestion.jobs.storage.fetch_to", side_effect=fake_fetch) as fetch, \
         patch("app.ingestion.jobs.storage.delete"), \
         patch("app.ingestion.jobs.ingest.run", return_value={"added": ["doc.md"]}) as ran:
        jobs.process_job("job-fetch")

    fetch.assert_called_once()
    ran.assert_called_once()
    assert ran.call_args.kwargs["only"] == ["doc.md"]


def test_process_job_fails_when_staged_files_are_missing_from_the_bucket(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "UPLOAD_BUCKET", "test-bucket")

    from app.ingestion import jobs

    statuses = []
    with patch("app.ingestion.jobs.get_job", return_value={
                   "session_id": "sess-gone", "files": ["missing.md"]}), \
         patch("app.ingestion.jobs.update_job_status",
               side_effect=lambda jid, st, **kw: statuses.append(st)), \
         patch("app.ingestion.jobs.storage.enabled", return_value=True), \
         patch("app.ingestion.jobs.storage.fetch_to", return_value=[]), \
         patch("app.ingestion.jobs.ingest.run") as ran:
        with pytest.raises(RuntimeError, match="not present on this instance"):
            jobs.process_job("job-gone")

    ran.assert_not_called()
    assert statuses[-1] == "failed"


# --- bounded upload reads ---------------------------------------------------

def test_oversized_file_is_refused_without_buffering_all_of_it(client, tmp_path, monkeypatch):
    """The per-file cap used to be enforced after `await file.read()` had
    already pulled the whole part into memory. The Content-Length pre-gate
    does not cover a chunked or mis-declared upload, so the read itself has
    to be bounded."""
    from app import config

    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB", 1)
    monkeypatch.setattr(config, "MAX_UPLOAD_FILES", 3)

    oversized = b"x" * (3 * 1024 * 1024)   # 3MB against a 1MB cap
    resp = client.post(
        "/upload",
        files={"files": ("big.txt", oversized, "text/plain")},
        headers={"X-Session-Id": "sess-big"},
    )

    assert resp.status_code == 413
    assert "larger than" in resp.json()["detail"]["error"]
    # Nothing may reach disk when the file is refused.
    staged = list((tmp_path / "uploads").rglob("*")) if (tmp_path / "uploads").exists() else []
    assert [p for p in staged if p.is_file()] == []


def test_upload_reads_in_bounded_chunks():
    """Guards the mechanism, not just the outcome: a future refactor back to
    a single unbounded read would still return 413 for the test above while
    reinstating the memory exposure."""
    import inspect

    from app import main

    src = inspect.getsource(main.upload_files)
    # Strip comments first: this function's comments legitimately quote the
    # old `await file.read()` when explaining why it changed, and matching
    # those would make the assertion below fail on documentation.
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "await file.read(_UPLOAD_CHUNK_BYTES)" in code, "upload no longer reads in bounded chunks"
    assert "await file.read()" not in code, "unbounded read reintroduced"
