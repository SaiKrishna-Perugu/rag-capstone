"""Per-visitor TOTAL file cap (MAX_SESSION_FILES / MAX_SESSION_FILES_AUTHED).

The gap this closes: MAX_UPLOAD_FILES caps one request, so a guest could
accumulate an unbounded number of documents by repeating 3-file uploads --
observed live ("uploaded more than 6 files without signing in"). The cap
counts live chunks AND files still staged for pending jobs, so two rapid
back-to-back requests cannot race past it.
"""
from unittest.mock import patch

from app import config
from app.api.auth import ANONYMOUS, Identity, session_file_limit


def _doc_rows(session: str, names: list[str]) -> list[dict]:
    return [
        {"source": f"uploads/{session}/{n}", "chunks": 1, "expires_at": None}
        for n in names
    ]


def _upload(client, names_and_bodies, session="sess-cap"):
    files = [
        ("files", (name, body, "text/markdown")) for name, body in names_and_bodies
    ]
    return client.post("/upload", headers={"X-Session-Id": session}, files=files)


# --- unit: the floored limits ------------------------------------------------

def test_session_file_limit_floors_authed_at_anonymous():
    assert session_file_limit(ANONYMOUS) == config.MAX_SESSION_FILES


def test_session_file_limit_never_downgrades_on_sign_in(monkeypatch):
    # The inversion that bit the per-request limits once (50MB anon vs 10MB
    # authed) must not be reachable via a single careless env var here.
    monkeypatch.setattr(config, "MAX_SESSION_FILES", 10)
    monkeypatch.setattr(config, "MAX_SESSION_FILES_AUTHED", 4)
    assert session_file_limit(Identity(uid="u1")) == 10


# --- endpoint: the cap itself ------------------------------------------------

def test_guest_blocked_at_total_file_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    rows = _doc_rows("sess-cap", [f"f{i}.md" for i in range(6)])
    with patch("app.main.database.list_session_documents", return_value=rows), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=False), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("one-more.md", b"content")])

    assert resp.status_code == 507
    body = resp.json()["detail"]
    assert "per-visitor limit" in body["error"] or "per-visitor" in body["error"]
    # The refusal must point the visitor at the actual unlock action.
    assert "Sign in" in body["error"]


def test_guest_under_cap_can_upload(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    rows = _doc_rows("sess-cap", ["a.md", "b.md"])
    with patch("app.main.database.list_session_documents", return_value=rows), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=False), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("new.md", b"content")])

    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-1"


def test_same_name_reupload_does_not_count_twice(client, tmp_path, monkeypatch):
    """Replacing an existing document must not consume the cap twice --
    the batch is unioned into the owned set before comparing."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    rows = _doc_rows("sess-cap", [f"f{i}.md" for i in range(5)] + ["replace-me.md"])
    with patch("app.main.database.list_session_documents", return_value=rows), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=False), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("replace-me.md", b"updated content")])

    assert resp.status_code == 202


# --- the back-to-back race ---------------------------------------------------

def test_staged_files_close_the_rapid_request_race(client, tmp_path, monkeypatch):
    """The chunks table only knows about files whose job already ran.
    Without counting the staging area, two rapid uploads both see "zero
    documents" and jointly walk past the cap."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    staged_dir = tmp_path / "uploads" / "sess-cap"
    staged_dir.mkdir(parents=True)
    for i in range(6):
        (staged_dir / f"pending{i}.md").write_bytes(b"staged")

    with patch("app.main.database.list_session_documents", return_value=[]), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=False), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("raced.md", b"content")])

    assert resp.status_code == 507


def test_gcs_staged_files_counted_via_storage(client, tmp_path, monkeypatch):
    """Production stages uploads in Cloud Storage, which ingest.run() never
    globs -- the count has to come from the bucket, not the disk."""
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    with patch("app.main.database.list_session_documents", return_value=[]), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=True), \
         patch("app.main.storage.list_names",
               return_value={f"pending{i}.md" for i in range(6)}), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("raced.md", b"content")])

    assert resp.status_code == 507


# --- sign-in raises the cap ---------------------------------------------------

def test_signed_in_caller_gets_raised_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)

    rows = _doc_rows("sess-cap", [f"f{i}.md" for i in range(6)])
    with patch("app.main.auth.identity_from_header",
               return_value=Identity(uid="u1", email="a@b.c")), \
         patch("app.main.database.list_session_documents", return_value=rows), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=False), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("one-more.md", b"content")])

    # The same request a guest gets 507 on succeeds once signed in.
    assert resp.status_code == 202


def test_cap_can_be_disabled(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)
    monkeypatch.setattr(config, "MAX_SESSION_FILES", 0)

    rows = _doc_rows("sess-cap", [f"f{i}.md" for i in range(50)])
    with patch("app.main.database.list_session_documents", return_value=rows), \
         patch("app.main.database.get_chunk_count", return_value=0), \
         patch("app.main.storage.enabled", return_value=False), \
         patch("app.main.jobs.create_job", return_value="job-1"), \
         patch("app.main.jobs.enqueue_cloud_task"), \
         patch("app.main.jobs.process_job"):
        resp = _upload(client, [("one-more.md", b"content")])

    assert resp.status_code == 202


# --- /config advertises the caps ---------------------------------------------

def test_config_exposes_session_file_caps(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["max_session_files"] == session_file_limit(ANONYMOUS)
    assert body["authed_max_session_files"] == max(
        config.MAX_SESSION_FILES_AUTHED, config.MAX_SESSION_FILES
    )
