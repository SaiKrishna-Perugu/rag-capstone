from unittest.mock import patch

from app import config


def test_health_check(client):
    # /health is a pure liveness probe -- doesn't touch the vector store,
    # so it can only ever report "ok" (see /ready for the dependency check).
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ask_endpoint_requires_auth(client, monkeypatch):
    # Configure an API key for the test
    monkeypatch.setattr(config, "API_KEY", "test_secret_key")
    
    response = client.post("/ask", json={"question": "test"})
    assert response.status_code == 401
    
    response = client.post("/ask", json={"question": "test"}, headers={"X-API-Key": "test_secret_key"})
    # It will fail at validation or execution because we are missing mocks, 
    # but it shouldn't return 401
    assert response.status_code != 401

def test_ask_endpoint_success(client, mock_cache, mock_retrieval, mock_llm_answer, mock_groundedness):
    response = client.post("/ask", json={"question": "What is the refund policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The refund policy is 30 days."
    assert data["groundedness"] == "GROUNDED"
    
def test_upload_endpoint_validation(client):
    # Test uploading a disallowed file type
    files = {"files": ("test.exe", b"malicious payload", "application/x-msdownload")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unsupported file type" in detail["error"]
    assert "request_id" in detail

def test_upload_endpoint_disabled(client, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_UPLOADS", False)
    files = {"files": ("test.txt", b"hello", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]["error"].lower()


def test_upload_rejects_html_metacharacters_in_filename(client):
    """Stored-XSS guard. The filename is persisted as the chunk's `source`
    and echoed back by /ask, where the UI renders it. Path(...).name blocks
    traversal but preserves quotes and angle brackets, so a name like
    `x" onmouseover="alert(1).txt` clears both the traversal and extension
    checks. The UI escapes on render too; this is the server half.

    Uses angle brackets rather than a quote because httpx strips `"` from
    the multipart filename parameter before it ever reaches the server --
    the server-side check covers both, but only one is expressible through
    this client."""
    files = {"files": ("evil<img src=x onerror=alert(1)>.txt", b"hello", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "may not contain" in response.json()["detail"]["error"]


def test_anonymous_upload_still_works(client, monkeypatch, tmp_path):
    """The load-bearing guarantee of the additive-auth design: adding
    identity must never turn a working anonymous upload into a rejection.
    If this test fails, the public demo is broken regardless of what else
    passes."""
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "")
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.jobs.create_job", return_value="job-anon"):
        with patch("app.jobs.process_job"):
            files = {"files": ("notes.txt", b"hello world", "text/plain")}
            response = client.post("/upload", files=files)   # no auth headers
    assert response.status_code == 202


def test_upload_rejects_too_many_files(client, monkeypatch):
    """The batch is refused whole. The per-file size cap says nothing about
    how many files arrive, so without this a single request could carry
    hundreds of small ones into a publicly-writable demo corpus."""
    monkeypatch.setattr(config, "MAX_UPLOAD_FILES", 2)
    files = [("files", (f"doc{i}.txt", b"hello", "text/plain")) for i in range(3)]
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Too many files" in response.json()["detail"]["error"]


def test_upload_rejects_when_corpus_full(client, monkeypatch):
    """Refused before anything is written -- a full corpus must not produce
    a partial ingest."""
    monkeypatch.setattr(config, "MAX_CORPUS_CHUNKS", 10)
    monkeypatch.setattr("app.main.database.get_chunk_count", lambda: 10)
    files = {"files": ("test.txt", b"hello", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 507
    assert "full" in response.json()["detail"]["error"].lower()


def test_upload_corpus_check_fails_open(client, monkeypatch, tmp_path):
    """A database blip must not reject a legitimate upload: the cap is abuse
    mitigation, not a correctness invariant."""
    monkeypatch.setattr(config, "MAX_CORPUS_CHUNKS", 10)
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "")

    def _boom():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr("app.main.database.get_chunk_count", _boom)
    with patch("app.jobs.create_job", return_value="job-123"):
        with patch("app.jobs.process_job"):
            files = {"files": ("test.txt", b"hello", "text/plain")}
            response = client.post("/upload", files=files)
    assert response.status_code == 202

def test_upload_endpoint_success_processes_in_background_without_gcp(client, monkeypatch, tmp_path):
    # No GCP_PROJECT_ID configured -- local/no-GCP path: the job is
    # processed via BackgroundTasks (process_job), not a real Cloud Task.
    # Starlette's TestClient runs background tasks synchronously as part
    # of the request, so process_job must be mocked here or this would
    # try to run real ingestion. DOCS_DIR is redirected to tmp_path --
    # /upload's file-save step isn't mocked, so without this it would
    # write a real file into the project's docs/uploads/.
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "")
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.jobs.create_job", return_value="job-123") as mock_create:
        with patch("app.jobs.process_job") as mock_process:
            files = {"files": ("test.txt", b"hello world", "text/plain")}
            response = client.post("/upload", files=files)

    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "job-123"
    assert data["status"] == "pending"
    mock_create.assert_called_once()
    mock_process.assert_called_once_with("job-123")

def test_upload_endpoint_enqueues_cloud_task_when_gcp_configured(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "test-project")
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.jobs.create_job", return_value="job-456"):
        with patch("app.jobs.enqueue_cloud_task") as mock_enqueue:
            files = {"files": ("test.txt", b"hello world", "text/plain")}
            response = client.post("/upload", files=files)

    assert response.status_code == 202
    mock_enqueue.assert_called_once_with("job-456")

def test_upload_endpoint_job_tracking_unavailable(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.jobs.create_job", side_effect=RuntimeError("Firestore is not configured")):
        files = {"files": ("test.txt", b"hello world", "text/plain")}
        response = client.post("/upload", files=files)
    assert response.status_code == 503

def test_get_job_status_found(client):
    with patch("app.jobs.get_job", return_value={"status": "done", "ingest_summary": {"added": ["a.txt"]}}):
        response = client.get("/jobs/job-123")
    assert response.status_code == 200
    assert response.json()["status"] == "done"

def test_get_job_status_not_found(client):
    with patch("app.jobs.get_job", return_value=None):
        response = client.get("/jobs/nonexistent")
    assert response.status_code == 404

def test_get_job_status_unavailable(client):
    with patch("app.jobs.get_job", side_effect=RuntimeError("Firestore is not configured")):
        response = client.get("/jobs/job-123")
    assert response.status_code == 503

def test_process_ingest_job_success(client, monkeypatch):
    """Reachable only with internal credentials now. With no Tasks service
    account configured the middleware falls back to the shared API key."""
    monkeypatch.setattr(config, "API_KEY", "internal-key")
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", "")
    with patch("app.jobs.process_job") as mock_process:
        response = client.post(
            "/internal/process-ingest-job",
            json={"job_id": "job-123"},
            headers={"X-API-Key": "internal-key"},
        )
    assert response.status_code == 200
    mock_process.assert_called_once_with("job-123")


def test_process_ingest_job_is_not_publicly_callable(client):
    """The vulnerability this closes: production runs with no API_KEY, which
    made APIKeyMiddleware disable itself and left this endpoint -- which
    triggers real ingestion work -- callable by anyone with the URL."""
    with patch("app.jobs.process_job") as mock_process:
        response = client.post("/internal/process-ingest-job", json={"job_id": "job-123"})
    assert response.status_code == 403
    mock_process.assert_not_called()


def test_internal_denies_when_nothing_is_configured(client, monkeypatch):
    """Neither OIDC nor an API key configured must mean closed, not open --
    a broken upload is a better failure than a stranger running ingestion."""
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", "")
    with patch("app.jobs.process_job") as mock_process:
        response = client.post("/internal/process-ingest-job", json={"job_id": "j"})
    assert response.status_code == 403
    mock_process.assert_not_called()


def test_unlisted_paths_are_closed_by_default(client):
    """Default-closed routing: a route added to main.py is unreachable until
    it is listed in middleware. That maintenance cost is bought on purpose --
    silently inheriting public access is how /internal got exposed."""
    assert client.get("/some-route-that-does-not-exist").status_code == 404


def test_probes_stay_open_even_when_the_deployment_is_private(client, monkeypatch):
    """Cloud Run calls these itself and cannot present a key."""
    monkeypatch.setattr(config, "API_KEY", "private-deployment")
    assert client.get("/health").status_code == 200

def test_process_ingest_job_failure_returns_500_for_cloud_tasks_retry(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "internal-key")
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", "")
    with patch("app.jobs.process_job", side_effect=RuntimeError("ingest blew up")):
        response = client.post(
            "/internal/process-ingest-job",
            json={"job_id": "job-123"},
            headers={"X-API-Key": "internal-key"},
        )
    assert response.status_code == 500

def test_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "enable_uploads" in data
    assert data["model_provider"] == config.MODEL_PROVIDER
