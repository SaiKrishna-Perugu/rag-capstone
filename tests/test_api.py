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

def test_process_ingest_job_success(client):
    with patch("app.jobs.process_job") as mock_process:
        response = client.post("/internal/process-ingest-job", json={"job_id": "job-123"})
    assert response.status_code == 200
    mock_process.assert_called_once_with("job-123")

def test_process_ingest_job_failure_returns_500_for_cloud_tasks_retry(client):
    with patch("app.jobs.process_job", side_effect=RuntimeError("ingest blew up")):
        response = client.post("/internal/process-ingest-job", json={"job_id": "job-123"})
    assert response.status_code == 500

def test_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "enable_uploads" in data
    assert data["model_provider"] == config.MODEL_PROVIDER
