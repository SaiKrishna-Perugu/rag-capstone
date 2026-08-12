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

def test_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "enable_uploads" in data
    assert data["model_provider"] == config.MODEL_PROVIDER
