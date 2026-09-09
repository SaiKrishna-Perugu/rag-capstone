from unittest.mock import patch

from app import config

# /upload requires an owning session: without one the file can never be
# ingested (ingest.run()'s ownership gate refuses it), so the endpoint now
# refuses at the boundary instead of returning 202 and failing the job with
# a misleading error. Browsers always send this header; these tests are
# "anonymous" in the sense of not signed in, which is a different axis.
_SESSION = {"X-Session-Id": "test-session"}


def test_upload_without_session_header_is_refused(client):
    """The 400 that replaced a 202-then-fail. The message must name the
    header, because that is the only thing the caller can act on."""
    files = {"files": ("notes.txt", b"hello world", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "X-Session-Id" in response.json()["detail"]["error"]


def test_upload_with_malformed_session_header_is_refused(client):
    """_doc_session's allowlist returns None for a traversal attempt, and
    None must now be refused rather than silently treated as 'no session'."""
    files = {"files": ("notes.txt", b"hello world", "text/plain")}
    response = client.post("/upload", files=files, headers={"X-Session-Id": "../../app"})
    assert response.status_code == 400


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
    response = client.post("/upload", files=files, headers=_SESSION)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unsupported file type" in detail["error"]
    assert "request_id" in detail

def test_upload_endpoint_disabled(client, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_UPLOADS", False)
    files = {"files": ("test.txt", b"hello", "text/plain")}
    response = client.post("/upload", files=files, headers=_SESSION)
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
    response = client.post("/upload", files=files, headers=_SESSION)
    assert response.status_code == 400
    assert "may not contain" in response.json()["detail"]["error"]


def test_anonymous_upload_still_works(client, monkeypatch, tmp_path):
    """The load-bearing guarantee of the additive-auth design: adding
    identity must never turn a working anonymous upload into a rejection.
    If this test fails, the public demo is broken regardless of what else
    passes."""
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "")
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.ingestion.jobs.create_job", return_value="job-anon"):
        with patch("app.ingestion.jobs.process_job"):
            files = {"files": ("notes.txt", b"hello world", "text/plain")}
            response = client.post("/upload", files=files, headers=_SESSION)   # no auth headers
    assert response.status_code == 202


def test_upload_rejects_too_many_files(client, monkeypatch):
    """The batch is refused whole. The per-file size cap says nothing about
    how many files arrive, so without this a single request could carry
    hundreds of small ones into a publicly-writable demo corpus."""
    monkeypatch.setattr(config, "MAX_UPLOAD_FILES", 2)
    files = [("files", (f"doc{i}.txt", b"hello", "text/plain")) for i in range(3)]
    response = client.post("/upload", files=files, headers=_SESSION)
    assert response.status_code == 400
    assert "Too many files" in response.json()["detail"]["error"]


def test_upload_rejects_when_corpus_full(client, monkeypatch):
    """Refused before anything is written -- a full corpus must not produce
    a partial ingest."""
    monkeypatch.setattr(config, "MAX_CORPUS_CHUNKS", 10)
    monkeypatch.setattr("app.main.database.get_chunk_count", lambda: 10)
    files = {"files": ("test.txt", b"hello", "text/plain")}
    response = client.post("/upload", files=files, headers=_SESSION)
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
    with patch("app.ingestion.jobs.create_job", return_value="job-123"):
        with patch("app.ingestion.jobs.process_job"):
            files = {"files": ("test.txt", b"hello", "text/plain")}
            response = client.post("/upload", files=files, headers=_SESSION)
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
    with patch("app.ingestion.jobs.create_job", return_value="job-123") as mock_create:
        with patch("app.ingestion.jobs.process_job") as mock_process:
            files = {"files": ("test.txt", b"hello world", "text/plain")}
            response = client.post("/upload", files=files, headers=_SESSION)

    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "job-123"
    assert data["status"] == "pending"
    mock_create.assert_called_once()
    mock_process.assert_called_once_with("job-123")

def test_upload_endpoint_enqueues_cloud_task_when_gcp_configured(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "test-project")
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.ingestion.jobs.create_job", return_value="job-456"):
        with patch("app.ingestion.jobs.enqueue_cloud_task") as mock_enqueue:
            files = {"files": ("test.txt", b"hello world", "text/plain")}
            response = client.post("/upload", files=files, headers=_SESSION)

    assert response.status_code == 202
    mock_enqueue.assert_called_once_with("job-456")

def test_upload_endpoint_job_tracking_unavailable(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DOCS_DIR", str(tmp_path))
    with patch("app.ingestion.jobs.create_job", side_effect=RuntimeError("Firestore is not configured")):
        files = {"files": ("test.txt", b"hello world", "text/plain")}
        response = client.post("/upload", files=files, headers=_SESSION)
    assert response.status_code == 503

def test_get_job_status_found(client):
    """The owner must match. This test used to pass a job with NO session_id
    and no X-Session-Id header, which only succeeded because the ownership
    check was `if owner and owner != caller` -- i.e. it encoded the fail-open
    as the contract, the way an earlier test encoded an information leak."""
    job = {
        "job_id": "job-123",
        "status": "done",
        "session_id": "sess-1",
        "ingest_summary": {"added": ["a.txt"]},
    }
    with patch("app.ingestion.jobs.get_job", return_value=job):
        response = client.get("/jobs/job-123", headers={"X-Session-Id": "sess-1"})
    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_get_job_status_ownerless_job_is_refused(client):
    """A job with no owner is unreadable rather than readable by anyone.
    /upload requires a session now, so new jobs always carry one -- but
    records predating that guard live out the 48h TTL and list filenames."""
    with patch("app.ingestion.jobs.get_job", return_value={"status": "done"}):
        response = client.get("/jobs/job-123", headers={"X-Session-Id": "sess-1"})
    assert response.status_code == 404


def test_get_job_status_other_session_is_refused(client):
    with patch("app.ingestion.jobs.get_job",
               return_value={"status": "done", "session_id": "sess-owner"}):
        response = client.get("/jobs/job-123", headers={"X-Session-Id": "sess-other"})
    assert response.status_code == 404


def test_job_response_is_projected_not_the_raw_record(client):
    """The response is an explicit projection, so a field added to the
    Firestore document is not published to an anonymous browser by default."""
    job = {
        "job_id": "job-123", "status": "done", "session_id": "sess-1",
        "expires_at": "2030-01-01T00:00:00Z",
        "internal_debug_blob": "must not be published",
    }
    with patch("app.ingestion.jobs.get_job", return_value=job):
        response = client.get("/jobs/job-123", headers={"X-Session-Id": "sess-1"})
    assert response.status_code == 200
    allowed = {
        "job_id", "status", "files", "error", "error_code",
        "warning", "ingest_summary", "created_at", "updated_at",
    }
    assert set(response.json()) <= allowed
    assert "session_id" not in response.json()
    assert "internal_debug_blob" not in response.json()

def test_get_job_status_not_found(client):
    with patch("app.ingestion.jobs.get_job", return_value=None):
        response = client.get("/jobs/nonexistent")
    assert response.status_code == 404

def test_get_job_status_unavailable(client):
    with patch("app.ingestion.jobs.get_job", side_effect=RuntimeError("Firestore is not configured")):
        response = client.get("/jobs/job-123")
    assert response.status_code == 503

def test_process_ingest_job_success(client, monkeypatch):
    """Reachable only with internal credentials now. With no Tasks service
    account configured the middleware falls back to the shared API key."""
    monkeypatch.setattr(config, "API_KEY", "internal-key")
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", "")
    with patch("app.ingestion.jobs.process_job") as mock_process:
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
    with patch("app.ingestion.jobs.process_job") as mock_process:
        response = client.post("/internal/process-ingest-job", json={"job_id": "job-123"})
    assert response.status_code == 403
    mock_process.assert_not_called()


def test_internal_denies_when_nothing_is_configured(client, monkeypatch):
    """Neither OIDC nor an API key configured must mean closed, not open --
    a broken upload is a better failure than a stranger running ingestion."""
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", "")
    with patch("app.ingestion.jobs.process_job") as mock_process:
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
    with patch("app.ingestion.jobs.process_job", side_effect=RuntimeError("ingest blew up")):
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


# --- Cache-hit disambiguation ---------------------------------------------
# A cache hit returns sources: [] because only the answer was stored. Without
# the `cached` flag that is indistinguishable from "retrieval found nothing",
# which is the opposite situation.

def test_cache_hit_is_flagged_and_returns_no_sources(client):
    from unittest.mock import patch

    hit = {"answer": "Cached: 30 days.", "groundedness": "GROUNDED", "similarity_score": 0.97}
    with patch("app.retrieval.cache.session_has_uploads", return_value=False):
        with patch("app.retrieval.cache.get_cached_answer", return_value=hit):
            resp = client.post("/ask", json={"question": "refund window?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert body["sources"] == []
    assert body["answer"] == "Cached: 30 days."


def test_live_answer_is_not_flagged_as_cached(
    client, mock_cache, mock_retrieval, mock_llm_answer, mock_groundedness
):
    resp = client.post("/ask", json={"question": "What is the refund policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is False
    assert len(body["sources"]) > 0


# ---------------------------------------------------------------------------
# Auth-comparison robustness.
#
# secrets.compare_digest raises TypeError on non-ASCII str. Unhandled, that is
# a 500 on an unauthenticated path -- and on /metrics it also defeats the
# deliberate 404-not-401 concealment, because a 500 confirms the route exists
# where a 404 does not. Verified against the live service before the fix:
# `X-Admin-Key: café` returned 500 while a wrong ASCII key returned 404.
# ---------------------------------------------------------------------------

# Sent as BYTES: httpx refuses a non-ASCII str header value client-side, so
# a str payload fails before a request is ever made. Real headers arrive as
# bytes and Starlette decodes them latin-1, so the server sees a non-ASCII
# str -- and secrets.compare_digest raises TypeError on that. Built with an
# explicit byte rather than a source literal so the file stays pure ASCII.
# Verified live before the fix: 500 here where a wrong ASCII key gave 404,
# which is what defeats /metrics' 404-not-401 concealment.
_NON_ASCII = b"caf" + bytes([0xE9])


def test_non_ascii_admin_key_does_not_500(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_KEY", "correct-admin-key")
    response = client.get("/metrics", headers={"X-Admin-Key": _NON_ASCII})
    assert response.status_code == 404, (
        "a non-ASCII admin key must be rejected like any other wrong key, "
        f"not crash the handler (got {response.status_code})"
    )


def test_metrics_rejection_is_indistinguishable_across_bad_keys(client, monkeypatch):
    """The whole point of 404-not-401 is that probing cannot confirm the route.
    A different status for a malformed key hands that back."""
    monkeypatch.setattr(config, "ADMIN_KEY", "correct-admin-key")
    absent = client.get("/metrics")
    wrong = client.get("/metrics", headers={"X-Admin-Key": "wrong"})
    malformed = client.get("/metrics", headers={"X-Admin-Key": _NON_ASCII})
    assert absent.status_code == wrong.status_code == malformed.status_code == 404
    assert absent.json() == wrong.json() == malformed.json()


def test_non_ascii_api_key_does_not_500(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "correct-api-key")
    response = client.post(
        "/ask", json={"question": "hi"}, headers={"X-API-Key": _NON_ASCII}
    )
    assert response.status_code == 401


def test_non_ascii_api_key_on_internal_tier_does_not_500(client, monkeypatch):
    """The /internal fallback path compares the same way when no Tasks
    service account is configured."""
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", "")
    monkeypatch.setattr(config, "API_KEY", "correct-api-key")
    response = client.post(
        "/internal/process-ingest-job",
        json={"job_id": "x"},
        headers={"X-API-Key": _NON_ASCII},
    )
    assert response.status_code == 403


def test_internal_denied_when_audience_cannot_be_verified(client, monkeypatch):
    """TASKS_SERVICE_ACCOUNT_EMAIL set but INGEST_TARGET_URL unset means the
    token's audience cannot be checked -- google-auth SKIPS audience validation
    entirely when passed None, so a token minted for any other audience would
    replay here. Must fail closed rather than verify less.

    Signature verification is stubbed to SUCCEED with the expected identity, so
    the audience gap is the only thing left that can reject this request. A test
    that passes a garbage token would pass for the wrong reason."""
    sa = "tasks@example.iam.gserviceaccount.com"
    monkeypatch.setattr(config, "TASKS_SERVICE_ACCOUNT_EMAIL", sa)
    monkeypatch.setattr(config, "INGEST_TARGET_URL", "")
    with patch(
        "google.oauth2.id_token.verify_oauth2_token",
        return_value={"email": sa, "email_verified": True},
    ):
        response = client.post(
            "/internal/process-ingest-job",
            json={"job_id": "x"},
            headers={"Authorization": "Bearer valid.looking.token"},
        )
    assert response.status_code == 403


def test_health_does_not_resolve_identity(client):
    """Probes sit behind IdentityMiddleware, so a junk bearer token would
    otherwise trigger cert-fetch + RSA verification on the shared executor
    before any gate -- with a 1s liveness timeout behind it."""
    with patch("app.api.auth.identity_from_header") as resolve:
        response = client.get("/health", headers={"Authorization": "Bearer junk"})
    assert response.status_code == 200
    resolve.assert_not_called()
