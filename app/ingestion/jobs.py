"""
Async ingestion job tracking (Firestore) + Cloud Tasks enqueueing.

Backs the "upload -> get a job ID -> poll status" flow in app/main.py:
POST /upload creates a job here and hands off the actual ingestion work
instead of running it inline in the request. Job records live in the
`ingest_jobs` Firestore collection, one document per job_id, with an
`expires_at` TTL field (same pattern as app/retrieval/memory.py's
`conversation_sessions` -- a computed future timestamp, not
firestore.SERVER_TIMESTAMP, and the TTL policy itself is a one-time
`gcloud firestore fields ttls update` call, not something this code sets).

Unlike app/retrieval/memory.py and app/retrieval/cache.py, Firestore here is NOT fail-open.
Conversation memory and the semantic cache degrade to "no history"/"cache
miss" when Firestore is unreachable because that's a pure latency/UX
optimization on top of a request that still works without it. Job
tracking IS the /upload contract -- there's nothing sensible to silently
fall back to, so a missing/unreachable Firestore surfaces as a real
exception that app/main.py turns into a 503, not a silent behavior change.

Processing itself has two paths, chosen by whether config.GCP_PROJECT_ID
is set:
  - Set (real deployment): app/main.py enqueues a Cloud Task that later
    calls POST /internal/process-ingest-job, which runs process_job().
  - Unset (local dev -- Google ships no official Cloud Tasks emulator):
    app/main.py runs process_job() itself via FastAPI's BackgroundTasks,
    so /upload still returns 202 immediately and the job-status polling
    contract is fully exercised locally, just without a real queue.
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app import config
from app.ingestion import ingest

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazily construct the Firestore client -- same pattern as
    app/retrieval/memory.py's _get_client(). Returns None if unconfigured; callers
    here (unlike memory.py) treat that as an error, not a fallback."""
    global _client
    if _client is not None:
        return _client

    if not config.GCP_PROJECT_ID and not os.getenv("FIRESTORE_EMULATOR_HOST"):
        return None

    from google.cloud import firestore
    _client = firestore.Client(project=config.GCP_PROJECT_ID or None)
    return _client


def _jobs_collection():
    client = _get_client()
    if client is None:
        raise RuntimeError(
            "Firestore is not configured (set GCP_PROJECT_ID or "
            "FIRESTORE_EMULATOR_HOST) -- job tracking requires it."
        )
    return client.collection("ingest_jobs")


def create_job(files: list[str], session_id: str | None = None) -> str:
    """Create a new job record with status=pending, return its job_id."""
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    _jobs_collection().document(job_id).set({
        "status": "pending",
        "files": files,
        # Who uploaded this. Used to scope the resulting chunks, and to stop
        # /jobs/{id} leaking one visitor's upload activity to another via an
        # enumerable ID.
        "session_id": session_id,
        "ingest_summary": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(hours=config.JOB_TTL_HOURS),
    })
    return job_id


def get_job(job_id: str) -> dict | None:
    """Return the job's current state, or None if it doesn't exist (never
    created, or TTL-expired)."""
    snap = _jobs_collection().document(job_id).get()
    return snap.to_dict() if snap.exists else None


def update_job_status(
    job_id: str,
    status: str,
    ingest_summary: dict | None = None,
    error: str | None = None,
) -> None:
    _jobs_collection().document(job_id).update({
        "status": status,
        "ingest_summary": ingest_summary,
        "error": error,
        "updated_at": datetime.now(UTC),
    })


def _cleanup_files(files: list[str], session_id: str | None) -> None:
    """Delete the raw uploads once their chunks are in Postgres.

    Deliberate, not incidental: after ingestion the raw file has no further
    purpose -- the embeddings live in the shared database, and every Cloud Run
    instance has its own ephemeral disk anyway. Not keeping user-uploaded
    content is also one less thing to secure. Runs on permanent failure too;
    a file that cannot be ingested will not ingest on a retry either.
    """
    upload_dir = Path(config.DOCS_DIR) / "uploads"
    if session_id:
        upload_dir = upload_dir / session_id
    for name in files:
        try:
            (upload_dir / name).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"Could not delete uploaded file {name}: {exc}")


def process_job(job_id: str) -> None:
    """Actually run ingestion for a job and record the outcome. Shared by
    both the local BackgroundTasks path and the real
    /internal/process-ingest-job endpoint Cloud Tasks calls, so there's
    one place that defines "what does processing a job mean" regardless
    of which path triggered it.

    Re-raises on failure (after recording it) so the caller's own error
    handling still sees it -- app/main.py's internal endpoint uses that to
    return 500 and let Cloud Tasks retry; the local BackgroundTasks path
    just logs it (the job's Firestore state is already correct either way).
    """
    job = get_job(job_id) or {}
    session_id = job.get("session_id")
    files = job.get("files", [])
    # Uploaded documents expire on their own; the curated corpus never does.
    expires_at = datetime.now(UTC) + timedelta(hours=config.UPLOAD_TTL_HOURS)

    update_job_status(job_id, "processing")
    try:
        summary = ingest.run(force=False, session_id=session_id, expires_at=expires_at)
    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
        update_job_status(job_id, "failed", error=str(exc))
        _cleanup_files(files, session_id)
        raise
    _cleanup_files(files, session_id)
    update_job_status(job_id, "done", ingest_summary=summary)


def enqueue_cloud_task(job_id: str) -> None:
    """Enqueue a real Cloud Task pointing at /internal/process-ingest-job.
    Only called when config.GCP_PROJECT_ID is set -- app/main.py's
    /upload handler runs process_job() directly via BackgroundTasks
    otherwise. Sync client + called via asyncio.to_thread at the call
    site, consistent with how ingest.run/psycopg2/firestore calls are
    already bridged elsewhere in this codebase (CloudTasksAsyncClient
    would add a grpc_asyncio transport for no benefit on one infrequent
    call per upload).
    """
    from google.cloud import tasks_v2

    client = tasks_v2.CloudTasksClient()  # lazy -- no eager auth/network call
    parent = client.queue_path(config.GCP_PROJECT_ID, config.GCP_LOCATION, config.CLOUD_TASKS_QUEUE)
    target = f"{config.INGEST_TARGET_URL}/internal/process-ingest-job"
    http_request = tasks_v2.HttpRequest(
        http_method=tasks_v2.HttpMethod.POST,
        url=target,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"job_id": job_id}).encode(),
    )

    if config.TASKS_SERVICE_ACCOUNT_EMAIL:
        # Cloud Tasks mints a signed OIDC token per task, audience-bound to
        # the target URL. app/api/middleware.py verifies both the signature and
        # the identity, which is what makes /internal/* unreachable from the
        # public internet rather than merely undocumented. The audience must
        # match what the verifier expects exactly -- both read
        # config.INGEST_TARGET_URL, so they cannot drift apart.
        http_request.oidc_token = tasks_v2.OidcToken(
            service_account_email=config.TASKS_SERVICE_ACCOUNT_EMAIL,
            audience=config.INGEST_TARGET_URL,
        )
    else:
        # Pre-migration fallback, matching middleware's. Note this is only
        # meaningful when API_KEY is set; with both unset the internal
        # endpoint denies everything and uploads will fail loudly rather
        # than the endpoint standing open.
        http_request.headers["X-API-Key"] = config.API_KEY

    task = tasks_v2.Task(http_request=http_request)
    client.create_task(request=tasks_v2.CreateTaskRequest(parent=parent, task=task))
