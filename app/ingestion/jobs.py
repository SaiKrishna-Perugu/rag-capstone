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
from app.ingestion import errors, ingest, storage

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
    error_code: str | None = None,
    warning: str | None = None,
) -> None:
    """Record a job's outcome.

    `error` is visitor-facing and must already be classified -- this record is
    returned verbatim by `/jobs/{id}` and rendered by ui.html, so a raw
    provider exception put here reaches an anonymous browser. `error_code` is
    the stable label for logs and metrics; it survives rewording of the
    message. `warning` carries partial success: the job is done, but some
    file in it is not queryable.
    """
    _jobs_collection().document(job_id).update({
        "status": status,
        "ingest_summary": ingest_summary,
        "error": error,
        "error_code": error_code,
        "warning": warning,
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
    # The staged copy too, or the bucket accumulates every file ever uploaded.
    # Best-effort by design: the chunks are already in Postgres, so a failed
    # delete costs storage rather than correctness.
    storage.delete(session_id, files)


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

    # Cloud Tasks targets the SERVICE url, which is load-balanced across
    # instances. /upload does not write to this instance's disk, so materialise
    # the job's files here first.
    #
    # With UPLOAD_BUCKET set they come from Cloud Storage, which is why this
    # works regardless of which instance the task landed on. Without it, they
    # were written locally by whichever instance served /upload -- fine on a
    # single instance, and the existence check below is what makes the
    # multi-instance case fail loudly instead of reporting success over an
    # empty ingest.
    if session_id and files:
        upload_dir = Path(config.DOCS_DIR) / "uploads" / session_id
        if storage.enabled():
            fetched = storage.fetch_to(session_id, files, upload_dir)
            logger.info(
                f"Job {job_id}: fetched {len(fetched)}/{len(files)} staged uploads."
            )
        missing = [f for f in files if not (upload_dir / f).exists()]
        if missing:
            where = (
                "They are not in the upload bucket -- staging may have failed, "
                "or they were already cleaned up by an earlier run."
                if storage.enabled() else
                "The Cloud Task was routed to an instance that did not receive "
                "them. Set UPLOAD_BUCKET so uploads no longer depend on which "
                "instance serves the request."
            )
            msg = f"Uploaded files are not present on this instance: {missing}. {where}"
            # A job that already failed keeps its original error. Cleanup runs
            # on failure, so a Cloud Tasks retry of a job that failed during
            # ingestion always lands here -- the files it is looking for were
            # deleted by its own first attempt. Overwriting made every such
            # failure read as a storage problem: a real
            # "input token count is 33360 but the model supports up to 20000"
            # was replaced, two seconds later, by this message, and that is
            # the one the visitor saw. The retry has nothing to add over the
            # attempt that actually ran.
            prior = job.get("error")
            if job.get("status") == "failed" and prior:
                logger.warning(
                    f"Job {job_id} retried after a terminal failure; keeping the "
                    f"original error. Retry would have reported: {msg}"
                )
                raise RuntimeError(prior)
            logger.error(f"Job {job_id} failed: {msg}")
            update_job_status(job_id, "failed", error=msg)
            raise RuntimeError(msg)

    try:
        summary = ingest.run(
            force=False,
            session_id=session_id,
            expires_at=expires_at,
            # The job's own file list is the ingestion allowlist. A concurrent
            # visitor's upload in the same tree is then never even considered.
            only=files or None,
        )
    except Exception as exc:
        # The full exception goes to the logs; the visitor gets a classified
        # message. `/jobs/{id}` returns this record verbatim and ui.html
        # renders `error` directly, so str(exc) here put provider URLs, the
        # project id and the upload bucket's name in front of anonymous
        # visitors -- see app/ingestion/errors.py.
        failure = errors.classify(exc)
        logger.error(
            f"Job {job_id} failed ({failure.code}): {exc}", exc_info=True
        )
        update_job_status(job_id, "failed", error=failure.message, error_code=failure.code)
        _cleanup_files(files, session_id)
        raise
    _cleanup_files(files, session_id)

    # New documents are retrievable now, so answers cached before this
    # ingest were computed without them: the visitor uploads a file, asks
    # the question it answers, and gets a pre-upload cached answer saying
    # the documents do not cover it -- with no sources, since cache hits
    # return none. cache.py already skips cache READS for a session that
    # has uploads, which covers that visitor; this covers everyone else,
    # whose cached answers are now equally out of date.
    #
    # Best-effort: the documents ARE ingested, and failing a finished job
    # over a cache flush would turn a slower next question into a reported
    # ingestion failure.
    if summary.get("added") or summary.get("updated"):
        try:
            from app.db import database
            database.invalidate_cache()
        except Exception as exc:
            logger.warning(f"Job {job_id}: cache invalidation after ingest failed: {exc}")

    # ingest.run() isolates failures per file, so it returns normally even
    # when nothing was indexed. Reporting that as "done" is the worst of the
    # outcomes available: the visitor is told their upload succeeded, then
    # asks a question about a document that was never ingested and is told
    # the documents do not cover it. Nothing anywhere says why.
    failed = summary.get("failed") or []
    indexed = (summary.get("added") or []) + (summary.get("updated") or [])
    if failed and not indexed:
        failure = errors.classify(failed[0].get("error", ""))
        logger.error(
            f"Job {job_id} indexed nothing ({failure.code}); "
            f"per-file errors: {failed}"
        )
        update_job_status(
            job_id, "failed", error=failure.message,
            error_code=failure.code, ingest_summary=summary,
        )
        raise RuntimeError(failure.message)

    if failed:
        # Partial success. The job is done -- some documents ARE queryable --
        # but the visitor must not have to infer that a file is missing from
        # its absence in a later answer.
        names = ", ".join(Path(f.get("file", "?")).name for f in failed)
        logger.warning(f"Job {job_id} completed with {len(failed)} failed file(s): {failed}")
        update_job_status(
            job_id, "done", ingest_summary=summary,
            warning=f"Some files could not be processed: {names}.",
        )
        return

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
