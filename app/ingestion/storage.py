"""
Durable staging for uploaded files, between /upload and the ingestion job.

The problem this exists to solve: /upload used to write bytes to the serving
instance's local disk and then enqueue a Cloud Task against the load-balanced
service URL. Cloud Run instances are stateless with independent disks, so the
task could land on the instance that never received the file -- and the startup
hook clears that directory anyway. `app/db/database.py` already applies exactly
this reasoning to the vector store; uploads were the one path that didn't.

With UPLOAD_BUCKET set, bytes go to Cloud Storage keyed by session and
filename, and process_job pulls its own keys down onto whichever instance it
lands on. The file stops being instance-local, so the class of bug disappears
rather than being made less likely.

**Not fail-open**, unlike cache.py and memory.py. Same reasoning as jobs.py:
staging IS the upload contract, not a latency optimisation. A missing or
unreachable bucket is a real error that /upload turns into a 503, because
silently falling back to local disk would restore the exact failure mode this
module removes -- and it would do it invisibly.

Inert with UPLOAD_BUCKET unset: enabled() is False and callers keep the
local-disk path, which is what local development and the test suite use.
"""
import logging

from app import config

logger = logging.getLogger(__name__)

_client = None


def enabled() -> bool:
    """True when durable staging is configured."""
    return bool(config.UPLOAD_BUCKET)


def _bucket():
    """The configured bucket. Raises rather than returning None -- see module
    docstring on why this path is not fail-open."""
    global _client
    if not enabled():
        raise RuntimeError("UPLOAD_BUCKET is not configured")
    if _client is None:
        from google.cloud import storage as gcs
        _client = gcs.Client(project=config.GCP_PROJECT_ID or None)
    return _client.bucket(config.UPLOAD_BUCKET)


def object_key(session_id: str | None, filename: str) -> str:
    """Where a given upload lives in the bucket.

    Mirrors the on-disk layout (uploads/<session>/<file>) so the two paths stay
    mentally interchangeable, and so a session's objects can be listed or purged
    with a single prefix.
    """
    return f"uploads/{session_id or '_nosession'}/{filename}"


def put(session_id: str | None, filename: str, content: bytes) -> str:
    """Stage one uploaded file. Returns the object key."""
    key = object_key(session_id, filename)
    _bucket().blob(key).upload_from_string(content)
    return key


def fetch_to(session_id: str | None, filenames: list[str], dest_dir) -> list[str]:
    """Download this job's files into dest_dir, returning the ones that arrived.

    Deliberately reports what it actually fetched rather than raising on the
    first miss: process_job compares the result against the job's file list and
    fails the job with the specific names that are gone, which is a more useful
    error than a stack trace naming one blob.
    """
    from pathlib import Path

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    bucket = _bucket()
    fetched = []
    for name in filenames:
        blob = bucket.blob(object_key(session_id, name))
        try:
            blob.download_to_filename(str(dest / name))
            fetched.append(name)
        except Exception as exc:
            logger.warning(f"Could not fetch staged upload {name}: {exc}")
    return fetched


def list_names(session_id: str | None) -> set[str]:
    """Filenames currently staged for this session.

    Used by /upload's per-visitor file cap to count files whose ingestion
    job hasn't finished yet -- the chunks table only knows about files whose
    job already ran, so without this two rapid back-to-back uploads race
    past the cap (both count against a database that says the visitor has
    zero documents). Mirrors the on-disk fallback in main.py's local path.
    """
    if not enabled():
        return set()
    bucket = _bucket()
    prefix = f"uploads/{session_id or '_nosession'}/"
    return {
        blob.name[len(prefix):]
        for blob in bucket.list_blobs(prefix=prefix)
        if blob.name[len(prefix):]
    }


def delete(session_id: str | None, filenames: list[str]) -> None:
    """Drop staged objects once their chunks are in Postgres.

    Best-effort: the chunks are already durable at this point, so a failed
    delete costs storage rather than correctness. Logged, never raised -- the
    job succeeded and should not be reported as failed over cleanup.
    """
    if not enabled():
        return
    bucket = _bucket()
    for name in filenames:
        try:
            bucket.blob(object_key(session_id, name)).delete()
        except Exception as exc:
            logger.warning(f"Could not delete staged upload {name}: {exc}")
