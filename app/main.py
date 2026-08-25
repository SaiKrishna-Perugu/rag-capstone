"""
FastAPI service exposing the RAG pipeline.

Run:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import filetype
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import config, metrics
from app.api import auth, security, streaming
from app.api.middleware import AccessControlMiddleware, IdentityMiddleware
from app.db import database
from app.ingestion import jobs, storage
from app.llm import budget, cost
from app.retrieval import cache, memory
from app.retrieval.agent import run_agentic_rag
from app.retrieval.rag import answer_question

# --- Structured logging setup -------------------------------------------------
# Every request is logged as one JSON line: question, sources used, answer,
# groundedness verdict, latency. One line per request keeps it greppable and
# parseable without a log-shipping stack; Cloud Logging ingests it as
# structured JSON automatically. This is the first thing to read when
# debugging request behaviour.
LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)
logger = logging.getLogger("rag_service")
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH / "requests.log")
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.addHandler(logging.StreamHandler())  # also print to console


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB schema, clean uploaded files. Shutdown: close DB pool."""
    # Initialise database schema (idempotent — safe on every cold start)
    database.init_db()

    # Only clear the dedicated uploads/ subdirectory, not all of docs_dir --
    # docs_dir also holds permanent, non-upload content (the sample_*
    # corpus files that ship with the repo).
    # Everything under uploads/ is, by construction, something /upload
    # wrote, so it's always safe to clear here: the vectors it produced
    # already live in the shared Postgres store, so deleting the local
    # raw file doesn't lose data, it just resets the per-instance disk.
    uploads_dir = Path(config.DOCS_DIR) / "uploads"
    if uploads_dir.exists():
        for file_path in uploads_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted uploaded file on startup: {file_path.name}")
                except Exception as exc:
                    logger.warning(f"Failed to delete {file_path.name}: {exc}")
    yield
    # Shutdown: close the database connection pool cleanly
    database.close_pool()


app = FastAPI(
    title="Grounded Document Q&A API",
    description=(
        "Retrieval-Augmented Generation over a document set: hybrid retrieval "
        "(full-text + vector, RRF-fused), LLM reranking, grounded generation, "
        "and a groundedness check on every answer."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# --- Production middleware ------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    # No cookie-based auth exists anywhere in this app -- Firebase identity
    # travels as an `Authorization: Bearer` header (api/auth.py), session
    # scoping as `X-Session-Id` (both set explicitly by ui.html's fetch()
    # calls, neither relies on the browser's credentialed-request mode). With
    # allow_credentials=True, Starlette's CORSMiddleware is spec-required to
    # echo the request's own Origin back instead of a literal "*" -- so the
    # default CORS_ORIGINS=* combined with True let any origin make
    # credentialed cross-origin requests, verified live via curl. False here
    # keeps the public demo's "*" default honest: unrestricted, but not
    # credentialed.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AccessControlMiddleware)
# Registered after AccessControlMiddleware so it runs FIRST (Starlette applies
# middleware in reverse registration order). Identity is therefore resolved
# before the access gate, which matters only for ordering clarity -- the two
# are independent, and IdentityMiddleware never rejects anything.
app.add_middleware(IdentityMiddleware)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_LENGTH, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=10)
    check_hallucination: bool = True
    session_id: str | None = Field(default=None, description="Optional session ID for conversation memory.")


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    groundedness: str
    sources: list[SourceChunk]
    latency_ms: int
    # A semantic-cache hit returns no source chunks -- only the answer was
    # stored. Without this flag an empty `sources` is indistinguishable from
    # a genuinely sourceless answer, so a caller cannot tell "served from
    # cache" from "retrieval found nothing", which are opposite situations.
    cached: bool = False


class AgenticAskResponse(BaseModel):
    question: str
    final_query: str          # may differ from question if the agent rewrote it
    answer: str
    groundedness: str
    sources: list[SourceChunk]
    retries_used: int
    latency_ms: int


# Extensions whose content carries a magic-byte signature, mapped to what
# that signature must be. Anything not listed here is a magic-less text
# format (.txt/.md/.csv/.html) and must therefore be UNdetected.
# .docx is an OOXML zip, so a bare "application/zip" is the honest floor --
# filetype cannot always distinguish it from any other zip, and rejecting on
# that would refuse legitimate documents.
_EXPECTED_MIME = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    },
}


# Session ids that are safe as both a path segment and a SQL key. The charset is
# what does the security work -- no dot, slash or backslash means no traversal.
# The 64-char cap just keeps paths sane. Matches both crypto.randomUUID() and
# ui.html's 'sid-<base36>' fallback; see _doc_session().
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _uploads_root() -> Path:
    """The directory every uploaded file must land inside.

    Resolved per call, not once at import: config.DOCS_DIR is monkeypatched by
    the test suite and settable per deployment, and an import-time constant
    silently stops matching the directory uploads actually go to -- which would
    make the containment check below reject every legitimate upload.
    """
    return (Path(config.DOCS_DIR) / "uploads").resolve()


class InternalJobRequest(BaseModel):
    job_id: str


def _doc_session(request: Request) -> str | None:
    """The visitor's document-visibility scope, from X-Session-Id.

    Deliberately NOT the same thing as AskRequest.session_id, which scopes
    conversation memory. A visitor could reasonably want chat history without
    exposing their uploads to it, and the two have different lifetimes. The
    UI generates this one once and keeps it in localStorage.

    None means "curated corpus only" -- the safe default for any caller that
    does not send the header.

    **Validated against a character allowlist, and that validation is
    load-bearing.** This value becomes a filesystem path segment in /upload
    (docs/uploads/<session>/) and a SQL visibility key in hybrid_search().
    Unvalidated, `../../app` escaped the uploads tree entirely and let an
    anonymous POST overwrite app/ui.html, which serve_ui() re-reads from disk on
    every request -- stored XSS on the public landing page from one
    unauthenticated call.

    An allowlist rather than a UUID parse, deliberately. ui.html's docSessionId()
    emits crypto.randomUUID() only in a secure context and falls back to
    'sid-<base36>' otherwise, and any visitor whose localStorage already holds
    such a value keeps sending it forever. Requiring a UUID would silently demote
    all of them to the curated corpus and make their existing uploads vanish.
    The security requirement is "cannot escape the directory", not "is a UUID":
    barring dot, forward slash and backslash satisfies it for every id shape the
    client actually emits.

    Returning None rather than raising keeps a malformed value degrading that
    visitor to the curated corpus, the same posture auth.py takes for expired
    tokens.
    """
    raw = request.headers.get("X-Session-Id")
    if not raw or not _SESSION_ID_RE.fullmatch(raw):
        return None
    return raw


def _enforce_daily_budget(request_id: str, endpoint: str) -> None:
    """Refuse the request if today's estimated LLM spend ceiling is reached.

    Called AFTER injection screening (which is free and gives a more useful
    400) but BEFORE contextualization, which is itself the first paid call.
    Refusing here means a rejected request costs nothing, rather than
    discovering the ceiling partway through generation with two calls
    already billed.

    503 rather than 429: the caller did nothing wrong and retrying sooner
    will not help -- the service is deliberately unavailable until the UTC
    day rolls over. See app/llm/budget.py for why the number is an estimate
    and why the counter is per-process.
    """
    if not budget.is_exceeded():
        return
    metrics.record_budget_exceeded()
    logger.warning(json.dumps({
        "request_id": request_id,
        "event": "budget_exceeded",
        "endpoint": endpoint,
        "spent_today_usd": round(budget.spent_today(), 6),
        "limit_usd": config.DAILY_BUDGET_USD,
    }))
    raise HTTPException(
        status_code=503,
        detail={
            "error": (
                "This demo has reached its daily usage limit and will reset at "
                "00:00 UTC. Nothing is broken -- the cap exists to keep a public "
                "demo affordable."
            ),
            "request_id": request_id,
        },
    )


@app.get("/health")
def health() -> dict:
    """Liveness probe -- checks that the FastAPI process is running and responsive."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe -- returns 200 only when the service can serve
    requests (vector store has documents loaded)."""
    try:
        count = database.get_chunk_count()
        if count == 0:
            raise HTTPException(status_code=503, detail="Vector store is empty -- no documents ingested.")
        return {"status": "ready", "chunks_indexed": count}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Not ready: {exc}")


@app.get("/config")
def get_ui_config(request: Request) -> dict:
    """Returns runtime configuration for the web UI."""
    identity = getattr(request.state, "identity", auth.ANONYMOUS)
    max_files, max_mb = auth.upload_limits(identity)
    return {
        "enable_uploads": config.ENABLE_UPLOADS,
        "model_provider": config.MODEL_PROVIDER,
        # The limits that apply to THIS caller, so the UI states the real
        # ceiling before someone picks a file rather than letting them
        # discover it via a 400/413 after waiting for an upload to fail.
        "max_upload_files": max_files,
        "max_upload_size_mb": max_mb,
        "authenticated": identity.is_authenticated,
        "user_email": identity.email,
        # Advertised so the UI can show what signing in would gain. Shown
        # even to anonymous callers -- that's the point of surfacing it.
        "authed_max_upload_files": config.MAX_UPLOAD_FILES_AUTHED,
        "authed_max_upload_size_mb": config.MAX_UPLOAD_SIZE_MB_AUTHED,
        # Public by design (Firebase web API keys are identifiers, not
        # credentials). Empty when Firebase isn't configured, which is how
        # the UI knows to hide sign-in entirely rather than render a button
        # that cannot work.
        "firebase": {
            "api_key": config.FIREBASE_WEB_API_KEY,
            "auth_domain": config.FIREBASE_AUTH_DOMAIN,
            "project_id": config.FIREBASE_PROJECT_ID if config.FIREBASE_WEB_API_KEY else "",
        },
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the interactive web UI."""
    ui_path = Path(__file__).parent / "ui.html"
    return ui_path.read_text(encoding="utf-8")


@app.get("/metrics")
def get_metrics() -> Response:
    """Prometheus text exposition format (was JSON prior to the
    OpenTelemetry migration -- see app/metrics.py)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/upload")
@limiter.limit(config.RATE_LIMIT)
async def upload_files(
    request: Request, background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)
):
    """Accepts multiple files, saves them to docs/uploads/, and hands off
    ingestion as an async job -- returns 202 + job_id immediately rather
    than blocking on ingest.run() (see app/ingestion/jobs.py)."""
    request_id = str(uuid.uuid4())
    if not config.ENABLE_UPLOADS:
        raise HTTPException(
            status_code=403,
            detail={"error": "Uploads are disabled in this environment.", "request_id": request_id},
        )

    # Signed-in callers get raised ceilings; anonymous visitors keep the
    # public defaults so the demo still works without an account.
    identity = getattr(request.state, "identity", auth.ANONYMOUS)
    max_files, max_size_mb = auth.upload_limits(identity)

    # Reject the whole batch up front rather than partway through the loop
    # below, which would otherwise leave the first N files already written
    # to disk and queued for ingestion.
    if len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Too many files ({len(files)}). Max {max_files} per upload.",
                "request_id": request_id,
            },
        )

    # Ceiling on the shared corpus, for deployments that accept uploads from
    # anonymous visitors. Checked before writing anything; a full corpus is a
    # refusal, not a partial ingest. Fails open on a database error -- the
    # cap is abuse mitigation, not a correctness invariant, and a transient
    # DB blip shouldn't reject a legitimate upload.
    doc_session = _doc_session(request)

    # Per-session ceiling, checked first: without it one visitor can consume
    # the entire global budget below and lock everyone else out of uploading.
    if doc_session and config.MAX_SESSION_CHUNKS > 0:
        try:
            mine = database.get_chunk_count(session_id=doc_session)
        except Exception as e:
            logger.warning(f"Session size check failed, allowing upload: {e}")
            mine = 0
        if mine >= config.MAX_SESSION_CHUNKS:
            raise HTTPException(
                status_code=507,
                detail={
                    "error": (
                        f"You have reached this demo's per-visitor limit "
                        f"({mine} chunks). Your documents expire automatically "
                        f"after {config.UPLOAD_TTL_HOURS}h."
                    ),
                    "request_id": request_id,
                },
            )

    if config.MAX_CORPUS_CHUNKS > 0:
        try:
            current = database.get_chunk_count()
        except Exception as e:
            logger.warning(f"Corpus size check failed, allowing upload: {e}")
            current = 0
        if current >= config.MAX_CORPUS_CHUNKS:
            raise HTTPException(
                status_code=507,
                detail={
                    "error": (
                        "The demo knowledge base is full "
                        f"({current} chunks). Uploads are paused until it is reset."
                    ),
                    "request_id": request_id,
                },
            )

    metrics.record_request("upload")
    # Uploads go in their own subdirectory, not directly in docs_dir --
    # keeps them separable from permanent/reference content also living
    # under docs_dir, so the startup cleanup above can safely clear only
    # this subdirectory. ingest.py discovers files recursively, so this
    # doesn't change what gets indexed.
    # One directory per session. Without this two visitors uploading files
    # with the same name overwrite each other on disk AND collide on the
    # chunks table's (source, content_hash) unique index, so the second
    # uploader would silently take ownership of the first one's document.
    docs_dir = Path(config.DOCS_DIR) / "uploads"
    if doc_session:
        docs_dir = docs_dir / doc_session
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []

    for file in files:
        # --- Input validation --------------------------------------------------
        # Sanitize filename to prevent path traversal (e.g., "../../etc/passwd")
        safe_name = Path(file.filename).name if file.filename else "uploaded_file"
        if not safe_name or safe_name.startswith("."):
            raise HTTPException(
                status_code=400,
                detail={"error": f"Invalid filename: {file.filename}", "request_id": request_id},
            )

        # The filename is stored as the chunk's `source` and echoed back in
        # /ask responses, where the UI renders it. Path(...).name stops
        # traversal but happily preserves HTML metacharacters, so a name
        # like `x" onmouseover="alert(1).txt` passes the traversal and
        # extension checks and becomes stored XSS for every later visitor
        # whose question retrieves that document. The UI escapes on render
        # too; this is the server half of that defence, and the half that
        # also protects any non-browser consumer of the API.
        if any(c in safe_name for c in '<>"\'&'):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Filename may not contain < > \" ' or & characters.",
                    "request_id": request_id,
                },
            )

        # Validate file extension against supported loaders
        suffix = Path(safe_name).suffix.lower()
        supported = {".pdf", ".txt", ".md", ".csv", ".html", ".htm", ".docx"}
        if suffix not in supported:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(supported))}",
                    "request_id": request_id,
                },
            )

        content = await file.read()

        # Magic-byte check. The extension whitelist above is trivially
        # defeated by renaming, and these files are parsed by document
        # loaders that were never written to be hostile-input-safe.
        #
        # The discriminator is that real text has NO signature: a renamed
        # executable, archive or image is detected and rejected, while
        # .txt/.md/.csv/.html legitimately return None. So "undetected" is
        # only acceptable for the formats that genuinely have no magic bytes.
        kind = filetype.guess(content[:2048])
        detected = kind.mime if kind else None
        allowed = _EXPECTED_MIME.get(suffix)
        if allowed is None:
            ok = detected is None          # magic-less text format
        else:
            ok = detected in allowed
        if not ok:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        f"{safe_name} does not look like a real '{suffix}' file "
                        f"(detected: {detected or 'no recognised signature'})."
                    ),
                    "request_id": request_id,
                },
            )

        # Check file size against this caller's ceiling, not the global one.
        max_bytes = max_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": f"File {safe_name} too large ({len(content) / 1024 / 1024:.1f}MB). Max: {max_size_mb}MB.",
                    "request_id": request_id,
                },
            )

        if storage.enabled():
            # Durable staging. The bytes never touch this instance's disk, so
            # the ingestion job can run anywhere. Not fail-open: a bucket error
            # is a real 503 rather than a silent fallback to the local path,
            # because that fallback is the failure mode this replaces.
            try:
                await asyncio.to_thread(storage.put, doc_session, safe_name, content)
            except Exception as exc:
                logger.error(json.dumps({
                    "request_id": request_id, "event": "upload_staging_failed",
                    "file": safe_name, "error": str(exc),
                }), exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Upload storage is unavailable. Please try again.",
                        "request_id": request_id,
                    },
                )
        else:
            # Local-disk path: development, tests, and any single-instance
            # deployment. Containment check is deliberately independent of
            # _doc_session()'s validation -- two layers, because either alone is
            # one refactor away from being the only thing standing between an
            # anonymous POST and an arbitrary file write. The filename is
            # sanitised above; every other component came from a request header.
            file_path = (docs_dir / safe_name).resolve()
            if not file_path.is_relative_to(_uploads_root()):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid upload path.",
                        "request_id": request_id,
                    },
                )
            file_path.write_bytes(content)
        saved_files.append(safe_name)

    logger.info(f"Saved uploaded files: {saved_files}. Creating ingestion job...")

    try:
        job_id = await asyncio.to_thread(jobs.create_job, saved_files, doc_session)
    except Exception as exc:
        metrics.record_error("upload")
        logger.error(
            json.dumps({"request_id": request_id, "event": "error", "endpoint": "upload", "error": str(exc)}),
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail={"error": f"Files saved but job tracking is unavailable: {exc!s}", "request_id": request_id},
        )

    if config.GCP_PROJECT_ID:
        # Real deployment: hand off to Cloud Tasks, which calls back into
        # /internal/process-ingest-job (with retries) -- survives this
        # request/instance going away.
        await asyncio.to_thread(jobs.enqueue_cloud_task, job_id)
    else:
        # Local/no-GCP dev (no official Cloud Tasks emulator exists):
        # process the job after this response is sent, via FastAPI's
        # BackgroundTasks, instead of enqueueing a real task. Same job
        # record, same GET /jobs/{id} contract either way.
        background_tasks.add_task(jobs.process_job, job_id)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "pending", "request_id": request_id},
    )


@app.get("/documents")
async def list_documents(request: Request) -> dict:
    """The caller's own uploaded documents.

    Returns an empty list rather than 404 when there is no session, so the UI
    has one code path for "new visitor" and "visitor with nothing uploaded".
    Curated docs/ files never appear -- a visitor manages their own uploads,
    not the shared sample corpus.
    """
    session = _doc_session(request)
    if not session:
        return {"documents": []}

    try:
        rows = await asyncio.to_thread(database.list_session_documents, session)
    except Exception as exc:
        logger.error(
            json.dumps({"event": "error", "endpoint": "documents", "error": str(exc)}),
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail={"error": "Could not list documents."})

    return {
        "documents": [
            {
                "name": Path(r["source"]).name,
                "chunks": r["chunks"],
                "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
            }
            for r in rows
        ]
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request) -> dict:
    """Remove one of the caller's own uploaded documents.

    Lets a visitor swap a file out instead of waiting out the TTL or
    exhausting MAX_SESSION_CHUNKS -- "I uploaded the wrong file and I'm stuck"
    is otherwise a dead end on a demo meant to be tried by a stranger.

    The client sends a FILENAME, never a source path. The actual `source` used
    for deletion is looked up among this session's own documents, so it can
    only ever be a value the database already associated with this caller --
    there is no path for a crafted input to reach the curated corpus or
    another visitor's file. database.delete_session_document() then scopes by
    session_id again as a second, independent check.
    """
    session = _doc_session(request)
    # Traversal rejected early and explicitly, even though the lookup below
    # makes it unreachable -- a clear 400 beats a confusing 404.
    if not session or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail={"error": "Document not found."})

    try:
        rows = await asyncio.to_thread(database.list_session_documents, session)
        match = next((r for r in rows if Path(r["source"]).name == filename), None)
        if match is None:
            # Same posture as /jobs/{id}: do not confirm what exists.
            raise HTTPException(status_code=404, detail={"error": "Document not found."})

        source = match["source"]
        deleted = await asyncio.to_thread(database.delete_session_document, session, source)
        # Not tidy-up: leaving the manifest row makes re-uploading the same
        # file a silent no-op, because ingest.run() would treat it as
        # unchanged and skip it. See database.delete_manifest_entry().
        await asyncio.to_thread(database.delete_manifest_entry, source)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            json.dumps({"event": "error", "endpoint": "delete-document", "error": str(exc)}),
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail={"error": "Could not remove the document."})

    logger.info(json.dumps({
        "event": "document_deleted", "filename": filename, "chunks_deleted": deleted,
    }))
    return {"deleted": filename, "chunks_deleted": deleted}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request) -> dict:
    """Polled by ui.html (and anyone else holding a job_id from /upload)."""
    try:
        job = await asyncio.to_thread(jobs.get_job, job_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": f"Job tracking is unavailable: {exc!s}"})
    if job is None:
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id} not found."})

    # Job IDs are UUIDs, but "unguessable" is not an access control. A job
    # records which files someone uploaded, so a mismatched session gets the
    # same 404 as a nonexistent job -- not a 403, which would confirm the ID
    # is real.
    owner = job.get("session_id")
    if owner and owner != _doc_session(request):
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id} not found."})
    return job


@app.post("/internal/process-ingest-job")
async def process_ingest_job(body: InternalJobRequest) -> dict:
    """Cloud Tasks' HTTP target for real deployments (see
    app/ingestion/jobs.py::enqueue_cloud_task) -- not used by the local/no-GCP dev
    path, which runs process_job() directly via BackgroundTasks instead
    of going through HTTP. Behind APIKeyMiddleware like every other
    non-public route; enqueue_cloud_task() sends the same X-API-Key
    header Cloud Tasks needs to get past it."""
    try:
        await asyncio.to_thread(jobs.process_job, body.job_id)
    except Exception as exc:
        # 500, not a caught-and-200 -- lets Cloud Tasks retry per the
        # queue's bounded --max-attempts policy. A later successful retry
        # just overwrites the job back to "done" (see app/ingestion/jobs.py).
        raise HTTPException(status_code=500, detail={"error": str(exc)})
    return {"status": "done"}


@app.post("/internal/cleanup-expired")
async def cleanup_expired() -> dict:
    """Sweep uploaded chunks past their TTL. Cloud Scheduler's target.

    Postgres has no native TTL the way Firestore does, so expired rows need
    deleting by something. Retrieval already filters on expires_at, so a
    delayed sweep costs storage, never correctness -- an expired document is
    invisible from the moment it expires, whether or not this has run.

    Behind the internal tier (see app/api/middleware.py), same as the ingest
    callback.
    """
    try:
        deleted = await asyncio.to_thread(database.delete_expired_chunks)
    except Exception as exc:
        logger.error(
            json.dumps({"event": "error", "endpoint": "cleanup-expired", "error": str(exc)}),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail={"error": str(exc)})
    logger.info(json.dumps({"event": "cleanup_expired", "chunks_deleted": deleted}))
    return {"chunks_deleted": deleted}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(config.RATE_LIMIT)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    request_id = str(uuid.uuid4())
    metrics.record_request("ask")
    # Begin per-request cost accumulation. Context-local, so concurrent
    # requests can't attribute each other's tokens -- see app/llm/cost.py.
    cost.start_request()
    start = time.perf_counter()

    # --- Prompt-injection screening ---------------------------------------
    # Deliberately first: before contextualization, before the cache, before
    # retrieval. A refused request should cost nothing, and screening a
    # rewritten question rather than the one actually typed would let the
    # rewrite launder the payload.
    verdict = security.screen_question(body.question)
    if verdict.flagged:
        metrics.record_injection_blocked(verdict.reason)
        logger.warning(json.dumps({
            "request_id": request_id, "event": "injection_blocked", "endpoint": "ask",
            "reason": verdict.reason,
            "user": getattr(request.state, "identity", auth.ANONYMOUS).log_value,
        }))
        raise HTTPException(
            status_code=400,
            detail={"error": security.REFUSAL_MESSAGE, "reason": verdict.reason,
                    "request_id": request_id},
        )

    _enforce_daily_budget(request_id, "ask")

    # --- Conversation Memory: Contextualize Question ----------------------
    contextualized_q = await asyncio.to_thread(
        memory.contextualize_question, body.session_id, body.question
    )

    # --- Check Semantic Cache First ---------------------------------------
    # Skipped for visitors with their own documents; see _session_has_uploads.
    cached_hit = None
    if not await asyncio.to_thread(cache.session_has_uploads, _doc_session(request)):
        cached_hit = await asyncio.to_thread(cache.get_cached_answer, contextualized_q)
    if cached_hit:
        # Save to memory even if it was a cache hit
        if body.session_id:
            await asyncio.to_thread(memory.add_to_history, body.session_id, body.question, cached_hit["answer"])
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics.record_latency(latency_ms)
        redacted = await security.redact_log_fields({
            "question": body.question,
            "answer": cached_hit["answer"],
        })
        logger.info(json.dumps({
            "request_id": request_id,
            "event": "ask",
            # Phase 5.2: "who asked this" is answerable now. The uid, not
            # the email -- a stable identifier without accumulating personal
            # data in logs. "anonymous" for unauthenticated visitors.
            "user": getattr(request.state, "identity", auth.ANONYMOUS).log_value,
            "cache": "HIT",
            **redacted,
            "similarity_score": cached_hit["similarity_score"],
            "latency_ms": latency_ms,
            # Not always zero: with a session_id, contextualize_question()
            # has already made an LLM call before the cache was consulted.
            **cost.current().as_log_fields(),
        }))
        return AskResponse(
            question=body.question,
            answer=cached_hit["answer"],
            groundedness=cached_hit["groundedness"],
            sources=[],  # Cached answers don't return full source chunks
            latency_ms=latency_ms,
            cached=True,
        )

    try:
        result = await asyncio.to_thread(
            answer_question,
            question=contextualized_q,
            k=body.top_k,
            check_hallucination=body.check_hallucination,
            session_id=_doc_session(request),
        )
    except Exception as exc:
        metrics.record_error("ask")
        logger.error(
            json.dumps({"request_id": request_id, "event": "error", "endpoint": "ask", "question": body.question, "error": str(exc)}),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to generate an answer. Check server logs.", "request_id": request_id},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics.record_latency(latency_ms)
    metrics.record_groundedness(result.groundedness)
    if not result.sources:
        metrics.record_empty_retrieval()

    # Output screening: catches the case input screening cannot, where the
    # payload arrived inside an ingested document rather than the question.
    answer_verdict = security.screen_answer(result.answer)
    if answer_verdict.flagged:
        metrics.record_prompt_leak()
        logger.warning(json.dumps({
            "request_id": request_id, "event": "prompt_leak_suppressed",
            "endpoint": "ask", "reason": answer_verdict.reason,
        }))
        result.answer = security.LEAKED_PROMPT_REPLACEMENT
        result.groundedness = "NOT_CHECKED"

    redacted = await security.redact_log_fields({
        "question": body.question,
        "contextualized_query": contextualized_q,
        "answer": result.answer,
    })
    logger.info(json.dumps({
        "request_id": request_id,
        "event": "ask",
        "user": getattr(request.state, "identity", auth.ANONYMOUS).log_value,
        **redacted,
        "groundedness": result.groundedness,
        "num_sources": len(result.sources),
        "latency_ms": latency_ms,
        # Phase 8: what this request actually cost, broken down by pipeline
        # stage. /ask makes three LLM calls, so the breakdown is the useful
        # part -- it shows whether reranking earns its share.
        **cost.current().as_log_fields(),
    }))

    # Cache the successful result for future similar questions -- but never
    # one grounded in a visitor's private upload. The cache is global and is
    # consulted BEFORE retrieval, so caching such an answer would replay it to
    # the next visitor asking something similar and silently undo the session
    # isolation entirely.
    if not result.used_private_docs:
        await asyncio.to_thread(cache.set_cached_answer, contextualized_q, result.answer, result.groundedness)

    if body.session_id:
        await asyncio.to_thread(memory.add_to_history, body.session_id, body.question, result.answer)

    return AskResponse(
        question=body.question,
        answer=result.answer,
        groundedness=result.groundedness,
        sources=[SourceChunk(**s) for s in result.sources],
        latency_ms=latency_ms,
    )


@app.post("/ask-stream")
@limiter.limit(config.RATE_LIMIT)
async def ask_stream(request: Request, body: AskRequest):
    """
    Streaming version of /ask. Returns Server-Sent Events (SSE).
    Clients should read the stream for `{"token": "..."}` payloads,
    followed by a `{"type": "final", ...}` payload at the end.
    """
    _enforce_daily_budget(str(uuid.uuid4()), "ask-stream")

    return StreamingResponse(
        streaming.stream_answer(
            body.question, body.session_id, body.top_k, _doc_session(request)
        ),
        media_type="text/event-stream"
    )

@app.post("/ask-agentic", response_model=AgenticAskResponse)
@limiter.limit(config.RATE_LIMIT)
async def ask_agentic(request: Request, body: AskRequest) -> AgenticAskResponse:
    """
    Self-correcting RAG: retrieve -> grade -> (generate | rewrite & retry) ->
    fallback if still insufficient after MAX_RETRIES. See app/retrieval/agent.py.
    """
    request_id = str(uuid.uuid4())
    metrics.record_request("ask-agentic")
    # Same context-local accumulator as /ask. The agentic loop makes MORE
    # LLM calls than /ask, not fewer -- grade and rewrite run per retry on
    # top of generate/groundedness -- so this is the endpoint whose cost is
    # least predictable and most worth recording.
    cost.start_request()
    start = time.perf_counter()

    # Same screening as /ask, and worth more here: the agentic loop makes
    # the most LLM calls per request, so a refused payload avoids the most.
    verdict = security.screen_question(body.question)
    if verdict.flagged:
        metrics.record_injection_blocked(verdict.reason)
        logger.warning(json.dumps({
            "request_id": request_id, "event": "injection_blocked",
            "endpoint": "ask-agentic", "reason": verdict.reason,
        }))
        raise HTTPException(
            status_code=400,
            detail={"error": security.REFUSAL_MESSAGE, "reason": verdict.reason,
                    "request_id": request_id},
        )

    _enforce_daily_budget(request_id, "ask-agentic")

    # --- Conversation Memory: Contextualize Question ----------------------
    contextualized_q = await asyncio.to_thread(
        memory.contextualize_question, body.session_id, body.question
    )

    # --- Check Semantic Cache First ---------------------------------------
    cached_hit = None
    if not await asyncio.to_thread(cache.session_has_uploads, _doc_session(request)):
        cached_hit = await asyncio.to_thread(cache.get_cached_answer, contextualized_q)
    if cached_hit:
        if body.session_id:
            await asyncio.to_thread(memory.add_to_history, body.session_id, body.question, cached_hit["answer"])
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics.record_latency(latency_ms)
        redacted = await security.redact_log_fields({
            "question": body.question,
            "answer": cached_hit["answer"],
        })
        logger.info(json.dumps({
            "request_id": request_id,
            "event": "ask-agentic",
            "cache": "HIT",
            **redacted,
            "similarity_score": cached_hit["similarity_score"],
            "latency_ms": latency_ms,
            **cost.current().as_log_fields(),
        }))
        return AgenticAskResponse(
            question=body.question,
            final_query=contextualized_q,
            answer=cached_hit["answer"],
            groundedness=cached_hit["groundedness"],
            sources=[],
            retries_used=0,
            latency_ms=latency_ms,
        )

    try:
        final_state = await asyncio.to_thread(
            run_agentic_rag, contextualized_q, _doc_session(request)
        )
    except Exception as exc:
        metrics.record_error("ask-agentic")
        logger.error(
            json.dumps({
                "request_id": request_id,
                "event": "error",
                "endpoint": "ask-agentic",
                "question": body.question,
                "error": str(exc),
            }),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to generate an answer. Check server logs.", "request_id": request_id},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics.record_latency(latency_ms)
    metrics.record_groundedness(final_state["groundedness"])
    if final_state["retry_count"] > 0:
        for _ in range(final_state["retry_count"]):
            metrics.record_agent_retry()
    if not final_state["sources"]:
        metrics.record_empty_retrieval()

    answer_verdict = security.screen_answer(final_state["answer"])
    if answer_verdict.flagged:
        metrics.record_prompt_leak()
        logger.warning(json.dumps({
            "request_id": request_id, "event": "prompt_leak_suppressed",
            "endpoint": "ask-agentic", "reason": answer_verdict.reason,
        }))
        final_state["answer"] = security.LEAKED_PROMPT_REPLACEMENT
        final_state["groundedness"] = "NOT_CHECKED"

    redacted = await security.redact_log_fields({
        "question": body.question,
        "contextualized_query": contextualized_q,
        "answer": final_state["answer"],
    })
    logger.info(json.dumps({
        "request_id": request_id,
        "event": "ask-agentic",
        **redacted,
        "final_query": final_state["current_query"],
        "groundedness": final_state["groundedness"],
        "retries_used": final_state["retry_count"],
        "num_sources": len(final_state["sources"]),
        "latency_ms": latency_ms,
        **cost.current().as_log_fields(),
    }))

    # Same private-document rule as /ask above.
    if not any(c.metadata.get("_session_id") for c in final_state.get("chunks", [])):
        await asyncio.to_thread(
            cache.set_cached_answer,
            contextualized_q, final_state["answer"], final_state["groundedness"],
        )

    if body.session_id:
        await asyncio.to_thread(memory.add_to_history, body.session_id, body.question, final_state["answer"])

    return AgenticAskResponse(
        question=body.question,
        final_query=final_state["current_query"],
        answer=final_state["answer"],
        groundedness=final_state["groundedness"],
        sources=[SourceChunk(**s) for s in final_state["sources"]],
        retries_used=final_state["retry_count"],
        latency_ms=latency_ms,
    )
