"""
FastAPI service exposing the RAG pipeline.

Run:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

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

from app import auth, cache, config, cost, database, jobs, memory, metrics, streaming
from app.agent import run_agentic_rag
from app.middleware import APIKeyMiddleware, IdentityMiddleware
from app.rag import answer_question

# --- Structured logging setup -------------------------------------------------
# Every request is logged as one JSON line: question, sources used, answer,
# groundedness verdict, latency. This is the "monitoring" layer -- basic on
# purpose, but it's real, queryable, and shows you thought about observability
# instead of just shipping the happy path.
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
    title="RAG Capstone API",
    description="Document Q&A over a local knowledge base using RAG.",
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)
# Registered after APIKeyMiddleware so it runs FIRST (Starlette applies
# middleware in reverse registration order). Identity is therefore resolved
# before the API-key gate, which matters only for ordering clarity -- the two
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


class AgenticAskResponse(BaseModel):
    question: str
    final_query: str          # may differ from question if the agent rewrote it
    answer: str
    groundedness: str
    sources: list[SourceChunk]
    retries_used: int
    latency_ms: int


class InternalJobRequest(BaseModel):
    job_id: str


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
    than blocking on ingest.run() (see app/jobs.py)."""
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
    docs_dir = Path(config.DOCS_DIR) / "uploads"
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

        file_path = docs_dir / safe_name
        file_path.write_bytes(content)
        saved_files.append(safe_name)

    logger.info(f"Saved uploaded files: {saved_files}. Creating ingestion job...")

    try:
        job_id = await asyncio.to_thread(jobs.create_job, saved_files)
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


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Polled by ui.html (and anyone else holding a job_id from /upload)."""
    try:
        job = await asyncio.to_thread(jobs.get_job, job_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": f"Job tracking is unavailable: {exc!s}"})
    if job is None:
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id} not found."})
    return job


@app.post("/internal/process-ingest-job")
async def process_ingest_job(body: InternalJobRequest) -> dict:
    """Cloud Tasks' HTTP target for real deployments (see
    app/jobs.py::enqueue_cloud_task) -- not used by the local/no-GCP dev
    path, which runs process_job() directly via BackgroundTasks instead
    of going through HTTP. Behind APIKeyMiddleware like every other
    non-public route; enqueue_cloud_task() sends the same X-API-Key
    header Cloud Tasks needs to get past it."""
    try:
        await asyncio.to_thread(jobs.process_job, body.job_id)
    except Exception as exc:
        # 500, not a caught-and-200 -- lets Cloud Tasks retry per the
        # queue's bounded --max-attempts policy. A later successful retry
        # just overwrites the job back to "done" (see app/jobs.py).
        raise HTTPException(status_code=500, detail={"error": str(exc)})
    return {"status": "done"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(config.RATE_LIMIT)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    request_id = str(uuid.uuid4())
    metrics.record_request("ask")
    # Begin per-request cost accumulation. Context-local, so concurrent
    # requests can't attribute each other's tokens -- see app/cost.py.
    cost.start_request()
    start = time.perf_counter()

    # --- Conversation Memory: Contextualize Question ----------------------
    contextualized_q = await asyncio.to_thread(
        memory.contextualize_question, body.session_id, body.question
    )

    # --- Check Semantic Cache First ---------------------------------------
    cached_hit = await asyncio.to_thread(cache.get_cached_answer, contextualized_q)
    if cached_hit:
        # Save to memory even if it was a cache hit
        if body.session_id:
            await asyncio.to_thread(memory.add_to_history, body.session_id, body.question, cached_hit["answer"])
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics.record_latency(latency_ms)
        logger.info(json.dumps({
            "request_id": request_id,
            "event": "ask",
            # Phase 5.2: "who asked this" is answerable now. The uid, not
            # the email -- a stable identifier without accumulating personal
            # data in logs. "anonymous" for unauthenticated visitors.
            "user": getattr(request.state, "identity", auth.ANONYMOUS).log_value,
            "cache": "HIT",
            "question": body.question,
            "answer": cached_hit["answer"],
            "similarity_score": cached_hit["similarity_score"],
            "latency_ms": latency_ms,
        }))
        return AskResponse(
            question=body.question,
            answer=cached_hit["answer"],
            groundedness=cached_hit["groundedness"],
            sources=[],  # Cached answers don't return full source chunks
            latency_ms=latency_ms,
        )

    try:
        result = await asyncio.to_thread(
            answer_question,
            question=contextualized_q,
            k=body.top_k,
            check_hallucination=body.check_hallucination,
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

    logger.info(json.dumps({
        "request_id": request_id,
        "event": "ask",
        "user": getattr(request.state, "identity", auth.ANONYMOUS).log_value,
        "question": body.question,
        "contextualized_query": contextualized_q,
        "answer": result.answer,
        "groundedness": result.groundedness,
        "num_sources": len(result.sources),
        "latency_ms": latency_ms,
        # Phase 8: what this request actually cost, broken down by pipeline
        # stage. /ask makes three LLM calls, so the breakdown is the useful
        # part -- it shows whether reranking earns its share.
        **cost.current().as_log_fields(),
    }))

    # Cache the successful result for future similar questions
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
    return StreamingResponse(
        streaming.stream_answer(body.question, body.session_id, body.top_k),
        media_type="text/event-stream"
    )

@app.post("/ask-agentic", response_model=AgenticAskResponse)
@limiter.limit(config.RATE_LIMIT)
async def ask_agentic(request: Request, body: AskRequest) -> AgenticAskResponse:
    """
    Self-correcting RAG: retrieve -> grade -> (generate | rewrite & retry) ->
    fallback if still insufficient after MAX_RETRIES. See app/agent.py.
    """
    request_id = str(uuid.uuid4())
    metrics.record_request("ask-agentic")
    start = time.perf_counter()

    # --- Conversation Memory: Contextualize Question ----------------------
    contextualized_q = await asyncio.to_thread(
        memory.contextualize_question, body.session_id, body.question
    )

    # --- Check Semantic Cache First ---------------------------------------
    cached_hit = await asyncio.to_thread(cache.get_cached_answer, contextualized_q)
    if cached_hit:
        if body.session_id:
            await asyncio.to_thread(memory.add_to_history, body.session_id, body.question, cached_hit["answer"])
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics.record_latency(latency_ms)
        logger.info(json.dumps({
            "request_id": request_id,
            "event": "ask-agentic",
            "cache": "HIT",
            "question": body.question,
            "answer": cached_hit["answer"],
            "similarity_score": cached_hit["similarity_score"],
            "latency_ms": latency_ms,
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
        final_state = await asyncio.to_thread(run_agentic_rag, contextualized_q)
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

    logger.info(json.dumps({
        "request_id": request_id,
        "event": "ask-agentic",
        "question": body.question,
        "contextualized_query": contextualized_q,
        "final_query": final_state["current_query"],
        "answer": final_state["answer"],
        "groundedness": final_state["groundedness"],
        "retries_used": final_state["retry_count"],
        "num_sources": len(final_state["sources"]),
        "latency_ms": latency_ms,
    }))

    # Cache the successful result for future similar questions
    await asyncio.to_thread(cache.set_cached_answer, contextualized_q, final_state["answer"], final_state["groundedness"])
    
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
