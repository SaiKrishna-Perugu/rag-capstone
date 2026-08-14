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

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import cache, config, database, ingest, memory, metrics, streaming
from app.agent import run_agentic_rag
from app.middleware import APIKeyMiddleware
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
    # docs_dir also holds permanent, non-upload content (sample_* corpus
    # files, project reference docs like PRODUCTION_READINESS_PLAN.md).
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
def get_ui_config() -> dict:
    """Returns runtime configuration for the web UI."""
    return {
        "enable_uploads": config.ENABLE_UPLOADS,
        "model_provider": config.MODEL_PROVIDER,
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
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    """Accepts multiple files, saves them to docs/uploads/, and triggers ingestion."""
    request_id = str(uuid.uuid4())
    if not config.ENABLE_UPLOADS:
        raise HTTPException(
            status_code=403,
            detail={"error": "Uploads are disabled in this environment.", "request_id": request_id},
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

        # Check file size
        max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": f"File {safe_name} too large ({len(content) / 1024 / 1024:.1f}MB). Max: {config.MAX_UPLOAD_SIZE_MB}MB.",
                    "request_id": request_id,
                },
            )

        file_path = docs_dir / safe_name
        file_path.write_bytes(content)
        saved_files.append(safe_name)

    logger.info(f"Saved uploaded files: {saved_files}. Triggering ingestion...")

    # Run the ingestion pipeline in a thread to avoid blocking the event loop
    try:
        summary = await asyncio.to_thread(ingest.run, force=False)
        return {
            "message": f"Successfully processed {len(saved_files)} files: {', '.join(saved_files)}",
            "ingest_summary": summary,
            "request_id": request_id,
        }
    except Exception as exc:
        metrics.record_error("upload")
        logger.error(
            json.dumps({"request_id": request_id, "event": "error", "endpoint": "upload", "error": str(exc)}),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"File saved but ingestion failed: {exc!s}", "request_id": request_id},
        )


@app.post("/ask", response_model=AskResponse)
@limiter.limit(config.RATE_LIMIT)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    request_id = str(uuid.uuid4())
    metrics.record_request("ask")
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
        "question": body.question,
        "contextualized_query": contextualized_q,
        "answer": result.answer,
        "groundedness": result.groundedness,
        "num_sources": len(result.sources),
        "latency_ms": latency_ms,
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
