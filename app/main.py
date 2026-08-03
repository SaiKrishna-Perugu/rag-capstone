"""
FastAPI service exposing the RAG pipeline.

Run:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import json
import logging
import time
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import config
from app.rag import answer_question
from app.agent import run_agentic_rag, MAX_RETRIES
from app import ingest

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
    """Cleanup routine to run on startup."""
    docs_dir = Path(config.DOCS_DIR)
    if docs_dir.exists():
        for file_path in docs_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("sample_"):
                try:
                    file_path.unlink()
                    logger.info(f"Deleted uploaded file on startup: {file_path.name}")
                except Exception as exc:
                    logger.warning(f"Failed to delete {file_path.name}: {exc}")
    yield


app = FastAPI(
    title="RAG Capstone API",
    description="Document Q&A over a local knowledge base using RAG.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=10)
    check_hallucination: bool = True


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
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the interactive web UI."""
    ui_path = Path(__file__).parent / "ui.html"
    return ui_path.read_text(encoding="utf-8")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accepts a file, saves it to the docs/ directory, and triggers ingestion."""
    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(exist_ok=True)
    
    file_path = docs_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    
    logger.info(f"Saved uploaded file to {file_path}. Triggering ingestion...")
    
    # Run the ingestion pipeline synchronously
    try:
        summary = ingest.run(force=False)
        return {
            "message": f"Successfully processed {file.filename}",
            "ingest_summary": summary
        }
    except Exception as exc:
        logger.error(json.dumps({"event": "error", "endpoint": "upload", "error": str(exc)}))
        raise HTTPException(status_code=500, detail=f"File saved but ingestion failed: {str(exc)}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()

    try:
        result = answer_question(
            question=request.question,
            k=request.top_k,
            check_hallucination=request.check_hallucination,
        )
    except Exception as exc:
        # Don't leak internals to the client, but log the real error.
        logger.error(json.dumps({"event": "error", "question": request.question, "error": str(exc)}))
        raise HTTPException(status_code=500, detail="Failed to generate an answer. Check server logs.")

    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(json.dumps({
        "event": "ask",
        "question": request.question,
        "answer": result.answer,
        "groundedness": result.groundedness,
        "num_sources": len(result.sources),
        "latency_ms": latency_ms,
    }))

    return AskResponse(
        question=request.question,
        answer=result.answer,
        groundedness=result.groundedness,
        sources=[SourceChunk(**s) for s in result.sources],
        latency_ms=latency_ms,
    )


@app.post("/ask-agentic", response_model=AgenticAskResponse)
def ask_agentic(request: AskRequest) -> AgenticAskResponse:
    """
    Self-correcting RAG: retrieve -> grade -> (generate | rewrite & retry) ->
    fallback if still insufficient after MAX_RETRIES. See app/agent.py.

    Same request shape as /ask, so you can call both with the same question
    and compare single-pass vs agentic behavior directly -- useful for
    demonstrating the difference in an interview.
    """
    start = time.perf_counter()

    try:
        final_state = run_agentic_rag(request.question)
    except Exception as exc:
        logger.error(json.dumps({"event": "error", "endpoint": "ask-agentic",
                                  "question": request.question, "error": str(exc)}))
        raise HTTPException(status_code=500, detail="Failed to generate an answer. Check server logs.")

    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(json.dumps({
        "event": "ask-agentic",
        "question": request.question,
        "final_query": final_state["current_query"],
        "answer": final_state["answer"],
        "groundedness": final_state["groundedness"],
        "retries_used": final_state["retry_count"],
        "num_sources": len(final_state["sources"]),
        "latency_ms": latency_ms,
    }))

    return AgenticAskResponse(
        question=request.question,
        final_query=final_state["current_query"],
        answer=final_state["answer"],
        groundedness=final_state["groundedness"],
        sources=[SourceChunk(**s) for s in final_state["sources"]],
        retries_used=final_state["retry_count"],
        latency_ms=latency_ms,
    )
