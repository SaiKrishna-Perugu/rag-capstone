"""
Centralized configuration. Every other module reads settings from here
instead of calling os.getenv() directly, so there's one place to change
defaults or add new settings.

Supported MODEL_PROVIDER values: "vertexai", "groq"
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Detect CI/test environments ------------------------------------------
_IS_CI = any(os.getenv(v) for v in ("CI", "GITHUB_ACTIONS", "PYTEST_CURRENT_TEST"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Model provider switch -------------------------------------------------
# "vertexai", or "groq". See app/providers.py -- this is the
# single knob that controls which provider get_llm()/get_embeddings() build.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq").lower()

def _get_secret(secret_name: str, default: str = "") -> str:
    """Fetch secret from env (local) or GCP Secret Manager (prod)."""
    val = os.getenv(secret_name)
    if val:
        return val
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.warning(f"Failed to fetch secret {secret_name} from GCP Secret Manager: {e}")
    return default

# --- Groq settings (free tier: Llama 3.3 70B) ------------------------------
GROQ_API_KEY = _get_secret("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
# Groq doesn't offer an embeddings API, so when MODEL_PROVIDER=groq we use
# FastEmbed (local ONNX-based embeddings, no API key, no torch dependency).
# This model is ~33MB, downloaded once on first run.
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_CACHE_PATH = os.getenv("FASTEMBED_CACHE_PATH", ".fastembed_cache")

if MODEL_PROVIDER == "groq" and not GROQ_API_KEY:
    if _IS_CI:
        logger.warning(
            "GROQ_API_KEY not set, but running in CI/test. "
            "Mocked tests will work; integration tests will need the secret."
        )
    else:
        raise RuntimeError(
            "MODEL_PROVIDER=groq requires GROQ_API_KEY to be set in .env. "
            "Get a free key at https://console.groq.com/keys"
        )

# --- GCP / Vertex AI settings -----------------------------------------------
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
# flash-lite over flash: roughly 3x cheaper on input and 6x on output,
# which matters because /ask makes three LLM calls per request (rerank,
# generate, groundedness check) and the RAGAS eval gate re-reads the
# retrieved context once per judged metric. Also measurably faster --
# ~3.5s vs ~8s on the same warm request -- since it does less internal
# reasoning. Grounded generation over retrieved context is an extraction
# task rather than a reasoning one, so the cheaper model is the right
# default here; override for a workload that genuinely needs more.
#
# Availability is project-specific and worth probing rather than assuming:
# "gemini-2.0-flash-001" and "gemini-2.0-flash-lite-001" both 404 on this
# project, confirmed directly against the publisher-model endpoint. The
# 2.5 series is what is actually served.
VERTEX_CHAT_MODEL = os.getenv("VERTEX_CHAT_MODEL", "gemini-2.5-flash-lite")
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005")

if MODEL_PROVIDER == "vertexai" and not GCP_PROJECT_ID:
    raise RuntimeError(
        "MODEL_PROVIDER=vertexai requires GCP_PROJECT_ID to be set in .env."
    )

DOCS_DIR = os.getenv("DOCS_DIR", "docs")

# --- Database (Cloud SQL + pgvector) ----------------------------------------
DATABASE_URL = _get_secret("DATABASE_URL", "")
DATABASE_POOL_MIN = int(os.getenv("DATABASE_POOL_MIN", "2"))
DATABASE_POOL_MAX = int(os.getenv("DATABASE_POOL_MAX", "10"))
# Defaults to whatever the selected provider actually produces: Vertex AI's
# text-embedding-005 is 768-dim, FastEmbed's bge-small-en-v1.5 is 384. This
# sets the VECTOR(N) column width at first schema creation (database.py
# init_db()), and pgvector rejects a mismatched insert outright -- so
# deriving it from MODEL_PROVIDER removes a silent footgun where switching
# provider without also remembering this value builds a table that refuses
# every write. Still overridable for a custom embedding model.
_DEFAULT_EMBEDDING_DIMENSION = "768" if MODEL_PROVIDER == "vertexai" else "384"
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", _DEFAULT_EMBEDDING_DIMENSION))

# --- Conversation memory (Firestore) ----------------------------------------
# Firestore is optional -- app/memory.py fails open (no history) when
# GCP_PROJECT_ID is unset and FIRESTORE_EMULATOR_HOST isn't either, so
# MODEL_PROVIDER=groq deployments keep working with zero GCP setup.
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "conversation_sessions")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))

# --- Async ingestion (Cloud Tasks + Firestore job tracking) -----------------
# Unlike memory/metrics, Firestore is REQUIRED here (not fail-open) -- job
# tracking is the /upload contract itself, not a latency optimization. The
# Firestore emulator (see README) covers local dev; the queue below is only
# used when GCP_PROJECT_ID is set -- otherwise app/jobs.py processes the job
# in-process immediately instead of enqueueing a real Cloud Task.
CLOUD_TASKS_QUEUE = os.getenv("CLOUD_TASKS_QUEUE", "ingest-queue")
INGEST_TARGET_URL = os.getenv("INGEST_TARGET_URL", "")
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "48"))

# --- Metrics (OpenTelemetry) -------------------------------------------------
# Prometheus export (GET /metrics) is always on and needs no GCP config.
# Cloud Monitoring push is opt-in -- requires GCP_PROJECT_ID too.
OTEL_GCP_EXPORT = os.getenv("OTEL_GCP_EXPORT", "false").lower() == "true"

TOP_K = int(os.getenv("TOP_K", "4"))

# --- API Security & Features ----------------------------------------------
API_KEY = _get_secret("API_KEY", "")  # empty string = auth disabled
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
ENABLE_UPLOADS = os.getenv("ENABLE_UPLOADS", "true").lower() == "true"
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
# Number of files accepted per /upload request. The per-file size cap
# above says nothing about how MANY files arrive, so without this a single
# request could carry hundreds of small ones.
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
# Ceiling on total indexed chunks, enforced before accepting an upload.
# 0 disables it. This exists because an openly-writable demo corpus grows
# without bound, and corpus bloat is not merely a storage cost -- retrieval
# quality degrades measurably as unrelated content crowds out the real
# answers, and every ingested chunk costs an embedding call.
MAX_CORPUS_CHUNKS = int(os.getenv("MAX_CORPUS_CHUNKS", "0"))

# --- Optional Firebase identity (app/auth.py) -------------------------------
# Additive, never a gate: signing in raises the upload ceiling below, it is
# not what grants access. With FIREBASE_PROJECT_ID unset the whole feature is
# inert -- tokens are ignored, everyone is anonymous, and the UI hides its
# sign-in button. That keeps this deployable as a public demo by default.
#
# Defaults to GCP_PROJECT_ID because a Firebase project IS a GCP project;
# the token audience is the project ID.
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "") or GCP_PROJECT_ID
# These two are safe to serve to the browser via /config. Firebase web API
# keys are public identifiers by design, not credentials -- access is
# controlled by Firebase security rules and authorized domains, not by
# keeping this string secret.
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "")
FIREBASE_AUTH_DOMAIN = os.getenv("FIREBASE_AUTH_DOMAIN", "")
# Raised ceilings for signed-in callers. Anonymous visitors keep the limits
# above, so the demo stays usable without an account.
MAX_UPLOAD_FILES_AUTHED = int(os.getenv("MAX_UPLOAD_FILES_AUTHED", "10"))
MAX_UPLOAD_SIZE_MB_AUTHED = int(os.getenv("MAX_UPLOAD_SIZE_MB_AUTHED", "10"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))

# --- LLM Resilience -------------------------------------------------------
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

# --- LangSmith tracing ---------------------------------------------------
# LangChain/LangGraph auto-trace every LLM call once these env vars are
# set -- no code changes needed for that part. The @traceable decorators
# in app/agent.py add named, granular traces for the custom logic
# (grading, rewriting) that isn't itself an LLM call LangSmith would
# otherwise group meaningfully on its own.
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-capstone")
    # LANGSMITH_API_KEY / LANGCHAIN_API_KEY is read directly from the
    # environment by the langsmith SDK -- set it in .env, not here.
