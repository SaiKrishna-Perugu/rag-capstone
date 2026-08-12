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
VERTEX_CHAT_MODEL = os.getenv("VERTEX_CHAT_MODEL", "gemini-2.0-flash-001")
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005")

if MODEL_PROVIDER == "vertexai" and not GCP_PROJECT_ID:
    raise RuntimeError(
        "MODEL_PROVIDER=vertexai requires GCP_PROJECT_ID to be set in .env."
    )

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
COLLECTION_NAME = "capstone_rag"

TOP_K = int(os.getenv("TOP_K", "4"))

# --- API Security & Features ----------------------------------------------
API_KEY = _get_secret("API_KEY", "")  # empty string = auth disabled
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
ENABLE_UPLOADS = os.getenv("ENABLE_UPLOADS", "true").lower() == "true"
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
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
