"""
Centralized configuration. Every other module reads settings from here
instead of calling os.getenv() directly, so there's one place to change
defaults or add new settings.

Supported MODEL_PROVIDER values: "vertexai", "groq"
"""
import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Model provider switch -------------------------------------------------
# "vertexai", or "groq". See app/providers.py -- this is the
# single knob that controls which provider get_llm()/get_embeddings() build.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq").lower()

# --- Groq settings (free tier: Llama 3.3 70B) ------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
# Groq doesn't offer an embeddings API, so when MODEL_PROVIDER=groq we use
# FastEmbed (local ONNX-based embeddings, no API key, no torch dependency).
# This model is ~33MB, downloaded once on first run.
GROQ_EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

if MODEL_PROVIDER == "groq" and not GROQ_API_KEY:
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
