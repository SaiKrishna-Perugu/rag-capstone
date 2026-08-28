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
# "vertexai", or "groq". See app/llm/providers.py -- this is the
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

# --- Groq settings (free tier: gpt-oss-20b) ---------------------------------
# Groq stopped serving any Llama chat model; the old llama-3.3-70b-versatile
# default 404s. See app/llm/cost.py's pricing table for the matching price
# row -- changing this without one silently zero-prices every call.
GROQ_API_KEY = _get_secret("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
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
# Firestore is optional -- app/retrieval/memory.py fails open (no history) when
# GCP_PROJECT_ID is unset and FIRESTORE_EMULATOR_HOST isn't either, so
# MODEL_PROVIDER=groq deployments keep working with zero GCP setup.
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "conversation_sessions")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))

# --- Async ingestion (Cloud Tasks + Firestore job tracking) -----------------
# Unlike memory/metrics, Firestore is REQUIRED here (not fail-open) -- job
# tracking is the /upload contract itself, not a latency optimization. The
# Firestore emulator (see README) covers local dev; the queue below is only
# used when GCP_PROJECT_ID is set -- otherwise app/ingestion/jobs.py processes the job
# in-process immediately instead of enqueueing a real Cloud Task.
# Cloud Storage bucket used to stage uploaded files between /upload and the
# ingestion job. Empty = disabled, and uploads fall back to instance-local disk.
#
# That fallback is correct for local development and the test suite, and WRONG
# for Cloud Run: instances have independent disks, so a Cloud Task can land on
# the instance that never received the file. Set this in any multi-instance
# deployment. See app/ingestion/storage.py.
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "")
CLOUD_TASKS_QUEUE = os.getenv("CLOUD_TASKS_QUEUE", "ingest-queue")
INGEST_TARGET_URL = os.getenv("INGEST_TARGET_URL", "")
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "48"))
# How long a visitor's uploaded documents stay retrievable. Uploads are
# scoped to the uploading session and swept by the cleanup endpoint; the
# curated docs/ corpus never expires.
UPLOAD_TTL_HOURS = int(os.getenv("UPLOAD_TTL_HOURS", "24"))
# Per-session ceiling on indexed chunks, so one visitor cannot consume the
# whole MAX_CORPUS_CHUNKS budget and lock everyone else out. 0 disables.
# 100 was too tight: at CHUNK_SIZE=800 a single moderate PDF can exceed it,
# so a visitor's first legitimate upload could be their last. Note the cap is
# checked BEFORE a batch is written, so it bounds accumulation across uploads
# rather than the size of any one upload.
MAX_SESSION_CHUNKS = int(os.getenv("MAX_SESSION_CHUNKS", "300"))

# --- Metrics (OpenTelemetry) -------------------------------------------------
# Prometheus export (GET /metrics) is always on and needs no GCP config.
# Cloud Monitoring push is opt-in -- requires GCP_PROJECT_ID too.
OTEL_GCP_EXPORT = os.getenv("OTEL_GCP_EXPORT", "false").lower() == "true"

TOP_K = int(os.getenv("TOP_K", "4"))

# --- API Security & Features ----------------------------------------------
API_KEY = _get_secret("API_KEY", "")  # empty string = auth disabled
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
# 10/minute, not 20: no legitimate demo visitor needs more, and the
# limiter is per-IP and in-process, so it is a speed bump rather than a
# real quota. See maxScale in the Cloud Run YAMLs for why that matters.
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
# Warm the embeddings/LLM clients on a background thread at startup, so the
# first real request doesn't pay for credential acquisition and the first
# connection (measured at ~20s on the live service). Off is the right
# setting for `uvicorn --reload`, where every code edit would otherwise fire
# a fresh embedding call. See main.py::_warm_providers.
ENABLE_STARTUP_WARMUP = os.getenv("ENABLE_STARTUP_WARMUP", "true").lower() == "true"
ENABLE_UPLOADS = os.getenv("ENABLE_UPLOADS", "true").lower() == "true"
# These two are the ANONYMOUS ceilings, and the defaults deliberately match
# what production runs rather than being generous. They were 50MB and 5
# files, which was wrong twice over: 50x5 is a 250MB request on an endpoint
# anonymous visitors reach, and 50 was larger than the signed-in ceiling
# below, so signing in *lowered* what you could upload. A default is what a
# deployment falls back to when its env var is missing -- which has already
# happened on this project -- so it has to be the safe value, not the
# permissive one.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "2"))
# Number of files accepted per /upload request. The per-file size cap
# above says nothing about how MANY files arrive, so without this a single
# request could carry hundreds of small ones.
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "3"))
# Ceiling on total indexed chunks, enforced before accepting an upload.
# 0 disables it. This exists because an openly-writable demo corpus grows
# without bound, and corpus bloat is not merely a storage cost -- retrieval
# quality degrades measurably as unrelated content crowds out the real
# answers, and every ingested chunk costs an embedding call.
# Counts LIVE chunks only (database.get_chunk_count excludes expired rows),
# so capacity returns on expiry whether or not the cleanup sweep ever runs.
#
# This can be generous now. The original rationale was retrieval quality --
# unrelated uploads crowding out real answers -- but session scoping means a
# visitor only ever retrieves their own documents plus the curated corpus, so
# corpus size no longer degrades anyone's results. What remains is abuse and
# cost mitigation, and at ~3KB per chunk even 3000 is under 10MB.
MAX_CORPUS_CHUNKS = int(os.getenv("MAX_CORPUS_CHUNKS", "0"))

# --- Optional Firebase identity (app/api/auth.py) -------------------------------
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
# above, so the demo stays usable without an account. These must stay >= the
# anonymous values or signing in becomes a downgrade; auth.upload_limits()
# enforces that as a floor rather than trusting whoever sets the env vars.
MAX_UPLOAD_FILES_AUTHED = int(os.getenv("MAX_UPLOAD_FILES_AUTHED", "10"))
MAX_UPLOAD_SIZE_MB_AUTHED = int(os.getenv("MAX_UPLOAD_SIZE_MB_AUTHED", "10"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))

# --- Access tiers (app/api/middleware.py) ---------------------------------------
# Separate from API_KEY on purpose. API_KEY makes a WHOLE deployment private
# (staging uses it). ADMIN_KEY gates only the operator surface -- /metrics
# leaks token counts, spend and error rates, which must not be public even
# on a demo whose whole point is being publicly usable.
ADMIN_KEY = _get_secret("ADMIN_KEY", "")

# Service account Cloud Tasks signs its OIDC tokens as. When set,
# /internal/* accepts ONLY a valid OIDC token from this identity -- the
# endpoint stops being reachable from the public internet at all. Unset
# falls back to requiring API_KEY, and if that is also unset /internal/*
# denies everything rather than standing open.
TASKS_SERVICE_ACCOUNT_EMAIL = os.getenv("TASKS_SERVICE_ACCOUNT_EMAIL", "")

# --- Security hardening (app/api/security.py) ----------------------------------
# Prompt-injection screening. Regex-only, so it costs no tokens and no
# network round-trip -- on by default because a blocked request is refused
# BEFORE retrieval, which makes it cheaper than answering, not dearer.
# Patterns are tuned for precision over recall: this is a public demo, so
# refusing a real question is worse than missing a borderline one.
ENABLE_INJECTION_SCREENING = os.getenv("ENABLE_INJECTION_SCREENING", "true").lower() == "true"

# PII redaction of logged questions/answers via Cloud DLP. OFF by default:
# it needs GCP_PROJECT_ID, the DLP API enabled, and it adds one API call to
# each /ask* request before the log line is written. Turn it on wherever
# real users type real things -- production does, local dev does not.
ENABLE_PII_REDACTION = os.getenv("ENABLE_PII_REDACTION", "false").lower() == "true"
# PERSON_NAME and STREET_ADDRESS are deliberately absent: both fire on
# ordinary product/company nouns in a support corpus, and a log where every
# other word is [PERSON_NAME] is not a log anyone will read.
# Findings below this confidence are ignored. POSSIBLE, not LIKELY: this is
# a privacy control, so over-redaction (an order number masked as an SSN) is
# a cheaper mistake than under-redaction (a real one kept for 14 days).
# Measured against the real API -- at LIKELY, IP_ADDRESS findings (which come
# back as POSSIBLE) were silently dropped.
DLP_MIN_LIKELIHOOD = os.getenv("DLP_MIN_LIKELIHOOD", "POSSIBLE").upper()
DLP_INFO_TYPES = [
    t.strip() for t in os.getenv(
        "DLP_INFO_TYPES",
        "EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD_NUMBER,US_SOCIAL_SECURITY_NUMBER,IBAN_CODE,IP_ADDRESS",
    ).split(",") if t.strip()
]

# --- Cost controls ----------------------------------------------------------
# Fraction of answered requests that also run the LLM-as-judge groundedness
# check, 0.0-1.0. Defaults to 1.0 -- every request -- and that default is
# deliberate: the verdict is one of this demo's more distinctive outputs, and
# showing "SKIPPED" on an arbitrary subset reads as unreliable rather than
# economical. Lower it where per-request cost matters more than the display;
# the check is a whole extra LLM call over the same context, so 0.25 removes
# roughly a quarter of per-request spend.
GROUNDEDNESS_SAMPLE_RATE = float(os.getenv("GROUNDEDNESS_SAMPLE_RATE", "1.0"))

# Ceiling on estimated LLM spend per UTC day, in USD. 0 disables it. Once
# exceeded, /ask* refuses with a friendly message instead of making further
# paid calls -- the point being that a runaway loop against a publicly
# advertised URL stops on its own rather than at the end of a billing cycle.
#
# Two limitations, both worth stating because they bound what this can
# promise. The figure is app/llm/cost.py's ESTIMATE from a hand-maintained
# price table, not Cloud Billing. And the counter is PER PROCESS, exactly
# like app/llm/circuit.py's breaker state, so with maxScale=2 the real
# worst case is about twice this number. This is abuse and runaway
# mitigation; the authoritative stop remains the billing budget alert.
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "0"))

# --- LLM Resilience -------------------------------------------------------
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

# Circuit breaker (app/llm/circuit.py). Always active -- failing fast on a dead
# provider is worth having on its own, with or without a fallback below.
# The threshold counts *consecutive* failures as seen outside LangChain's
# own retry, so each one is already LLM_MAX_RETRIES upstream attempts:
# the default 3 is roughly 9 real attempts before the circuit opens.
LLM_CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "3"))
LLM_CIRCUIT_COOLDOWN_SECONDS = int(os.getenv("LLM_CIRCUIT_COOLDOWN_SECONDS", "30"))

# Automatic failover: which provider get_llm() should route *chat* calls to
# while MODEL_PROVIDER's circuit is open. "groq", "vertexai", or empty to
# disable. Off by default and opt-in on purpose -- enabling it means this
# deployment must also hold working credentials for the other provider, and
# silently inferring that from a stray GROQ_API_KEY being present would be a
# surprising way to start spending money with a second vendor.
#
# Embeddings deliberately do NOT fail over; see app/llm/providers.py.
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").lower()

# --- LangSmith tracing ---------------------------------------------------
# LangChain/LangGraph auto-trace every LLM call once these env vars are
# set -- no code changes needed for that part. The @traceable decorators
# in app/retrieval/agent.py add named, granular traces for the custom logic
# (grading, rewriting) that isn't itself an LLM call LangSmith would
# otherwise group meaningfully on its own.
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-capstone")
    # LANGSMITH_API_KEY / LANGCHAIN_API_KEY is read directly from the
    # environment by the langsmith SDK -- set it in .env, not here.
