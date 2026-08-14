# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service that answers questions grounded in a local document set via
a Retrieval-Augmented Generation (RAG) pipeline: hybrid retrieval (BM25 +
vector, RRF-fused), LLM reranking, grounded generation, groundedness/
hallucination checking, a self-correcting agentic loop, semantic caching,
conversation memory, and two eval harnesses. Model/embeddings provider
(Groq or GCP Vertex AI) is switchable via a single config value.

## Commands

```bash
# Setup
uv sync --frozen
cp .env.example .env   # then set GROQ_API_KEY (or Vertex AI vars)

# Build/refresh the vector index (incremental by default -- content-hashed,
# unchanged files skipped; see app/ingest.py)
uv run python -m app.ingest
uv run python -m app.ingest --force   # full re-embed, e.g. after switching MODEL_PROVIDER

# Run the API (http://127.0.0.1:8000/docs for Swagger UI)
uv run uvicorn app.main:app --reload

# Lint
uv run ruff check .

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_rag.py -v                    # single file
uv run pytest tests/test_rag.py::test_generate_answer  # single test

# Eval harnesses (require a populated Postgres vector store -- DATABASE_URL -- and live API access -- not mocked)
uv run python eval.py          # custom LLM-as-judge: correctness + groundedness
uv run python eval_ragas.py    # RAGAS: faithfulness, relevancy, context precision/recall
```

CI (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest tests/ -v` on
every push/PR to `main`, via `uv sync --frozen`.

## Architecture

Everything routes through `app/config.py` for settings and `app/providers.py`
for model access — no module calls `os.getenv()` or instantiates an LLM/
embeddings client directly outside these two files. Switching
`MODEL_PROVIDER` between `groq` and `vertexai` in `.env` is the only change
needed to swap providers; **the vector store is provider-specific**, so a
switch requires `uv run python -m app.ingest --force` (mixing embedding
spaces from different providers in the same Postgres `chunks` table
silently produces bad retrieval, not an error — unless the embedding
*dimension* also changed, e.g. Groq's 384-dim FastEmbed vs. Vertex AI's
768-dim `text-embedding-005`, in which case pgvector rejects the insert
outright rather than silently corrupting retrieval; see `EMBEDDING_DIMENSION`
below and the vector store paragraph).

The vector store is PostgreSQL + pgvector (`app/database.py`,
`app/db_schema.sql`), not a local/embedded store — this is deliberate:
Cloud Run instances are stateless with independent local disks, so a local
vector store would mean every instance sees a different, divergent copy of
the index. Postgres's `tsvector`/`tsquery` full-text search also means
hybrid retrieval (`app/retrieval.py` `hybrid_retrieve()` →
`database.hybrid_search()`) runs as a single SQL query doing vector search
+ full-text search + RRF fusion, rather than a separately maintained BM25
index. Schema creation (`database.init_db()`) is idempotent
(`CREATE ... IF NOT EXISTS` throughout) and runs both from `main.py`'s
FastAPI `lifespan` startup and from `ingest.py`'s `run()` — the latter
matters because ingestion is commonly the first command run against a
brand-new database, before the API has ever started. `EMBEDDING_DIMENSION`
(config default `384`) sets the `VECTOR(N)` column width at *first*
creation only — it does not widen an existing column, so changing it after
the schema already exists requires dropping and letting `init_db()`
recreate the `chunks`/`semantic_cache` tables, then re-ingesting.

Groq has no embeddings API, so `groq` mode uses local FastEmbed (ONNX,
no API key, no torch) for embeddings while still using Groq for chat.
FastEmbed's cache location is `FASTEMBED_CACHE_PATH` (config default:
`.fastembed_cache`, overridden to `/app/.fastembed_cache` in the Docker
image) — the Dockerfile pre-downloads the model into that path at build
time so containers never hit Hugging Face's API at runtime (its default
cache is `/tmp`, which doesn't persist across instances anyway, and Cloud
Run's shared outbound IPs routinely hit HF's anonymous rate limit).

**Request flow for `/ask` and `/ask-stream`** (`app/main.py` → `app/rag.py`):
1. `app/memory.py` rewrites the question using conversation history if a
   `session_id` is given (`contextualize_question`).
2. `app/cache.py` checks the semantic cache; on hit, returns immediately
   without touching retrieval/generation.
3. `app/retrieval.py` `hybrid_retrieve()`: BM25 + vector search each over a
   candidate pool 3x the final top-k, fused via Reciprocal Rank Fusion, then
   `rerank()` does a single listwise LLM call to narrow to top-k. Reranking
   falls back to pre-rerank order if the LLM response is malformed.
4. `app/rag.py` `generate_answer()` does strict context-only generation,
   then `check_groundedness()` runs an LLM-as-judge hallucination check.
5. Result is cached (`app/cache.py`) and appended to session history
   (`app/memory.py`) before returning.

**`/ask-agentic`** (`app/agent.py`) replaces the single retrieve→generate
pass with a LangGraph loop: retrieve → grade (LLM judges if context is
actually sufficient) → generate if sufficient, else rewrite the query and
retry (capped at `MAX_RETRIES=2`, tracked via `retry_count` in graph state)
→ fallback to an honest "not enough information" response if retries are
exhausted. `/ask` is kept alongside `/ask-agentic` deliberately so both can
be called with the same question to compare behavior.

**Module map** (`app/`):
- `config.py` — all env/config loading; detects CI via `CI`/`GITHUB_ACTIONS`/
  `PYTEST_CURRENT_TEST` to relax the `GROQ_API_KEY`-required check.
- `providers.py` — `get_llm()` / `get_embeddings()` factory, the only place
  that branches on `MODEL_PROVIDER`.
- `database.py` — PostgreSQL + pgvector connection pool
  (`psycopg2.pool.ThreadedConnectionPool`), idempotent schema init
  (`init_db()`, executes `db_schema.sql`), and the query functions
  `ingest.py`/`retrieval.py`/`cache.py` call: `upsert_chunks()`,
  `delete_chunks_by_source()`, `hybrid_search()` (vector + full-text + RRF
  in one query), `cache_get()`/`cache_set()`. All non-critical-path
  functions (cache) follow the fail-open pattern used elsewhere in the app.
- `ingest.py` — load → chunk → embed → persist to PostgreSQL + pgvector
  (via `database.py`); content-hash based incremental re-ingestion tracked
  in the `ingest_manifest` table (unchanged files skipped, changed files'
  old chunks replaced, one bad file is recorded/skipped rather than
  failing the whole batch). The manifest lives in Postgres, not a local
  file — deliberately, so it can't desync from whichever database
  `DATABASE_URL` currently points at (see `database.py`'s module
  docstring). Manifest entries are written per-file, immediately after
  that file's chunks are upserted, so an interrupted run doesn't lose
  progress.
- `retrieval.py` — hybrid retrieval + LLM reranking (see module docstring
  for why an LLM reranker was chosen over a cross-encoder here).
- `rag.py` — single-pass retrieve/generate/groundedness-check, used by both
  `/ask` and as the retrieval base for `/ask-agentic`.
- `agent.py` — the self-correcting LangGraph loop described above; every
  node is wrapped with `@traceable` for LangSmith tracing (off by default,
  `LANGSMITH_TRACING=false` in `.env`).
- `memory.py` — per-session conversation history + query contextualization,
  backed by Firestore (one document per `session_id`, capped at 5 turns,
  `expires_at` field for Firestore's native TTL -- the policy itself is a
  one-time `gcloud firestore fields ttls update` call, not something the
  code sets). Fails open: with no `GCP_PROJECT_ID` and no
  `FIRESTORE_EMULATOR_HOST`, or on any Firestore error, behaves as if
  there's no history rather than raising -- same posture as `cache.py`.
- `cache.py` — semantic cache (embedding-similarity match, not exact-string)
  for repeated/similar questions.
- `streaming.py` — SSE streaming for `/ask-stream`.
- `middleware.py` — API key auth (`APIKeyMiddleware`; auth is disabled when
  `API_KEY` is unset), CORS, rate limiting (slowapi).
- `metrics.py` — real OpenTelemetry instruments (Counter/Histogram), not a
  hand-rolled dataclass. `GET /metrics` always serves Prometheus
  exposition format (`prometheus_client.generate_latest()`, no separate
  HTTP server) with zero GCP config; setting `OTEL_GCP_EXPORT=true` (needs
  `GCP_PROJECT_ID`) additionally pushes to Cloud Monitoring every 60s via
  `opentelemetry-exporter-gcp-monitoring` (pre-1.0/alpha, deprecated
  upstream in favor of native OTLP — noted as accepted debt in the module
  docstring). `record_*()` function signatures are unchanged from the old
  implementation, so call sites in `main.py` didn't need to change.
- `main.py` — FastAPI app/routes; every `/ask*` request is logged as one
  structured JSON line to `logs/requests.log` (question, sources,
  groundedness, retries, latency) — this is the primary observability
  signal, check it when debugging request behavior. `/health` is a pure
  liveness probe (`{"status": "ok"}`, no dependency checks) — `/ready`
  is the one that checks the vector store and is what Cloud Run's
  readiness probe hits. `/config` exposes non-secret runtime flags
  (`enable_uploads`, `model_provider`) for `ui.html` to adapt to, e.g.
  hiding the upload form when `ENABLE_UPLOADS=false` — set this in
  public-demo deployments to stop random callers from mutating the
  prod knowledge base via `/upload`. All handled exceptions log via
  `logger.error(json_payload, exc_info=True)`, not `logger.exception()`
  — keep that convention (`G201` is ignored in `pyproject.toml` for it).

**Secrets**: `config._get_secret()` reads from env first, falling back to
GCP Secret Manager when `GCP_PROJECT_ID` is set (used in Cloud Run
deployment; local dev always uses `.env`).

**Deployment**: Dockerfile + `cloudrun-groq.yaml` / `cloudrun-vertexai.yaml`
for GCP Cloud Run. Both configs mount `DATABASE_URL` from Secret Manager
and connect to Cloud SQL via the Auth Proxy sidecar
(`run.googleapis.com/cloudsql-instances` annotation). Ingestion does not
run at build time or read from anything baked into the image — it writes
straight to the external Postgres database, so `uv run python -m app.ingest`
can run before or after a deploy, from anywhere with network access to
that database (locally via the Cloud SQL Auth Proxy, or from Cloud Shell);
see README "Deploying to GCP" for the full flow.

## Testing conventions

Tests mock at the module-function boundary via `unittest.mock.patch`, not
by mocking LLM clients directly — e.g. `tests/conftest.py` patches
`app.rag.retrieve_with_hybrid_and_rerank`, `app.rag.generate_answer`,
`app.rag.check_groundedness`, and `app.cache.get_cached_answer` /
`set_cached_answer` as shared fixtures (`mock_retrieval`, `mock_llm_answer`,
`mock_groundedness`, `mock_cache`). Follow this pattern for new endpoint
tests rather than mocking `get_llm()`/provider clients. Ruff ignores
`BLE001`, `B008`, `SIM117` project-wide (see `pyproject.toml`).
