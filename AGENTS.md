# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
# unchanged files skipped; see app/ingestion/ingest.py)
uv run python -m app.ingestion.ingest
uv run python -m app.ingestion.ingest --force   # full re-embed, e.g. after switching MODEL_PROVIDER

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
every push/PR to `main`, via `uv sync --frozen`. A separate
`.github/workflows/eval.yml` runs `eval_ragas.py` against an ephemeral,
job-scoped `pgvector/pgvector:pg16` service container (not any real Cloud
SQL instance) and `scripts/check_thresholds.py` on the same triggers —
kept separate from `ci.yml` since it makes real, paid LLM calls, a
different risk/cost profile than the fully-mocked `test` job. It runs on
**Vertex AI** via the same keyless WIF auth as `cd.yml`: Groq's free tier
is 100k tokens/day account-wide, shared with local development, and a few
pushes exhausted it — the gate then failed with a 429 having scored
nothing, which is a quota result masquerading as a quality one. It also
uploads `ragas_results.json` as an artifact so a failed gate stays
inspectable without re-running (and re-paying for) the eval.

Two failure modes here are easy to misread, both hit for real: RAGAS
degrades a timed-out job to **NaN**, and `check_thresholds.py` correctly
treats NaN as failure — so a too-short `RunConfig(timeout=)` looks exactly
like a quality collapse unless you read the per-job logs.
`LLMContextPrecisionWithoutReference` is the most timeout-prone metric
because it makes one LLM call per retrieved context.

`.github/workflows/cd.yml` (push to `main` only) automates what "Deploying
to GCP" in README used to be a fully-manual walkthrough for: build →
deploy to the staging Cloud Run service → smoke test → canary-promote to
production behind a required-reviewer `production` environment. It
authenticates via Workload Identity Federation
(`google-github-actions/auth`, no long-lived key) — and note the
`service_account:` input is what makes it impersonate `rag-capstone-sa`;
without it the action authenticates as the bare federated principal, and
every IAM grant made to that service account is silently inert. It carries
a `concurrency` group with `cancel-in-progress`, because a run parked at
the approval gate otherwise deploys *its* commit's image when approved
later, rolling production backwards — that happened. The whole workflow is
inert until the one-time GCP/GitHub setup in README's "Automated deploys"
is completed by hand; that setup can't be done from this codebase alone.

## Architecture

Everything routes through `app/config.py` for settings and `app/llm/providers.py`
for model access — no module calls `os.getenv()` or instantiates an LLM/
embeddings client directly outside these two files. Switching
`MODEL_PROVIDER` between `groq` and `vertexai` in `.env` is the only change
needed to swap providers; **the vector store is provider-specific**, so a
switch requires `uv run python -m app.ingestion.ingest --force` (mixing embedding
spaces from different providers in the same Postgres `chunks` table
silently produces bad retrieval, not an error — unless the embedding
*dimension* also changed, e.g. Groq's 384-dim FastEmbed vs. Vertex AI's
768-dim `text-embedding-005`, in which case pgvector rejects the insert
outright rather than silently corrupting retrieval; see `EMBEDDING_DIMENSION`
below and the vector store paragraph).

The vector store is PostgreSQL + pgvector (`app/db/database.py`,
`app/db/db_schema.sql`), not a local/embedded store — this is deliberate:
Cloud Run instances are stateless with independent local disks, so a local
vector store would mean every instance sees a different, divergent copy of
the index. Postgres's `tsvector`/`tsquery` full-text search also means
hybrid retrieval (`app/retrieval/hybrid.py` `hybrid_retrieve()` →
`database.hybrid_search()`) runs as a single SQL query doing vector search
+ full-text search + RRF fusion, rather than a separately maintained BM25
index. Schema creation (`database.init_db()`) is idempotent
(`CREATE ... IF NOT EXISTS` throughout) and runs both from `main.py`'s
FastAPI `lifespan` startup and from `ingest.py`'s `run()` — the latter
matters because ingestion is commonly the first command run against a
brand-new database, before the API has ever started. `EMBEDDING_DIMENSION`
is **derived from `MODEL_PROVIDER`** (768 for Vertex AI's
`text-embedding-005`, 384 for Groq/FastEmbed) rather than defaulting to a
fixed number, so switching provider can't silently build a table that
rejects every write; override it only for a custom embedding model. It
sets the `VECTOR(N)` column width at *first* creation only — it does not
widen an existing column, so changing it after the schema exists requires
dropping `chunks`/`semantic_cache`/`ingest_manifest` and letting
`init_db()` recreate them, then re-ingesting. Drop the manifest too, or
the next ingest skips every file as "unchanged" and leaves the tables
empty.

The vector index is **HNSW, not IVFFLAT**, and this is load-bearing:
IVFFLAT clusters existing rows into centroids at *build* time, but
`init_db()` runs before anything is ingested, so it was always built on an
empty table. With the default `ivfflat.probes=1` a query then scanned a
single near-empty list — measured directly, a 12-candidate request against
a 30-row table returned 2 rows, starving the vector half of hybrid
retrieval and making answers depend on whether full-text search alone
happened to hit. HNSW needs no training step, so building it before the
data exists is fine. Don't "optimize" it back to IVFFLAT.

Groq has no embeddings API, so `groq` mode uses local FastEmbed (ONNX,
no API key, no torch) for embeddings while still using Groq for chat.
FastEmbed's cache location is `FASTEMBED_CACHE_PATH` (config default:
`.fastembed_cache`, overridden to `/app/.fastembed_cache` in the Docker
image) — the Dockerfile pre-downloads the model into that path at build
time so containers never hit Hugging Face's API at runtime (its default
cache is `/tmp`, which doesn't persist across instances anyway, and Cloud
Run's shared outbound IPs routinely hit HF's anonymous rate limit).

**Documents are session-scoped.** Every uploaded chunk carries the
`session_id` of the browser that uploaded it (an `X-Session-Id` UUID kept in
`localStorage` — no login, so the demo stays usable by a stranger) plus an
`expires_at` TTL. Curated `docs/` files carry `NULL` for both, meaning
*visible to everyone, never expires*. `database.hybrid_search()` filters on
`(session_id IS NULL OR session_id = :sid) AND (expires_at IS NULL OR
expires_at > now())` in **three** places — both CTEs and the final join;
dropping it from one still returns plausible results, just leaky ones.
Visitors manage their own uploads via `GET /documents` and
`DELETE /documents/{filename}`.

Three traps this created, all found by running it rather than reading it:

- **The semantic cache is consulted BEFORE retrieval and is not
  session-aware.** An answer grounded in a private upload would be replayed
  to the next visitor asking something similar — retrieval looking correct
  while the isolation it provides is undone. Answers touching private
  documents are never cached (`rag.RagResult.used_private_docs`), *and* cache
  reads are skipped for visitors who have uploads of their own — otherwise a
  cached global answer shadows their document and the upload appears to do
  nothing.
- **`ingest.run()` rescans the whole docs tree on every upload**, so only
  files under `uploads/` may be session-tagged. Tagging indiscriminately
  converts the shared sample corpus into one visitor's private documents.
- **Deleting a document must delete its `ingest_manifest` row.** Leave it and
  re-uploading the same file is silently skipped as "unchanged" — measured:
  `added=0, skipped=11`, document unrecoverable, no error anywhere.

`get_chunk_count()` excludes expired rows, so `MAX_CORPUS_CHUNKS` cannot be
consumed by documents nobody can retrieve. That makes the Cloud Scheduler
sweep a storage optimisation rather than something correctness depends on.

**Request flow for `/ask` and `/ask-stream`** (`app/main.py` → `app/retrieval/rag.py`):
1. `app/retrieval/memory.py` rewrites the question using conversation history if a
   `session_id` is given (`contextualize_question`).
2. `app/retrieval/cache.py` checks the semantic cache; on hit, returns immediately
   without touching retrieval/generation.
3. `app/retrieval/hybrid.py` `hybrid_retrieve()`: BM25 + vector search each over a
   candidate pool 3x the final top-k, fused via Reciprocal Rank Fusion, then
   `rerank()` does a single listwise LLM call to narrow to top-k. Reranking
   falls back to pre-rerank order if the LLM response is malformed.
4. `app/retrieval/rag.py` `generate_answer()` does strict context-only generation,
   then `check_groundedness()` runs an LLM-as-judge hallucination check.
5. Result is cached (`app/retrieval/cache.py`) and appended to session history
   (`app/retrieval/memory.py`) before returning.

**`/ask-agentic`** (`app/retrieval/agent.py`) replaces the single retrieve→generate
pass with a LangGraph loop: retrieve → grade (LLM judges if context is
actually sufficient) → generate if sufficient, else rewrite the query and
retry (capped at `MAX_RETRIES=2`, tracked via `retry_count` in graph state)
→ fallback to an honest "not enough information" response if retries are
exhausted. `/ask` is kept alongside `/ask-agentic` deliberately so both can
be called with the same question to compare behavior.

**Module map** (`app/`):
- `config.py` — all env/config loading; detects CI via `CI`/`GITHUB_ACTIONS`/
  `PYTEST_CURRENT_TEST` to relax the `GROQ_API_KEY`-required check.
- `llm/providers.py` — `get_llm()` / `get_embeddings()` factory, the only place
  that branches on `MODEL_PROVIDER`. `get_llm()` returns
  `_ResilientLLM(_CostTrackingLLM(client))` — cost tracking sits *inside*
  failover so a fallback-served call is priced against the model that
  actually ran. **`get_embeddings()` deliberately never fails over**: the
  pgvector store is built in one provider's embedding space (768-dim
  Vertex vs 384-dim FastEmbed), so embedding a query with the other
  provider is either rejected outright or silently returns nonsense
  neighbours. A degraded chat provider is recoverable; a broken retrieval
  path is not.
- `llm/circuit.py` — circuit breaker for LLM providers. Standard
  closed/open/half-open machine keyed by **provider name**, not by
  `get_llm()`'s `(temperature, stage)` cache key — otherwise one
  provider's health would be split across up to 16 independent copies and
  never reach the threshold. Counts **consecutive** failures (any success
  resets), which is what lets it filter noise without needing to tell a
  provider outage from a bad request — Groq and Vertex raise entirely
  different exception types for both. It sits *outside* LangChain's retry,
  so each counted failure is already `LLM_MAX_RETRIES` upstream attempts.
  State is per-process: each Cloud Run instance learns about an outage
  independently, a deliberate trade against putting a network round-trip
  on every LLM call. Failover (`LLM_FALLBACK_PROVIDER`) is opt-in and
  separate from the breaker, which is always on — failing fast is worth
  having with or without somewhere to fail over to. Streaming only fails
  over when the primary breaks **before the first token**; re-routing
  mid-stream would restart the answer and the reader would watch it
  duplicate itself.
- `llm/budget.py` — daily LLM spend ceiling (`DAILY_BUDGET_USD`, 0 =
  disabled, and it ships disabled). Distinct from `circuit.py`: the breaker
  stops calls to a provider that is *broken*, this stops calls that are
  merely *expensive*, and a healthy provider plus a scripted loop against a
  public URL trips neither retry, failover nor rate limiting. Fed from
  `cost.add_usage()`, the single funnel every priced call already passes
  through, and enforced at the request boundary in `main.py` so a refused
  request costs zero tokens rather than failing partway through generation.
  Two deliberate imprecisions: the figure is `cost.py`'s **estimate** from a
  hand-maintained price table, not Cloud Billing, and state is **per
  process** like the breaker's — with `maxScale=2` the real ceiling is about
  twice the configured one. The window is the UTC calendar day, not a
  rolling 24h, because that needs one float and one date rather than the
  timestamp of every call.
- `db/database.py` — PostgreSQL + pgvector connection pool
  (`psycopg2.pool.ThreadedConnectionPool`), idempotent schema init
  (`init_db()`, executes `db_schema.sql`), and the query functions
  `ingest.py`/`retrieval.py`/`cache.py` call: `upsert_chunks()`,
  `delete_chunks_by_source()`, `hybrid_search()` (vector + full-text + RRF
  in one query), `cache_get()`/`cache_set()`. All non-critical-path
  functions (cache) follow the fail-open pattern used elsewhere in the app.
- `ingestion/ingest.py` — load → chunk → embed → persist to PostgreSQL + pgvector
  (via `database.py`); content-hash based incremental re-ingestion tracked
  in the `ingest_manifest` table (unchanged files skipped, changed files'
  old chunks replaced, one bad file is recorded/skipped rather than
  failing the whole batch). The manifest lives in Postgres, not a local
  file — deliberately, so it can't desync from whichever database
  `DATABASE_URL` currently points at (see `database.py`'s module
  docstring). Manifest entries are written per-file, immediately after
  that file's chunks are upserted, so an interrupted run doesn't lose
  progress.
- `ingestion/jobs.py` — async ingestion job tracking, backing `/upload`'s
  `202 {job_id}` + `GET /jobs/{job_id}` polling contract (see `main.py`
  below). Job records live in Firestore's `ingest_jobs` collection, same
  TTL pattern as `memory.py`. Unlike `memory.py`/`cache.py`, Firestore is
  **not** fail-open here — job tracking is `/upload`'s actual contract,
  not a latency optimization, so a missing/unreachable Firestore is a
  real `RuntimeError` that `main.py` turns into a 503, not a silent
  fallback. `process_job()` (set `processing` → run `ingest.run()` → set
  `done`/`failed`) is the single place that defines what processing a job
  means, called from two places depending on `GCP_PROJECT_ID`:
  `enqueue_cloud_task()` hands it to a real Cloud Task hitting
  `POST /internal/process-ingest-job` in production, while local dev (no
  official Cloud Tasks emulator exists) calls it directly via FastAPI's
  `BackgroundTasks` from the `/upload` handler itself — same job record,
  same polling contract either way, just without a real queue locally.
- `retrieval/hybrid.py` — hybrid retrieval + LLM reranking (see module docstring
  for why an LLM reranker was chosen over a cross-encoder here).
- `retrieval/rag.py` — single-pass retrieve/generate/groundedness-check, used by both
  `/ask` and as the retrieval base for `/ask-agentic`.
  The groundedness check is sampled by `GROUNDEDNESS_SAMPLE_RATE` (default
  1.0, so inert). Sampled-out requests return **`SKIPPED`**, deliberately
  distinct from `NOT_CHECKED`, which means the check ran and failed — merging
  them would make a routine sampling decision look like a provider error in
  the metrics.
- `retrieval/agent.py` — the self-correcting LangGraph loop described above; every
  node is wrapped with `@traceable` for LangSmith tracing (off by default,
  `LANGSMITH_TRACING=false` in `.env`).
- `retrieval/memory.py` — per-session conversation history + query contextualization,
  backed by Firestore (one document per `session_id`, capped at 5 turns,
  `expires_at` field for Firestore's native TTL -- the policy itself is a
  one-time `gcloud firestore fields ttls update` call, not something the
  code sets). Fails open: with no `GCP_PROJECT_ID` and no
  `FIRESTORE_EMULATOR_HOST`, or on any Firestore error, behaves as if
  there's no history rather than raising -- same posture as `cache.py`.
- `retrieval/cache.py` — semantic cache (embedding-similarity match, not exact-string)
  for repeated/similar questions.
- `llm/cost.py` — per-request LLM cost attribution. Pricing is USD per 1M
  tokens and **goes stale** — every entry is overridable by env
  (`RAG_PRICE_<MODEL>_IN` / `_OUT`) so a correction needs no code change,
  and the authoritative number is always Cloud Billing. Accumulates via
  `contextvars`, not a module global, so concurrent requests can't bill
  each other. `providers.py` wraps every LLM in `_CostTrackingLLM` (and
  that in `_ResilientLLM`), which covers `invoke()` and `astream()` — a
  new invocation path would go unmeasured *and* unprotected by the circuit
  breaker, so add an override to both proxies if one appears. Unknown models
  price at zero rather than guessing. Measured on a real `/ask`:
  **rerank is the most expensive stage (~47%), more than generation**,
  because it feeds 12 candidate passages to the LLM where generation gets
  only the final 4.
- `api/streaming.py` — SSE streaming for `/ask-stream`.
- `api/middleware.py` — two deliberately opposite postures, plus CORS and rate
  limiting (slowapi). `AccessControlMiddleware` **gates**;
  `IdentityMiddleware` **enriches** (resolves an optional Firebase token, no
  rejection path at all). Keep them separate — collapsing them into one
  "auth" layer is how the public demo accidentally gets walled off.

  Access is **tiered, not a boolean**. It used to be one switch, and both of
  its positions were wrong for a public demo: `API_KEY` set locked out the
  visitors the demo exists for, `API_KEY` unset left `/metrics` (spend, token
  counts, error rates) and `POST /internal/process-ingest-job` (triggers real
  ingestion) callable by anyone with the URL. Both were verified open on the
  live service before the fix, not inferred.

  | Tier | Routes | Control |
  |---|---|---|
  | Probe | `/health`, `/ready` | always open — Cloud Run calls these itself and cannot present a key |
  | Public | `/`, `/config`, `/ask*`, `/upload`, `/jobs/*`, `/documents*`, `/docs` | open, unless `API_KEY` is set |
  | Admin | `/metrics` | `X-Admin-Key` (`ADMIN_KEY`), else **404 not 401** — a 401 confirms the route exists |
  | Internal | `/internal/*` | Cloud Tasks OIDC only (`TASKS_SERVICE_ACCOUNT_EMAIL`) |

  **Unlisted paths 404 at the middleware.** A route added to `main.py` is
  unreachable until it is listed here. That maintenance cost is bought
  deliberately: a new endpoint silently inheriting public access is exactly
  how `/internal/process-ingest-job` ended up exposed. `API_KEY` is *kept*
  rather than replaced — it makes a whole deployment private, which is what
  staging uses it for and what `cd.yml`'s smoke test depends on.
- `api/auth.py` — optional Firebase identity, deliberately *additive*. A valid
  token raises the caller's upload ceiling
  (`MAX_UPLOAD_FILES_AUTHED`/`MAX_UPLOAD_SIZE_MB_AUTHED`); an absent,
  expired, or malformed one silently yields `ANONYMOUS` with the public
  limits, so a stale token in a browser tab degrades that visitor rather
  than locking them out. Verification uses `google-auth`'s
  `verify_firebase_token` — already an indirect dependency via
  `google-cloud-firestore`, so no `firebase-admin`. Inert with
  `FIREBASE_PROJECT_ID` unset: everyone is anonymous and the UI hides
  sign-in.
- `api/security.py` — prompt-injection screening + Cloud DLP PII redaction, in
  three parts with three different postures. `screen_question()` **gates**
  (400 before retrieval, so a refused request costs zero tokens) and runs
  on the *raw* question, before contextualization — screening the rewritten
  query would let the rewrite launder the payload. `screen_answer()`
  **repairs** (replaces an answer containing this app's own system-prompt
  fingerprints) and is the only mitigation that still applies to *indirect*
  injection arriving via an uploaded document, since it checks the effect
  rather than the input. `redact_log_fields()` **fails closed** — the one
  deliberate exception to this codebase's fail-open norm, because degrading
  to "log it raw" writes exactly the PII it exists to remove. Consequence
  worth knowing: enabling `ENABLE_PII_REDACTION` without
  `gcloud services enable dlp.googleapis.com` turns every logged
  question/answer into `[redaction unavailable]`. On `/ask-stream` output
  screening cannot suppress anything (tokens already sent), so it instead
  refuses to *cache* a leaked answer, which stops one success being
  replayed to later visitors.
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
  readiness probe hits. `/config` exposes non-secret runtime flags for
  `ui.html` to adapt to: `enable_uploads`, `model_provider`, the upload
  limits **as they apply to the calling identity**, and the public Firebase
  web config (an identifier, not a credential). `ENABLE_UPLOADS=false`
  makes `/upload` return 403 outright, not merely hide the form. All
  handled exceptions log via `logger.error(json_payload, exc_info=True)`,
  not `logger.exception()` — keep that convention (`G201` is ignored in
  `pyproject.toml` for it).
  `/upload` itself doesn't run ingestion inline — it saves files, creates
  a job via `jobs.create_job()`, and returns `202 {job_id}` immediately;
  `GET /jobs/{job_id}` and `POST /internal/process-ingest-job` (the
  latter is Cloud Tasks' HTTP target, on the **internal** tier — it accepts
  only a Cloud Tasks OIDC token, and denies everything when neither that nor
  `API_KEY` is configured) complete the async contract — see `jobs.py`
  above. `GET /jobs/{job_id}` additionally checks the job belongs to the
  calling session, so IDs are not enumerable across visitors.
  Because production accepts uploads from anonymous visitors, `/upload` is
  bounded on three axes checked *before* anything is written (a rejected
  batch must not leave the first N files already saved and queued):
  `MAX_UPLOAD_FILES` per request, `MAX_UPLOAD_SIZE_MB` per file, and
  `MAX_CORPUS_CHUNKS` total indexed chunks (0 disables; returns 507 when
  full). The corpus cap **fails open** on a database error — it is abuse
  mitigation, not a correctness invariant. Filenames are rejected if they
  contain HTML metacharacters: the filename is stored as the chunk's
  `source` and echoed back by `/ask`, so an unescaped one was a real
  stored-XSS vector. `ui.html` builds source cards with
  `createElement`/`textContent` for the same reason — don't reintroduce an
  `innerHTML` template there.

**Secrets**: `config._get_secret()` reads from env first, falling back to
GCP Secret Manager when `GCP_PROJECT_ID` is set (used in Cloud Run
deployment; local dev always uses `.env`).

**Deployment**: Dockerfile + one Cloud Run YAML per provider/environment —
`cloudrun-groq.yaml`, `cloudrun-groq-staging.yaml`,
`cloudrun-vertexai.yaml`, `cloudrun-vertexai-staging.yaml`. Staging in
either provider means `metadata.name: rag-capstone-staging`, a separate
`ragdb_staging` database on the same Cloud SQL instance, and its own
`ingest-queue-staging` — but shared Firestore with production, a
documented simplification, not an oversight.

**What is actually deployed right now** (verify with `gcloud`, don't infer
from this file): both services run **Vertex AI**
(`gemini-2.5-flash-lite`, `text-embedding-005`, 768-dim). Production is a
**public demo** — no `API_KEY`, so the public tier is open and anyone with
the URL can ask questions and upload within the limits above. `ADMIN_KEY`
gates `/metrics`, and `TASKS_SERVICE_ACCOUNT_EMAIL` puts `/internal/*` behind
Cloud Tasks OIDC — both set imperatively, so a `gcloud run services replace`
drops them (same caveat as `INGEST_TARGET_URL` below). Staging keeps its API
key as the place to test authenticated behaviour.
The YAMLs still mount `API_KEY`, so a `gcloud run services replace`
against production re-enables auth and locks visitors out; the restore
command is recorded in `cloudrun-vertexai.yaml` next to that env var.
`cd.yml` deploys by image tag, which preserves each service's existing
env/secret configuration, so it never changes provider on its own.

The `cloudrun-*.yaml` configs mount `DATABASE_URL` from Secret Manager
and connect to Cloud SQL via the Auth Proxy sidecar
(`run.googleapis.com/cloudsql-instances` annotation). Ingestion does not
run at build time or read from anything baked into the image — it writes
straight to the external Postgres database, so `uv run python -m app.ingestion.ingest`
can run before or after a deploy, from anywhere with network access to
that database (locally via the Cloud SQL Auth Proxy, or from Cloud Shell);
see README "Deploying to GCP" for the full flow. Every Cloud Run
YAML also carries an `INGEST_TARGET_URL` placeholder for Cloud Tasks' HTTP target
(`/internal/process-ingest-job`) — Cloud Run doesn't know its own URL
until after first deploy, so this gets set imperatively via
`gcloud run services update --update-env-vars` post-deploy, not templated
into the YAML; a later `gcloud run services replace` silently resets it
to the checked-in placeholder, so that update has to be re-run afterward.

Cloud Run exposes each service under **two** hostnames: the legacy
`<service>-<hash>-uc.a.run.app` and a newer
`<service>-<project-number>.<region>.run.app`. `status.url` returns the
legacy one — that is what README advertises as the demo link and the one
to hand a visitor. `gcloud run deploy`/`update` print the *newer* one in
their closing `Service URL:` banner, so quoting that banner hands out a
different host than the documented link. Both reach the same service and
both work for Cloud Tasks; the distinction matters because the two DNS
zones resolve independently — on 2026-08-20 one consumer ISP resolver
answered `REFUSED` for the whole `*.<region>.run.app` zone
**intermittently**: refusing for hours, recovering, then refusing again
within the same day, while `*.a.run.app` and all four major public
resolvers (Google, Cloudflare, Quad9, OpenDNS) served it throughout. The
recurrence is the point — "it works now" is not evidence it is fixed, so
treat the legacy hostname as canonical rather than as a workaround. While
a refusal lasts no HTTP request is ever issued, so **nothing appears in
the Cloud Run request log** and a healthy service reads as an outage.


## Repository voice

The tracked repo documents the system, not its author. Job-search material —
JD-requirement mappings, CV phrasing, demo scripts, interview talking points —
belongs in `notes/` (gitignored) or another untracked file, never in README,
module docstrings, or code comments. `notes/jd_coverage.md` and `DEMO.md` hold
that material today.

This is a style rule with a practical edge: comments like "the kind of check
most candidates skip" or "shows you thought about observability" address a
reviewer rather than the next maintainer, and they date badly. Explain what
the code does and why that choice was made — the reasoning is what carries
weight, and it reads the same to a colleague and to an evaluator.
## Testing conventions

Tests mock at the module-function boundary via `unittest.mock.patch`, not
by mocking LLM clients directly — e.g. `tests/conftest.py` patches
`app.retrieval.rag.retrieve_with_hybrid_and_rerank`,
`app.retrieval.rag.generate_answer`,
`app.retrieval.rag.check_groundedness`, and
`app.retrieval.cache.get_cached_answer` /
`set_cached_answer` as shared fixtures (`mock_retrieval`, `mock_llm_answer`,
`mock_groundedness`, `mock_cache`). Follow this pattern for new endpoint
tests rather than mocking `get_llm()`/provider clients. Ruff ignores
`BLE001`, `B008`, `SIM117` project-wide (see `pyproject.toml`).
