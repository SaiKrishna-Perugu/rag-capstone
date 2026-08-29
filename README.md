# Grounded Document Q&A — Retrieval-Augmented Generation on Cloud Run

A FastAPI service that answers questions from a document set and refuses to
answer beyond it. Retrieval is hybrid — Postgres full-text search and
pgvector similarity fused with Reciprocal Rank Fusion **inside a single SQL
query** — followed by LLM reranking, strict context-only generation, and an
LLM-as-judge groundedness check on every answer.

Beyond the core pipeline: a self-correcting agentic variant that grades its
own retrieved context and rewrites the query when it is insufficient,
session-scoped document uploads with TTL, a semantic answer cache,
conversation memory, per-request cost attribution broken down by pipeline
stage, a provider circuit breaker with cross-provider failover, and two
evaluation harnesses — one of which gates CI.

Runs on Cloud Run against Cloud SQL (PostgreSQL + pgvector). The model and
embeddings provider switches between Groq and GCP Vertex AI through one
config value.

**Live demo:** <https://rag-capstone-jjinz2egfq-uc.a.run.app>

## Measured results

Numbers from real runs, not estimates. Both are reproducible: the RAGAS
figures come from `.github/workflows/eval.yml` on every push to `main`, and
the cost breakdown from the `cost_by_stage` field logged on every `/ask`.

**Retrieval quality** (RAGAS, commit `dd520e1`, Vertex AI
`gemini-2.5-flash-lite`, 768-dim `text-embedding-005`):

| Metric | Score |
|---|---|
| faithfulness | 0.875 |
| answer_relevancy | 0.62 |
| llm_context_precision_without_reference | 0.76 |
| context_recall | 1.00 |

`answer_relevancy` is the one to read carefully rather than the one to
optimise. The eval set includes a question the corpus deliberately cannot
answer ("What is the CEO's name?"); the system correctly replies "I don't
have enough information in the provided documents to answer that", and RAGAS
scores that honest refusal **0.0** relevancy and 0.0 faithfulness. The
aggregate therefore understates the behaviour that matters most in
production. `context_recall: 1.00` is the more trustworthy signal here.

**Cost per request.** Reranking is the single most expensive stage at **~47%
of per-request spend -- more than generation** -- because it feeds 12
candidate passages to the LLM where generation only ever sees the final 4.
That is the measurement that would justify swapping the LLM reranker for a
local cross-encoder; it has not been done here, deliberately, and the
tradeoff is written up in `app/retrieval/hybrid.py`'s module docstring.

Two levers exist for that cost without changing the pipeline:
`GROUNDEDNESS_SAMPLE_RATE` (the check is a whole extra LLM call over the same
context) and `DAILY_BUDGET_USD` (a hard per-day ceiling). The sampling lever
ships inert at 1.0; the budget is set to $0.25/day in the Cloud Run configs,
chosen to sit under the project's own Cloud Billing budget rather than picked
out of the air.

## Architecture

```mermaid
graph TD
    subgraph Ingestion
        A[docs/*.pdf, *.txt, etc.] --> B(ingest.py)
        B -->|Hash check| C{Changed?}
        C -->|Yes| D[Chunk & Embed]
        D --> E[(PostgreSQL + pgvector)]
        C -->|No| F[Skip]
    end

    subgraph API Request
        G[POST /ask-stream] --> H[Conversation Memory]
        H --> I{Semantic Cache}
        I -->|Hit| J[Return Cached Answer]
        I -->|Miss| K[Hybrid Retrieval: one SQL query, tsvector + pgvector]
        K --> L[RRF Fusion]
        L --> M[LLM Reranking]
        M --> N[Grounded Generation SSE Stream]
        N --> O[Check Groundedness]
        O --> P[Return Final Answer & Sources]
        P -.->|Update| H
        P -.->|Update| I
    end
```

### Hybrid retrieval + reranking

`app/retrieval/hybrid.py` replaces plain vector search with a two-stage pipeline,
used by both `/ask` and `/ask-agentic` (via `app/retrieval/rag.py`'s `retrieve()`):

1. **Hybrid candidate retrieval** — Postgres full-text search (`tsvector`/
   `plainto_tsquery`, the BM25-equivalent) and pgvector similarity search
   run over a candidate pool 3x larger than the final top-k, fused with
   Reciprocal Rank Fusion -- all in a single SQL query (`app/db/database.py`
   `hybrid_search()`), no separate index to keep in sync. Catches both
   exact-term queries (product codes, IDs -- vector search alone is often
   weak here) and paraphrase/synonym queries (full-text search alone is
   weak here).
2. **LLM reranking** — the fused candidate pool is re-scored by an LLM in
   a single listwise call, narrowing down to the final top-k actually
   passed to generation. Falls back safely to the pre-rerank order if the
   LLM's response is malformed, rather than failing the request.

See the module docstring in `app/retrieval/hybrid.py` for the full reasoning,
including why an LLM reranker was chosen over a cross-encoder here.

### Agentic RAG (`POST /ask-agentic`)

`app/retrieval/agent.py` wraps retrieval in a self-correcting LangGraph loop instead of
a single pass. This is the "agentic" upgrade over `/ask`:

```mermaid
graph TD
    retrieve[Retrieve chunks] --> grade[Grade relevance]
    grade -->|sufficient| generate[Generate answer]
    grade -->|insufficient, retries left| rewrite[Rewrite query]
    grade -->|insufficient, retries exhausted| fallback[Fallback]
    rewrite -->|loop back| retrieve
    generate --> end_[Return response]
    fallback --> end_
```

- **Grade**: an LLM call judges whether retrieved context is actually
  sufficient to answer the question -- strict grading, not a rubber stamp.
- **Rewrite**: on insufficient context, the query is rewritten (different
  phrasing/specificity) and retrieval runs again. Capped at `MAX_RETRIES`
  (2) to prevent infinite loops -- this cap, plus the explicit `retry_count`
  in graph state, is what makes the loop safe to run in production rather
  than a fun-but-dangerous while-loop.
- **Fallback**: if still insufficient after retries, returns an honest
  "I don't have enough information" message instead of letting the LLM
  guess -- the whole point of the groundedness check carried over from `/ask`.
- `/ask` (single-pass) is kept alongside `/ask-agentic` deliberately, so you
  can call both with the same question and compare behavior directly.

**Design decisions:**
- **PostgreSQL + pgvector, not a local vector store (e.g. ChromaDB)** —
  Cloud Run instances are stateless with independent local disks, so a
  local vector store means every instance sees a different, divergent
  copy of the index. An external, shared Postgres database (Cloud SQL in
  production) fixes that, and its `tsvector`/`tsquery` full-text search
  lets hybrid retrieval run as one SQL query (`app/db/database.py`
  `hybrid_search()`) instead of maintaining a separate, hand-rolled BM25
  index that has to be rebuilt to stay in sync with the vector store.
- **LLM-as-judge for both groundedness and eval correctness** — a real,
  widely-used technique for evaluating LLM outputs where exact-match scoring
  doesn't work (answers are free text, not fixed strings).
- **Structured JSON-line logging** — every `/ask` and `/ask-agentic` request
  logs the question, answer, sources used, groundedness verdict, retries
  used, and latency. Minimal, but it's a real, queryable monitoring signal.
- **`/ask` (single-pass) and `/ask-agentic` (self-correcting loop) kept
  side by side** — lets you call both with the same question and compare
  behavior directly, e.g. when grading fails and a query gets rewritten.
- **Provider abstraction (`app/llm/providers.py`)** — every LLM/embeddings call
  goes through `get_llm()`/`get_embeddings()`. Switching between Groq and
  Vertex AI is a `.env` change (`MODEL_PROVIDER`), not a code change --
  see "Switching to Vertex AI" below.

## Setup

```bash
uv sync --frozen
cp .env.example .env
# edit .env and set your API key and DATABASE_URL
```

Needs a reachable PostgreSQL instance with the `pgvector` extension
available (the schema runs `CREATE EXTENSION IF NOT EXISTS vector`, which
requires the extension to be installed on the server, not just enabled).
For local dev, the quickest option is the official `pgvector` Docker image:
```bash
docker run -d --name rag-postgres -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=ragdb -p 5432:5432 pgvector/pgvector:pg16
```
matching the default `DATABASE_URL` in `.env.example`
(`postgresql://postgres:dev@localhost:5432/ragdb`). Tables/indexes are
created automatically and idempotently the first time you run ingestion
or start the API (`app/db/database.py` `init_db()`) -- no manual schema step.

**Firestore is optional for conversation memory, but required for `/upload`.**
With no `GCP_PROJECT_ID` and no `FIRESTORE_EMULATOR_HOST` set,
`app/retrieval/memory.py` fails open (follow-up questions just aren't
contextualized using history; `/ask`/`/ask-agentic` work normally either
way). `app/ingestion/jobs.py` does **not** fail open the same way, though --
job tracking is `/upload`'s actual contract (see "Asynchronous ingestion"
above), so uploading a document needs one of these two configured. Run
the Firestore emulator in Docker to exercise both locally:
```bash
docker run -p 8080:8080 gcr.io/google.com/cloudsdktool/google-cloud-cli:emulators \
  gcloud emulators firestore start --host-port=0.0.0.0:8080
```
then set `FIRESTORE_EMULATOR_HOST=localhost:8080` in `.env` -- the
`google-cloud-firestore` client auto-detects this env var and talks to
the emulator instead of real GCP, no credentials needed. (Google is
nudging the `:emulators` image tag toward deprecation in favor of
`:stable` plus a `COMPONENTS=google-cloud-cli-firestore-emulator` runtime
env var -- `:emulators` still works as of this writing; switch if it stops.)

## Usage

**1. Build the index**: Run ingestion locally to embed documents into Postgres:
   ```bash
   uv run python -m app.ingestion.ingest
   ```
Supports `.pdf`, `.txt`, `.md`, `.csv`, `.html`, and `.docx` -- drop any mix
of these into `docs/` (including subfolders) and re-run. This is also the
first thing that needs to run against a brand-new database -- it calls
`database.init_db()` itself (idempotent), so you don't need to start the
API first or run any schema SQL by hand.

**This is incremental by default** (the "freshness pipeline"): each file's
content is hashed and tracked in the `ingest_manifest` table -- in Postgres
itself, alongside the chunks it describes, not a local file. That matters
specifically because it's a *shared, external* store now: a local manifest
file has no way of knowing it's pointed at a different (or freshly created)
database than the one it was last run against, and would wrongly skip
files that were never actually ingested into whatever database is
currently in use. Re-running `uv run python -m app.ingestion.ingest` after adding a
new file only embeds the new file; unchanged files are skipped entirely
(no re-embedding cost), and a changed file has its old chunks deleted and
replaced (`ON CONFLICT (source, content_hash)` in Postgres). One bad file
(corrupt PDF, malformed doc) is recorded and skipped, not a fatal crash for
the whole batch -- check the printed summary for `Failed:` count and details.

Force a full re-embed of every file regardless of whether it changed (e.g.
after switching `MODEL_PROVIDER`, since embeddings from different providers
aren't compatible with each other in the same collection):
```bash
uv run python -m app.ingestion.ingest --force
```

**2. Start the API**:
```bash
uv run uvicorn app.main:app --reload
```

**3. Query it** — open http://127.0.0.1:8000/docs for the interactive
Swagger UI, or:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long is the refund window?"}'
```

**4. Run the custom eval harness**: Custom LLM-as-judge (fastest, cheapest, validates core correctness and groundedness):
```bash
uv run python eval.py
```

**5. Run the RAGAS eval harness**: Standardized RAGAS metrics (slower, more expensive, deep analysis of retrieval/generation):
```bash
# ragas and datasets live in the `eval` dependency group, not the default
# install -- they add ~249MB that has no business in the Cloud Run image.
uv sync --frozen --group eval
uv run python eval_ragas.py
```
This scores four standardized RAGAS metrics: **Faithfulness** (hallucination
check against retrieved context), **ResponseRelevancy** (does the answer
address the question), **LLMContextPrecisionWithoutReference** (was
retrieval precise), and **LLMContextRecall** (did retrieval surface what
was needed). See the compatibility note at the top of `eval_ragas.py` if
`import ragas` fails on a different environment -- it's a real, currently-
open version conflict between `ragas` and recent `langchain-community`
releases, worked around with a small shim, not a bug in this code.

## Access model (public demo vs. private deployment)

Access is **tiered**, because a single on/off switch has no correct
position for a demo meant to be publicly usable — with the key set you lock
out the visitors it exists for, with it unset you expose the operator
surface:

| Tier | Routes | Control |
|---|---|---|
| Probe | `/health`, `/ready` | Always open — Cloud Run calls these itself and cannot present a key |
| Public | `/`, `/config`, `/ask*`, `/upload`, `/jobs/*`, `/documents*`, `/docs` | Open, unless `API_KEY` is set |
| Admin | `/metrics` | `X-Admin-Key` header matching `ADMIN_KEY`, else **404** |
| Internal | `/internal/*` | Cloud Tasks OIDC token only (`TASKS_SERVICE_ACCOUNT_EMAIL`) |

Alongside these, Firebase (if configured) **enriches** rather than gates: a
signed-in caller gets raised upload limits and is never rejected.

Three details worth keeping:

- **Admin returns 404, not 401** — a 401 confirms the route exists to anyone
  probing. `/metrics` exposes token counts, estimated spend, latency and
  error rates, so set `ADMIN_KEY` on anything reachable from the internet.
  Leaving it unset means `/metrics` 404s for *everyone*, including you.
- **Unlisted paths 404 at the middleware.** A route added to `main.py` is
  unreachable until it is listed in `app/api/middleware.py`. That is deliberate:
  a new endpoint silently inheriting public access is how
  `/internal/process-ingest-job` was once left callable by anyone.
- **`API_KEY` still makes a whole deployment private**, which is what staging
  uses it for. The admin and internal tiers sit on top rather than replacing
  it.

The deployed production service runs as a **public demo**: no `API_KEY`, so
anyone with the URL can ask questions and upload within these bounds. Each
visitor's uploads are scoped to their browser session and expire after
`UPLOAD_TTL_HOURS`; they can list and remove their own files at any time
(`GET /documents`, `DELETE /documents/{filename}`).

| Limit | Anonymous | Signed in |
|---|---|---|
| Files per upload | `MAX_UPLOAD_FILES` (3) | `MAX_UPLOAD_FILES_AUTHED` (10) |
| Size per file | `MAX_UPLOAD_SIZE_MB` (2) | `MAX_UPLOAD_SIZE_MB_AUTHED` (10) |
| Chunks per visitor | `MAX_SESSION_CHUNKS` (300) | same |
| Total corpus | `MAX_CORPUS_CHUNKS` — 507 when full, 0 disables | same |

`MAX_CORPUS_CHUNKS` counts only **live** chunks: expired uploads are already
invisible to retrieval, so counting them would let the demo refuse new
uploads over documents nobody could read.

All three are enforced in `app/main.py` *before* any file is written, so a
rejected batch never leaves the first few already saved and queued. The
browser-side equivalents in `ui.html` are a UX affordance only — anything
client-side is bypassed by posting to `/upload` directly.

`ENABLE_UPLOADS=false` makes `/upload` return 403 outright (not merely hide
the form) — use it if you want a read-only demo. Note that unsetting
`API_KEY` also disables auth on `/upload`, so those two settings belong
together.

**Log retention.** `app/main.py` writes one structured JSON line per
`/ask*` request including the verbatim question and answer, which on a
public demo is user-generated content from anonymous visitors. The
`_Default` Cloud Logging bucket is set to **14 days** rather than the
30-day default — ample for debugging a demo, half the window in which that
content is retained, and free either way (Cloud Logging only bills for
retention *beyond* 30 days):
```bash
gcloud logging buckets update _Default --location=global --retention-days=14
```
Retention limits how long the content is kept; it does not redact it.
Actual PII redaction (Cloud DLP on the logged copy) is a separate,
still-open item.

**Enabling sign-in** (entirely optional; the demo works without it): in the
Firebase console, add a Web App to the same GCP project, enable the Google
provider, add your Cloud Run domain under *Authentication → Settings →
Authorized domains*, then set `FIREBASE_WEB_API_KEY` and
`FIREBASE_AUTH_DOMAIN`. With them unset the feature is inert and the UI
hides its sign-in button. The web API key is a public identifier, not a
credential — access is controlled by authorized domains, not secrecy.

## Switching to Vertex AI (instead of Groq)

All LLM/embeddings calls go through `app/llm/providers.py`, so this is a config
change, not a code change:

```bash
# in .env
MODEL_PROVIDER=vertexai
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1

# authenticate locally (one-time)
gcloud auth application-default login
```

**Important:** the vector store's embeddings are tied to whichever provider
created them, and now also to a fixed vector width -- FastEmbed/Groq
produces 384-dim embeddings, Vertex AI's `text-embedding-005` produces 768.
`EMBEDDING_DIMENSION` in `.env` controls the `VECTOR(N)` column width that
`db_schema.sql` is created with, but only on **first** creation (`CREATE
TABLE IF NOT EXISTS` won't widen an existing column). So:
- **Switching providers on a brand-new database:** set `MODEL_PROVIDER` and
  the matching `EMBEDDING_DIMENSION` (384 or 768) in `.env` *before* the
  first `uv run python -m app.ingestion.ingest`, and the schema will be created at
  the right width automatically.
- **Switching providers on a database that's already been ingested into:**
  re-run with `--force` (`uv run python -m app.ingestion.ingest --force`) is *not*
  enough by itself if the dimension also changed -- a dimension mismatch
  fails loudly at insert time (pgvector rejects it) rather than silently
  producing bad retrieval. Drop and let `init_db()` recreate the
  `chunks`/`semantic_cache` tables at the new `EMBEDDING_DIMENSION`, then
  re-ingest with `--force`. If the dimension is unchanged (e.g. testing
  against a Vertex AI embedding model that also happens to be 384-dim),
  `--force` alone is sufficient, same as before -- mixing embedding spaces
  from two different providers in the same table silently produces bad
  retrieval, not an error.

## Deploying to GCP (Cloud Run + Vertex AI)

Uses GCP's free trial ($300 credit, 90 days), which comfortably covers this
workload. Requires the `gcloud` CLI installed and authenticated (Cloud Shell
has this pre-installed; `uv` isn't, so run
`curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env`
first if you're using it).

These steps assume `us-central1` and the Groq path (`cloudrun-groq.yaml`) --
substitute your own region/project ID, and see the Vertex AI note at the
end if you're using `cloudrun-vertexai.yaml` instead.

```bash
# 1. One-time project setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com secretmanager.googleapis.com \
    artifactregistry.googleapis.com sqladmin.googleapis.com \
    firestore.googleapis.com monitoring.googleapis.com \
    cloudtasks.googleapis.com

# 2. Provision Cloud SQL for PostgreSQL + pgvector -- the external, shared
#    vector store all Cloud Run instances read/write (db-f1-micro is the
#    smallest tier, right-sized for this workload)
#    --edition=ENTERPRISE is required, not optional: Cloud SQL now defaults
#    new instances to ENTERPRISE_PLUS, which rejects shared-core tiers with
#    "Invalid Tier (db-f1-micro) for (ENTERPRISE_PLUS) Edition". Confirmed
#    by hitting it directly.
gcloud sql instances create rag-capstone-db \
    --database-version=POSTGRES_16 \
    --edition=ENTERPRISE \
    --tier=db-f1-micro \
    --region=us-central1 \
    --root-password=YOUR_DB_PASSWORD
gcloud sql databases create ragdb --instance=rag-capstone-db
# No manual `CREATE EXTENSION vector` step needed -- app/db/db_schema.sql
# already runs it (idempotently) as part of database.init_db(), which
# fires on both API startup and `python -m app.ingestion.ingest`.

# 2b. Provision Firestore (conversation memory + job tracking) in Native
#     mode, and set a TTL policy on expires_at for BOTH collections that
#     use it -- app/retrieval/memory.py and app/ingestion/jobs.py set that field on every
#     write, but Firestore only actually deletes expired docs once this
#     server-side policy is enabled per collection (one-time, not
#     something Python can do)
gcloud firestore databases create --location=us-central1 --type=firestore-native
gcloud firestore fields ttls update expires_at \
    --collection-group=conversation_sessions --enable-ttl
gcloud firestore fields ttls update expires_at \
    --collection-group=ingest_jobs --enable-ttl

# 2c. Create the Cloud Tasks queue app/ingestion/jobs.py enqueues async ingestion
#     jobs onto (see "Asynchronous ingestion" above) -- max-attempts is
#     bounded deliberately, since a whole-job failure reaching Cloud Tasks
#     means something systemic (e.g. Postgres unreachable), not a
#     per-file issue (those are already handled inside ingest.run()
#     without raising)
gcloud tasks queues create ingest-queue --location=us-central1 --max-attempts=3

# 3. Create Artifact Registry repository
gcloud artifacts repositories create rag-repo \
    --repository-format=docker \
    --location=us-central1

# 4. Build and push the container image via Cloud Build (a few minutes --
#    this also pre-downloads the FastEmbed embedding model into the image,
#    see Dockerfile comments). Ingestion does NOT need to happen before
#    this step anymore -- documents are embedded straight into Cloud SQL,
#    not baked into the image.
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag-repo/rag-capstone:latest

# 5. Create the service account cloudrun-groq.yaml runs as
gcloud iam service-accounts create rag-capstone-sa \
    --display-name="RAG Capstone Cloud Run service account"

# 6. Store secrets in Secret Manager, and grant THAT service account (not
#    the default compute SA -- it's a different identity) access to read
#    them. database_url points at the Cloud SQL instance via its Unix
#    socket path, which the Cloud SQL Auth Proxy sidecar (configured via
#    the run.googleapis.com/cloudsql-instances annotation already in
#    cloudrun-groq.yaml) makes available at runtime.
echo -n "your-api-key" | gcloud secrets create rag_api_key --data-file=-
echo -n "your-groq-key" | gcloud secrets create groq_api_key --data-file=-
echo -n "postgresql://postgres:YOUR_DB_PASSWORD@/ragdb?host=/cloudsql/YOUR_PROJECT_ID:us-central1:rag-capstone-db" \
    | gcloud secrets create database_url --data-file=-
for secret in rag_api_key groq_api_key database_url; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:rag-capstone-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
# Also let the service account connect to Cloud SQL, read/write Firestore
# (conversation memory + job tracking), push metrics to Cloud Monitoring,
# and enqueue Cloud Tasks
for role in roles/cloudsql.client roles/datastore.user roles/monitoring.metricWriter roles/cloudtasks.enqueuer; do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:rag-capstone-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="$role"
done

# 7. Deploy using the declarative yaml (edit cloudrun-groq.yaml first --
#    replace YOUR_PROJECT_ID, including in the cloudsql-instances annotation)
gcloud run services replace cloudrun-groq.yaml --region=us-central1

# 7b. Cloud Run doesn't know its own URL until after this first deploy, but
#     Cloud Tasks needs it as the target for /internal/process-ingest-job
#     -- no way to template this into the YAML at deploy time, so it's set
#     imperatively here. IMPORTANT: re-run this after every future
#     `gcloud run services replace` -- that command applies the YAML
#     verbatim, which would silently reset INGEST_TARGET_URL back to the
#     placeholder value checked into cloudrun-groq.yaml.
#     Note: Cloud Run currently exposes this service under TWO hostnames --
#     the legacy `<service>-<hash>-uc.a.run.app`, and a newer
#     `<service>-<project-number>.<region>.run.app` in a different DNS zone.
#     `status.url` (used just below) returns the LEGACY one; `gcloud run
#     deploy`/`update` print the NEWER one in their closing `Service URL:`
#     banner, so the two disagree and it is easy to copy the wrong one.
#     Either resolves for Cloud Tasks. For browser visitors prefer the
#     `status.url` value, which is what this README advertises as the demo
#     link: on 2026-08-20 one consumer ISP resolver answered DNS `REFUSED`
#     for the whole `*.<region>.run.app` zone INTERMITTENTLY -- refusing for
#     hours, recovering, then refusing again the same day -- while
#     `*.a.run.app` resolved normally throughout, as did all four major
#     public resolvers (Google, Cloudflare, Quad9, OpenDNS). Treat the
#     legacy hostname as the canonical link rather than a workaround: the
#     failure recurs, so "it works now" is not evidence it is fixed. Worth
#     recognising too, because the symptom is misleading -- the browser
#     shows "site can't be reached" and, since no HTTP request is ever
#     issued, NOTHING appears in the Cloud Run request log, so a perfectly
#     healthy service reads as an outage.
SERVICE_URL=$(gcloud run services describe rag-capstone --region=us-central1 --format='value(status.url)')
gcloud run services update rag-capstone --region=us-central1 \
    --update-env-vars=INGEST_TARGET_URL=$SERVICE_URL

# 8. Ingest documents into Cloud SQL. Easiest from Cloud Shell (already
#    authenticated) or locally via the Cloud SQL Auth Proxy -- either way,
#    point DATABASE_URL at the instance and run the same ingest command;
#    it creates the schema on first run (see "Build the index" above) and
#    is what /ready checks a chunk count against.
uv run python -m app.ingestion.ingest

# 9. Cloud Run services aren't publicly reachable by default -- allow it
gcloud run services add-iam-policy-binding rag-capstone \
    --region=us-central1 \
    --member="allUsers" \
    --role="roles/run.invoker"
```

This first deploy is manual by design -- everything after it is automated
(see "Automated deploys" below). Do this once to get a working service,
then stop running these commands by hand.

**A secret can't be created empty** (`gcloud secrets create ... --data-file=-`
with no input errors with `INVALID_ARGUMENT: Secret Payload cannot be empty`).
If you want the deployed app's API-key auth effectively off (open demo),
`app/api/middleware.py`'s check is `if config.API_KEY and ...`, so any
non-empty placeholder value works the same as truly empty -- just don't
send that header when testing. If you want it protected, use a real value
and send it back as `X-API-Key: <value>` on every `/ask`, `/ask-agentic`,
`/upload`, `/jobs/{id}`, and `/metrics` call. `/internal/process-ingest-job`
is protected the same way, but you don't need to do anything extra for
it -- `app/ingestion/jobs.py::enqueue_cloud_task()` reads `config.API_KEY` and sets
it as a header on the Cloud Task's own HTTP request, so it authenticates
itself automatically as long as `API_KEY` is set consistently (i.e. the
same secret both the deployed app and its own outgoing Cloud Task use).

### Redeploying manually (before the CD pipeline is set up, or for debugging)

```bash
# only if docs/ changed -- DATABASE_URL must point at the Cloud SQL
# instance (directly, or via the Cloud SQL Auth Proxy) for this to reach it
uv run python -m app.ingestion.ingest
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag-repo/rag-capstone:latest
gcloud run deploy rag-capstone --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag-repo/rag-capstone:latest --region=us-central1
```

Use `gcloud run deploy --image=...` (not `gcloud run services replace`) for
routine image-only updates. `cloudrun-groq.yaml` references the image by
the mutable `:latest` tag, so its rendered spec text never changes between
builds -- `services replace` diffs spec text, sees nothing different, and
silently keeps serving the old revision even though a new image was just
pushed. `run deploy --image=...` always resolves `:latest` to its current
digest and creates a new revision when it differs. Reserve
`services replace` for when you actually edit the YAML itself (new env
var, different secret, resource limits).

**Once the CD pipeline (below) is set up, stop running these commands
against production** -- `cd.yml` deploys on every push to `main`, and a
manual `gcloud run deploy` outside it would send that revision live
traffic immediately, which breaks the canary rollout's assumption that
only the pipeline controls traffic splitting (see "Automated deploys"
below). Document ingestion (`uv run python -m app.ingestion.ingest`) is **not**
part of `cd.yml` and stays a manual, operator-triggered step either way
-- matches this project's existing stance that ingestion is decoupled
from deploys (see "Architecture" in `CLAUDE.md`).

### Automated deploys (CD pipeline)

After the one-time manual first deploy above, `.github/workflows/cd.yml`
handles every deploy from a push to `main`: build the image → deploy to
a staging service → smoke-test staging (`/health`, `/ready`, a real
`/ask` call) → canary-rollout to production (10% traffic, wait, then
100%) — automating the exact sequence documented above, plus a staging
environment and a real (if simplified) canary step neither manual
workflow had.

**This is inert until you complete this one-time setup** -- I can't
create or hold your GCP/GitHub credentials, so every piece below is real,
working configuration that only activates once you run it yourself.

```bash
# Workload Identity Federation -- lets GitHub Actions authenticate to GCP
# without a long-lived downloaded service-account key (Google's current
# recommended pattern over JSON keys). Reuses rag-capstone-sa, already
# created in step 5 above.
PROJECT_ID=YOUR_PROJECT_ID
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
REPO=YOUR_GITHUB_USERNAME/rag-capstone

gcloud iam workload-identity-pools create github-pool \
  --project=$PROJECT_ID --location=global --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=$PROJECT_ID --location=global --workload-identity-pool=github-pool \
  --display-name="GitHub OIDC Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository == '${REPO}'"

gcloud iam service-accounts add-iam-policy-binding rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --project=$PROJECT_ID --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}"

# Cloud Build's runtime identity needs explicit grants -- for any GCP
# project created on/after 2024-05-03, Cloud Build no longer auto-grants
# its default service account broad Editor access.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/logging.logWriter"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"
# rag-capstone-sa also needs to be able to consume project quota/billing
# and manage storage -- confirmed by direct trial: gcloud builds submit
# kept failing with "forbidden from accessing the bucket" for a WIF-
# federated identity even after the three grants above, because
# gcloud's auto-created default source-staging bucket (<project>_cloudbuild)
# was created under legacy ACLs that don't reliably honor IAM role grants.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/serviceusage.serviceUsageConsumer"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
# cd.yml's Cloud Build step points source uploads at a fresh bucket
# (--gcs-source-staging-dir) instead of that legacy-ACL default -- gcloud
# creates it automatically on first use with modern IAM, which the grant
# above then actually applies to.
#
# IMPORTANT: none of the grants above have any effect unless cd.yml's
# google-github-actions/auth steps also pass `service_account:
# rag-capstone-sa@...`. Without that input, the action uses Direct
# Workload Identity Federation -- authenticating as the raw federated
# principal itself, not rag-capstone-sa -- so every grant on
# rag-capstone-sa is silently inert. This was confirmed the hard way:
# `gcloud builds submit` kept failing identically across five different
# fixes (three IAM roles, a fresh bucket, --billing-project) until
# service_account: was added to actually impersonate the SA that held
# those grants. Testing via `gcloud ... --impersonate-service-account`
# is not a faithful stand-in for this -- it impersonates correctly and
# so can pass while the real WIF-only path still fails.
#
# gcloud run deploy (used by deploy-staging and promote-production) needs
# two more grants beyond what step 6 gave rag-capstone-sa for running the
# service: permission to call the Cloud Run Admin API, and permission to
# actAs itself as the runtime identity a new revision deploys with.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud iam service-accounts add-iam-policy-binding rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --project=$PROJECT_ID --role="roles/iam.serviceAccountUser" \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com"
# Once impersonation was actually wired up (see above), `gcloud builds
# submit` got past the bucket upload and hit a *different* PERMISSION_DENIED:
# rag-capstone-sa (the caller submitting the build) also needs to be
# allowed to act as the identity Cloud Build itself runs the build as --
# which defaults to the project's Compute Engine default service account
# (this is the same 2024-05-03 behavior change noted above: Cloud Build no
# longer auto-provisions a dedicated build SA with broad Editor access).
gcloud iam service-accounts add-iam-policy-binding ${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
  --project=$PROJECT_ID --role="roles/iam.serviceAccountUser" \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com"
# With the build itself now actually running, `gcloud builds submit` still
# errored -- this time only failing to *stream back the build's logs*
# (confirmed the underlying build reached SUCCESS via `gcloud builds
# describe` regardless). Neither --suppress-logs, roles/logging.viewer,
# nor even the primitive roles/viewer fixed it (all three tried, in that
# order) -- turned out to be the exact same root cause as the
# source-staging bucket earlier: gcloud's auto-created *default logs
# bucket* also predates Uniform Bucket-Level Access and uses legacy ACLs
# that don't honor project IAM grants at all, regardless of role. Fixed
# in cd.yml by adding --gcs-log-dir pointed at the same fresh bucket
# --gcs-source-staging-dir already uses -- not an IAM change. roles/viewer
# is broader than this project needs; keeping it isn't necessary, but
# it's harmless if already applied.

# Staging environment: same Cloud SQL instance, a separate database within
# it (not a second instance -- cheaper, still keeps staging traffic and
# smoke-test uploads out of production data), and its own Cloud Tasks
# queue (so a staging deploy's async ingestion jobs can never target the
# wrong service).
gcloud sql databases create ragdb_staging --instance=rag-capstone-db
echo -n "postgresql://postgres:YOUR_DB_PASSWORD@/ragdb_staging?host=/cloudsql/${PROJECT_ID}:us-central1:rag-capstone-db" \
  | gcloud secrets create database_url_staging --data-file=-
gcloud secrets add-iam-policy-binding database_url_staging \
  --member="serviceAccount:rag-capstone-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
gcloud tasks queues create ingest-queue-staging --location=us-central1 --max-attempts=3

# First staging deploy (same pattern as production's first deploy above --
# after this, cd.yml manages it). Edit cloudrun-groq-staging.yaml first:
# replace YOUR_PROJECT_ID (including in the cloudsql-instances annotation),
# and check the image tag -- the file pins `:latest`, which may be an old
# build. Pin a specific commit-SHA tag from Artifact Registry if so;
# cd.yml deploys SHA tags, so `:latest` can silently lag behind main.
gcloud run services replace cloudrun-groq-staging.yaml --region=us-central1
gcloud run services add-iam-policy-binding rag-capstone-staging \
    --region=us-central1 --member="allUsers" --role="roles/run.invoker"
STAGING_URL=$(gcloud run services describe rag-capstone-staging --region=us-central1 --format='value(status.url)')
gcloud run services update rag-capstone-staging --region=us-central1 \
    --update-env-vars=INGEST_TARGET_URL=$STAGING_URL
uv run python -m app.ingestion.ingest   # against DATABASE_URL pointed at ragdb_staging
```

**Deliberate simplification, documented not hidden:** staging shares
Firestore (conversation memory, job tracking) with production -- Firestore
Native mode is one database per GCP project by default, and fully
isolating staging would need multi-database Firestore, which is out of
scope for this staging environment.

Then, in the GitHub repo itself (UI steps, not YAML):
1. **Settings → Secrets and variables → Actions → Variables tab**: add
   `GCP_PROJECT_ID` and `WIF_PROVIDER` (the full provider resource name
   printed by the `providers create-oidc` command above --
   `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider`).
2. **Settings → Environments → New environment**, named exactly
   `production` → add yourself as a required reviewer. `cd.yml`'s
   `promote-production` job references this environment, but that
   reference is a no-op until this environment actually exists with a
   reviewer rule -- without it, production deploys happen unattended on
   every push to `main`, which defeats the point of the gate. (Required
   reviewers need a public repo or a paid plan; this repo is public, so
   the free tier covers it.)
3. **Settings → Branches → Branch protection rule** for `main`: require
   the `test` (from `ci.yml`) and `eval-gate` (from `eval.yml`) status
   checks to pass before merging. This is what makes the eval gate
   actually *block* a bad PR rather than just report a red X after the
   fact.

**If using Vertex AI:** also enable `aiplatform.googleapis.com` in step 1
(`gcloud services enable aiplatform.googleapis.com`), and grant the
service account permission to call it:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-capstone-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```
That single grant is all that's needed: `cloudrun-vertexai.yaml` and
`cloudrun-vertexai-staging.yaml` run as `rag-capstone-sa`, the same
identity as the Groq configs, so they inherit the Cloud SQL / Firestore /
Cloud Monitoring / Secret Manager access granted in step 6. (Earlier
versions of `cloudrun-vertexai.yaml` ran as the default compute service
account, which holds none of those grants and fails at startup on secret
access -- fixed, but check the `serviceAccountName` if you're working
from an older copy.)

`EMBEDDING_DIMENSION` no longer needs setting by hand: `app/config.py`
derives it from `MODEL_PROVIDER` (768 for Vertex AI's
`text-embedding-005`, 384 for FastEmbed). The Vertex YAMLs still set it
explicitly so the deployed value is visible in review. It fixes the
`VECTOR(N)` column width at first schema creation and pgvector rejects
mismatched inserts, so **switching an existing database between providers
means dropping and recreating `chunks`/`semantic_cache`/`ingest_manifest`,
not just re-ingesting** -- see "Switching model providers" above.

**Both provider paths are deployable through `cd.yml`.** The pipeline
deploys by image tag (`gcloud run deploy --image`), which updates only the
container and preserves each service's existing env/secret configuration --
so whichever provider a service was last configured with is what it keeps.
The YAMLs are for the first deploy or a deliberate reconfiguration via
`gcloud run services replace`; remember that `replace` applies the file
verbatim and will reset `INGEST_TARGET_URL` to its placeholder.

## Project structure

```text
app/
  main.py       # FastAPI app + routes (/, /ask*, /upload, /jobs/{id}, /documents) -- entrypoint: app.main:app
  config.py     # centralized env/config loading, imported by everything
  metrics.py    # OpenTelemetry -- Prometheus /metrics + optional Cloud Monitoring push
  ui.html       # frontend interface
  db/
    database.py   # connection pool, idempotent schema init, session-scoped hybrid search
    db_schema.sql # chunks / semantic_cache / ingest_manifest, session columns, HNSW + GIN indexes
  llm/
    providers.py  # get_llm()/get_embeddings() factory -- the only place that branches on MODEL_PROVIDER
    circuit.py    # circuit breaker per provider; backs the failover in providers.py
    cost.py       # per-request token/cost attribution, broken down by pipeline stage
    budget.py     # daily spend ceiling (DAILY_BUDGET_USD), fed by cost.add_usage()
  retrieval/
    hybrid.py     # hybrid retrieval (tsvector + pgvector, RRF) + LLM reranking
    rag.py        # single-pass retrieve -> generate -> groundedness check
    agent.py      # self-correcting LangGraph loop (grade / rewrite / fallback)
    cache.py      # semantic cache, and the gate that keeps private answers out of it
    memory.py     # conversation history (Firestore) + contextual query rewriting
  ingestion/
    ingest.py     # load -> chunk -> embed -> persist; incremental via content hash
    jobs.py       # async job tracking (Firestore) + Cloud Tasks enqueueing
    storage.py    # stages uploads in Cloud Storage so a job isn't tied to one instance
  api/
    middleware.py # tiered access (probe/public/admin/internal) + optional Firebase identity
    auth.py       # optional Firebase identity -- additive, raises upload limits, never gates
    security.py   # prompt-injection screening + Cloud DLP PII redaction of logs
    streaming.py  # Server-Sent Events (SSE) streaming for /ask-stream
docs/           # source documents (sample included)
tests/          # unit and integration tests
scripts/
  check_thresholds.py # CI eval-gate: fails the build if eval_ragas.py's scores drop below floor
eval.py         # custom eval harness (LLM-as-judge, correctness + groundedness)
eval_ragas.py   # RAGAS eval harness (faithfulness, relevancy, precision, recall)
TODOS.md        # deferred decisions and open items, with the measurement behind each
.github/workflows/
  ci.yml          # lint + fully-mocked unit tests
  eval.yml        # eval-gate: real Vertex AI calls against an ephemeral Postgres, blocks bad PRs
  cd.yml          # build -> deploy staging -> smoke test -> canary-promote to production
  uptime.yml      # scheduled /health check on the demo URL; a failed run emails the owner
cloudrun-groq.yaml             # Declarative Cloud Run configuration for Groq (production)
cloudrun-groq-staging.yaml     # Same, for the staging service cd.yml deploys to
cloudrun-vertexai.yaml         # Declarative Cloud Run configuration for Vertex AI
cloudrun-vertexai-staging.yaml # Same, for staging -- this is what is actually deployed
Dockerfile
pyproject.toml
```

## Scope and known limits

Everything listed under "Implemented" is real and working, not stubbed.
What follows it is the honest boundary — the things this system
deliberately does not do, and why.

**Implemented:**
- End-to-end RAG: chunking, embeddings, vector storage, retrieval, grounded generation
- **External vector store (PostgreSQL + pgvector, Cloud SQL in production)**
  -- fixes the stateless-Cloud-Run-instance problem a local vector store
  has (every instance would otherwise see a different, divergent local
  copy), and its `tsvector`/`tsquery` full-text search lets hybrid
  retrieval run as one SQL query instead of a separately maintained BM25
  index -- see `app/db/database.py` and "Design decisions" above
- **Self-correcting agentic loop** (`/ask-agentic`): grade retrieved context,
  rewrite and retry on insufficient context (capped retries), graceful
  fallback -- see `app/retrieval/agent.py`
- Groundedness / hallucination check on every answer
- Two eval harnesses: custom LLM-as-judge (`eval.py`) and standardized
  RAGAS metrics -- faithfulness, response relevancy, context precision,
  context recall (`eval_ragas.py`)
- **LangSmith tracing** -- every agent node (`retrieve`, `grade`, `rewrite`,
  `generate`, `fallback`) is wrapped with `@traceable`, so a full request
  shows up in LangSmith as one root trace with each step nested inside,
  including the actual LLM calls, latency, and token usage per step.
  Off by default (`LANGSMITH_TRACING=false`); flip it on in `.env`.
- Structured request logging (question, answer, sources, latency, retries used)
- Source citation in every response (which chunks were used)
- **Multi-cloud model provider support**: Groq (default) or GCP Vertex AI,
  switchable via config -- see `app/llm/providers.py` and "Switching to Vertex AI"
- **GCP Cloud Run deployment**: Dockerfile adapted for Cloud Run's dynamic
  `$PORT`, plus full `gcloud` deploy commands for both provider paths --
  see "Deploying to GCP" above
- **External session state (Firestore + OpenTelemetry)** -- the other half
  of the statelessness fix Phase 1 started for the vector store.
  Conversation history (`app/retrieval/memory.py`) now lives in Firestore, one
  document per `session_id`, with a native TTL policy for automatic
  cleanup, instead of a per-instance in-process dict -- a multi-turn
  conversation now survives a restart or landing on a different Cloud Run
  instance. Metrics (`app/metrics.py`) are now real OpenTelemetry
  instruments: `GET /metrics` serves Prometheus exposition format locally
  with zero GCP config, and optionally (`OTEL_GCP_EXPORT=true`) also
  pushes to Cloud Monitoring so metrics are centralized across instances
  rather than reset on every restart. Both fail open -- Firestore/Cloud
  Monitoring being unreachable or unconfigured degrades to "no history" /
  "local metrics only" rather than breaking requests, the same posture as
  `app/retrieval/cache.py` and `check_groundedness()`.
- **Asynchronous ingestion (Cloud Tasks + job polling)** -- `POST /upload`
  no longer blocks the request on `ingest.run()`. It saves the files,
  creates a job record (`app/ingestion/jobs.py`, Firestore `ingest_jobs`
  collection), and returns `202 {job_id}` almost immediately; a new
  `GET /jobs/{job_id}` endpoint (polled by `ui.html` every ~2s) reports
  `pending` → `processing` → `done`/`failed`. In a real deployment
  (`GCP_PROJECT_ID` set), the actual work happens via a Cloud Task
  calling `POST /internal/process-ingest-job`, with automatic retries if
  it fails. Google ships no official local Cloud Tasks emulator, so
  local dev runs the exact same job through FastAPI's `BackgroundTasks`
  instead of a real queue -- the full `202`/poll/`done` contract is still
  100% testable locally; only the real queue's retry/backoff behavior
  needs a live deployment to exercise. Unlike conversation memory/cache,
  Firestore here is a **required** dependency for `/upload` (not
  fail-open) -- job tracking is the endpoint's contract, not a latency
  optimization, so an unreachable Firestore is a clear `503`, not a
  silent behavior change.
- **Eval gate in CI + a real staging/CD pipeline** -- `.github/workflows/eval.yml`
  runs `eval_ragas.py` against an ephemeral, job-scoped Postgres on every
  PR and push to `main`, and `scripts/check_thresholds.py` fails the build
  if faithfulness/relevancy/precision/recall drop below a floor set from
  a real baseline run -- a prompt or retrieval regression now fails CI
  instead of shipping silently. `.github/workflows/cd.yml` replaces the
  fully-manual deploy walkthrough with build → deploy-to-staging →
  smoke-test → canary-promote-to-production (10% traffic, then 100%),
  authenticated via Workload Identity Federation (no long-lived
  credentials) and gated by a GitHub environment approval before
  production traffic shifts. See "Deploying to GCP" → "Automated deploys"
  for the one-time setup this needs (I can't create or hold your real
  GCP/GitHub credentials, so this ships as complete, working config that
  stays inert until you run that setup yourself).

- **Circuit breaker + automatic provider failover** (`app/llm/circuit.py`,
  `app/llm/providers.py`) — without one, every request rediscovers a provider
  outage the slow way: `LLM_MAX_RETRIES=3` at a 60s timeout, three LLM
  calls per `/ask`, each holding a worker thread. After
  `LLM_CIRCUIT_FAILURE_THRESHOLD` *consecutive* failures the provider is
  skipped entirely for a cooldown, then probed once to recover. Counting
  consecutive failures (rather than a rate) is what makes it safe without
  a provider-specific exception taxonomy — any success resets the count,
  so an isolated bad request can never trip it. Setting
  `LLM_FALLBACK_PROVIDER` additionally routes chat calls to the *other*
  provider while the circuit is open, which is the payoff for having
  built `get_llm()` as a real abstraction: an outage no longer needs a
  redeploy with a different `MODEL_PROVIDER`. Verified live end-to-end —
  Groq with a deliberately invalid key failed for real, the circuit
  opened after N 401s, and subsequent calls were answered by Vertex AI
  with cost correctly attributed to the model that actually ran.
  Embeddings deliberately never fail over (the pgvector store is built in
  one provider's embedding space). Off by default: enabling it means this
  deployment must hold working credentials for both providers.

- **Security hardening** (`app/api/security.py`) — prompt-injection screening
  refuses a crafted payload with a 400 *before* retrieval, so an attack
  costs zero LLM calls; output screening catches the case input screening
  cannot, where the payload arrived inside an uploaded document rather than
  the question. Logged questions and answers are de-identified through
  **Cloud DLP** (`ENABLE_PII_REDACTION`, off by default) — this matters
  because production is a public demo logging text typed by anonymous
  visitors. That control deliberately **fails closed**, the one exception
  to this codebase's fail-open norm: if DLP is unreachable the field is
  written as `[redaction unavailable]` rather than raw, because falling
  back to raw would write exactly the data the control exists to remove.
  Enabling it therefore requires `gcloud services enable dlp.googleapis.com`
  first, or you trade your request logs for nothing.

- **Tiered access control** (`app/api/middleware.py`) — access used to be one
  boolean, and both of its positions were wrong for a public demo: the key
  set locked out the visitors the demo exists for, the key unset left
  `/metrics` (spend, token counts, error rates) and
  `POST /internal/process-ingest-job` (triggers real ingestion) callable by
  anyone with the URL. Both were confirmed open on the live service before
  the fix. Now split into probe / public / admin / internal tiers, with
  `/metrics` behind `X-Admin-Key` returning **404 rather than 401** (a 401
  confirms a route exists), `/internal/*` accepting only a Cloud Tasks OIDC
  token, and unlisted paths closed by default.
- **Session-scoped documents with self-service removal** — every upload
  carries the uploading browser's `X-Session-Id` (a `localStorage` UUID, no
  login) and a TTL; retrieval is filtered so one visitor's documents are
  invisible to everyone else, and the curated `docs/` corpus stays shared and
  permanent. Visitors list and delete their own files via `GET /documents`
  and `DELETE /documents/{filename}`. Before this, any visitor could change
  what every later visitor saw — including leaving a prompt-injection payload
  in place for the next person.

**Explicitly deferred:** most items previously listed here — async
ingestion, external session state, the vector store migration, the eval
gate + CD pipeline, optional per-user identity (`app/api/auth.py`),
per-request cost attribution (`app/llm/cost.py`), circuit breaker / provider
failover (`app/llm/circuit.py`), and security hardening (`app/api/security.py`) —
are now implemented. Still deferred: **load testing** (Phase 9) and
multi-tenancy (Phase 10, deliberately skipped).
