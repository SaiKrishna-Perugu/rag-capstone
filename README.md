# RAG Capstone — Document Q&A API

A FastAPI service that answers questions grounded in a local set of documents,
using a Retrieval-Augmented Generation (RAG) pipeline. Started as a focused
2-day MVP; extended with agentic self-correction, hybrid retrieval,
reranking, standardized evaluation, observability, and multi-cloud model
support -- see "JD coverage" below for exactly what maps to what.

## JD coverage

Mapped directly against a "RAG Pipelines & Vector Intelligence" JD section
covering: *ingestion, chunking, embeddings, indexing, vector search, hybrid
retrieval, and grounding... vector databases... rerankers, retrieval
evaluators, and freshness pipelines.*

| JD requirement | Where it lives | Notes |
|---|---|---|
| Ingestion | `app/ingest.py` | Multi-format (pdf/txt/md/csv/html/docx), per-file error isolation |
| Embeddings | `app/providers.py` | Groq (FastEmbed local ONNX) or Vertex AI, config-switchable |
| Vector search | `app/retrieval.py` `hybrid_retrieve()` | pgvector cosine similarity (`<=>`), via `app/database.py` |
| **Hybrid retrieval** | `app/retrieval.py` `hybrid_retrieve()` | Postgres full-text (`tsvector`) + pgvector, fused via Reciprocal Rank Fusion in a single SQL query |
| Grounding | `app/rag.py` `generate_answer()`, `check_groundedness()` | Strict context-only prompting + LLM-as-judge hallucination check |
| Vector database integration | `app/database.py`, `app/db_schema.sql` | PostgreSQL + pgvector (Cloud SQL in production) -- external, shared store so multiple Cloud Run instances see the same index |
| **Rerankers** | `app/retrieval.py` `rerank()` | LLM-based listwise reranking (see file header for why, vs. cross-encoder) |
| **Retrieval evaluators** | `eval.py`, `eval_ragas.py` | Custom LLM-as-judge + standardized RAGAS metrics (faithfulness, relevancy, precision, recall) |
| **Freshness pipelines** | `app/ingest.py` (hash + manifest) | Incremental re-ingestion: unchanged files skipped, changed files replaced, new files added |

Also present, beyond this specific JD section: a self-correcting LangGraph
agent loop (`app/agent.py`), LangSmith tracing, and GCP Vertex AI + Cloud Run
deployment -- see the rest of this README.

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

`app/retrieval.py` replaces plain vector search with a two-stage pipeline,
used by both `/ask` and `/ask-agentic` (via `app/rag.py`'s `retrieve()`):

1. **Hybrid candidate retrieval** — Postgres full-text search (`tsvector`/
   `plainto_tsquery`, the BM25-equivalent) and pgvector similarity search
   run over a candidate pool 3x larger than the final top-k, fused with
   Reciprocal Rank Fusion -- all in a single SQL query (`app/database.py`
   `hybrid_search()`), no separate index to keep in sync. Catches both
   exact-term queries (product codes, IDs -- vector search alone is often
   weak here) and paraphrase/synonym queries (full-text search alone is
   weak here).
2. **LLM reranking** — the fused candidate pool is re-scored by an LLM in
   a single listwise call, narrowing down to the final top-k actually
   passed to generation. Falls back safely to the pre-rerank order if the
   LLM's response is malformed, rather than failing the request.

See the module docstring in `app/retrieval.py` for the full reasoning,
including why an LLM reranker was chosen over a cross-encoder here.

### Agentic RAG (`POST /ask-agentic`)

`app/agent.py` wraps retrieval in a self-correcting LangGraph loop instead of
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
  lets hybrid retrieval run as one SQL query (`app/database.py`
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
- **Provider abstraction (`app/providers.py`)** — every LLM/embeddings call
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
or start the API (`app/database.py` `init_db()`) -- no manual schema step.

**Conversation memory (Firestore) is optional for local dev** -- with no
`GCP_PROJECT_ID` and no `FIRESTORE_EMULATOR_HOST` set, `app/memory.py`
fails open (follow-up questions just aren't contextualized using history;
everything else works normally). To actually exercise it locally, run the
Firestore emulator in Docker:
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
   uv run python -m app.ingest
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
currently in use. Re-running `uv run python -m app.ingest` after adding a
new file only embeds the new file; unchanged files are skipped entirely
(no re-embedding cost), and a changed file has its old chunks deleted and
replaced (`ON CONFLICT (source, content_hash)` in Postgres). One bad file
(corrupt PDF, malformed doc) is recorded and skipped, not a fatal crash for
the whole batch -- check the printed summary for `Failed:` count and details.

Force a full re-embed of every file regardless of whether it changed (e.g.
after switching `MODEL_PROVIDER`, since embeddings from different providers
aren't compatible with each other in the same collection):
```bash
uv run python -m app.ingest --force
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

## Switching to Vertex AI (instead of Groq)

All LLM/embeddings calls go through `app/providers.py`, so this is a config
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
  first `uv run python -m app.ingest`, and the schema will be created at
  the right width automatically.
- **Switching providers on a database that's already been ingested into:**
  re-run with `--force` (`uv run python -m app.ingest --force`) is *not*
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

Uses GCP's free trial ($300 credit, 90 days) -- enough for portfolio-project
usage. Requires the `gcloud` CLI installed and authenticated (Cloud Shell
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
    firestore.googleapis.com monitoring.googleapis.com

# 2. Provision Cloud SQL for PostgreSQL + pgvector -- the external, shared
#    vector store all Cloud Run instances read/write (db-f1-micro is the
#    smallest tier, right-sized for portfolio-project traffic)
gcloud sql instances create rag-capstone-db \
    --database-version=POSTGRES_16 \
    --tier=db-f1-micro \
    --region=us-central1
gcloud sql databases create ragdb --instance=rag-capstone-db
# Connect (e.g. via `gcloud sql connect rag-capstone-db --user=postgres`)
# and enable the extension once:
#   CREATE EXTENSION IF NOT EXISTS vector;

# 2b. Provision Firestore (conversation memory) in Native mode, and set a
#     TTL policy on expires_at -- app/memory.py sets that field on every
#     write, but Firestore only actually deletes expired docs once this
#     server-side policy is enabled (one-time, not something Python can do)
gcloud firestore databases create --location=us-central1 --type=firestore-native
gcloud firestore fields ttls update expires_at \
    --collection-group=conversation_sessions --enable-ttl

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
# (conversation memory), and push metrics to Cloud Monitoring
for role in roles/cloudsql.client roles/datastore.user roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:rag-capstone-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="$role"
done

# 7. Deploy using the declarative yaml (edit cloudrun-groq.yaml first --
#    replace YOUR_PROJECT_ID, including in the cloudsql-instances annotation)
gcloud run services replace cloudrun-groq.yaml --region=us-central1

# 8. Ingest documents into Cloud SQL. Easiest from Cloud Shell (already
#    authenticated) or locally via the Cloud SQL Auth Proxy -- either way,
#    point DATABASE_URL at the instance and run the same ingest command;
#    it creates the schema on first run (see "Build the index" above) and
#    is what /ready checks a chunk count against.
uv run python -m app.ingest

# 9. Cloud Run services aren't publicly reachable by default -- allow it
gcloud run services add-iam-policy-binding rag-capstone \
    --region=us-central1 \
    --member="allUsers" \
    --role="roles/run.invoker"
```

**A secret can't be created empty** (`gcloud secrets create ... --data-file=-`
with no input errors with `INVALID_ARGUMENT: Secret Payload cannot be empty`).
If you want the deployed app's API-key auth effectively off (open demo),
`app/middleware.py`'s check is `if config.API_KEY and ...`, so any
non-empty placeholder value works the same as truly empty -- just don't
send that header when testing. If you want it protected, use a real value
and send it back as `X-API-Key: <value>` on every `/ask`, `/ask-agentic`,
`/upload`, and `/metrics` call.

### Redeploying after code or doc changes

```bash
# only if docs/ changed -- DATABASE_URL must point at the Cloud SQL
# instance (directly, or via the Cloud SQL Auth Proxy) for this to reach it
uv run python -m app.ingest
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

**If using Vertex AI:** also enable `aiplatform.googleapis.com` in step 1
(`gcloud services enable aiplatform.googleapis.com`). `cloudrun-vertexai.yaml`
runs as the default compute service account (unlike `cloudrun-groq.yaml`,
which runs as `rag-capstone-sa`), which needs `roles/aiplatform.user` for
Vertex AI calls, plus the same Cloud SQL / Firestore / Cloud Monitoring /
secret access that `rag-capstone-sa` was granted in step 6 above -- or
requests will fail with a permissions error:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
for role in roles/cloudsql.client roles/datastore.user roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="$role"
done
for secret in rag_api_key database_url; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```
(Find `YOUR_PROJECT_NUMBER` via `gcloud projects describe YOUR_PROJECT_ID`.)
Also set `EMBEDDING_DIMENSION=768` in that YAML's `env:` block if
ingesting into a brand-new database (`text-embedding-005` is 768-dim vs.
FastEmbed/Groq's 384 -- see "Switching to Vertex AI" above) -- already set
in the checked-in `cloudrun-vertexai.yaml`, along with `GCP_PROJECT_ID`
and `OTEL_GCP_EXPORT` (just replace `YOUR_PROJECT_ID` with the real value).

## Project structure

```text
app/
  agent.py      # self-correcting LangGraph loop (grade / rewrite / fallback)
  cache.py      # semantic caching for repeated questions (Postgres-backed)
  config.py     # centralized env/config loading
  database.py   # PostgreSQL + pgvector connection pool, schema init, hybrid search
  db_schema.sql # chunks / semantic_cache table + index definitions
  ingest.py     # load -> chunk -> embed -> persist to PostgreSQL + pgvector
  main.py       # FastAPI endpoints (/, /upload, /ask, /ask-agentic) + UI serving + logging
  memory.py     # conversation history (Firestore) and contextual query rewriting
  metrics.py    # OpenTelemetry metrics -- Prometheus /metrics + optional Cloud Monitoring push
  middleware.py # API key auth, CORS, rate limiting
  providers.py  # model provider factory (Groq / Vertex AI)
  rag.py        # single-pass retrieval, grounded generation, groundedness check
  retrieval.py  # hybrid retrieval + LLM reranking
  streaming.py  # Server-Sent Events (SSE) streaming
  ui.html       # frontend interface
docs/           # source documents (sample included)
tests/          # unit and integration tests
eval.py         # custom eval harness (LLM-as-judge, correctness + groundedness)
eval_ragas.py   # RAGAS eval harness (faithfulness, relevancy, precision, recall)
  cloudrun-groq.yaml    # Declarative Cloud Run configuration for Groq
  cloudrun-vertexai.yaml # Declarative Cloud Run configuration for Vertex AI
Dockerfile
pyproject.toml
```

## What this covers vs. a production system

Built deliberately as an MVP within a 2-day window. What's implemented is
real and working, not stubbed — but scoped down from a production system:

**Implemented:**
- End-to-end RAG: chunking, embeddings, vector storage, retrieval, grounded generation
- **External vector store (PostgreSQL + pgvector, Cloud SQL in production)**
  -- fixes the stateless-Cloud-Run-instance problem a local vector store
  has (every instance would otherwise see a different, divergent local
  copy), and its `tsvector`/`tsquery` full-text search lets hybrid
  retrieval run as one SQL query instead of a separately maintained BM25
  index -- see `app/database.py` and "Design decisions" above
- **Self-correcting agentic loop** (`/ask-agentic`): grade retrieved context,
  rewrite and retry on insufficient context (capped retries), graceful
  fallback -- see `app/agent.py`
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
  switchable via config -- see `app/providers.py` and "Switching to Vertex AI"
- **GCP Cloud Run deployment**: Dockerfile adapted for Cloud Run's dynamic
  `$PORT`, plus full `gcloud` deploy commands for both provider paths --
  see "Deploying to GCP" above
- **External session state (Firestore + OpenTelemetry)** -- the other half
  of the statelessness fix Phase 1 started for the vector store.
  Conversation history (`app/memory.py`) now lives in Firestore, one
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
  `app/cache.py` and `check_groundedness()`.

**Explicitly deferred (next steps, in priority order):**
1. **Asynchronous Ingestion Worker** — currently, the `/upload` endpoint runs ingestion synchronously in a thread. While this works well for demos and small files, for production at scale, document processing should be decoupled into a separate worker queue (e.g. Cloud Run Jobs + Pub/Sub) to prevent HTTP timeouts on massive files.
