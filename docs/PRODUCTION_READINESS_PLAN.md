# RAG Capstone — Production-Readiness Implementation Plan

A phased plan to close the gaps identified in the senior-engineer review.
Phases are ordered by **real dependency**, not just severity — several
later items genuinely cannot be done well until an earlier one lands
(e.g., load testing the current architecture would give misleading
numbers, since the stateless-violation bug means multi-instance behavior
is currently undefined).

**How to use this doc:** each phase has a goal, why it's sequenced where
it is, concrete technical steps, a rough effort estimate, and a
definition of done. Read Section 10 first if your time is limited — it
tells you honestly which phases matter most for a job search versus
which only matter if this becomes a real, live product.

---

## Phase 0 — The 10-Minute Fix (do this today, standalone)

**Goal:** `check_groundedness()` in `app/rag.py` has no exception
handling — a transient LLM failure on the *verification* call currently
crashes a request that already has a perfectly good answer.

**Fix:**
```python
def check_groundedness(answer: str, chunks: list) -> str:
    context = _format_context(chunks)
    try:
        llm = get_llm(temperature=0.0)
        messages = [
            ("system", _GROUNDEDNESS_SYSTEM_PROMPT),
            ("human", f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ]
        verdict = llm.invoke(messages).content.strip().upper()
        return verdict if verdict in ("GROUNDED", "UNSUPPORTED") else "UNKNOWN"
    except Exception:
        logger.warning("Groundedness check failed; returning answer unverified.", exc_info=True)
        return "NOT_CHECKED"
```
Add a test that mocks the LLM call to raise, and asserts the function
returns `"NOT_CHECKED"` instead of propagating — matching the pattern
already used for `cache.py` and `retrieval.py`'s fallbacks, and the test
style already used in `tests/test_rag.py`.

**Effort:** ~30 minutes including the test. **Definition of done:** a
test exists proving this path, and it passes in CI.

---

## Phase 1 — External Vector Store (the P0 architectural fix)

**Goal:** move the vector store off local container disk to an external,
shared store, so multiple Cloud Run instances see the same data instead
of divergent local copies.

**Decision: Cloud SQL for PostgreSQL + pgvector**, not Qdrant/Weaviate.
Reasoning, stated plainly: it keeps the whole stack on one cloud
provider (consistent with everything else you've built), and Postgres's
built-in full-text search (`tsvector`/`tsquery`) lets you implement
hybrid search as **one SQL query** instead of maintaining a separate,
hand-rolled, in-memory BM25 rebuild — which also directly resolves the
"BM25 doesn't scale" gap from the same migration. If you'd rather
prioritize a vector DB with hybrid search as a first-class built-in
feature over staying single-cloud, Qdrant is the alternative — but
commit to one; don't build both.

### 1.1 Provision the database
```bash
gcloud sql instances create rag-capstone-db \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region=us-central1
gcloud sql databases create ragdb --instance=rag-capstone-db
```
`db-f1-micro` is the smallest tier — right-sized for a portfolio
project's traffic, cheap against your remaining credit. Enable the
extension once connected:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.2 Schema design
```sql
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),           -- match your embedding model's dimension
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    content_hash TEXT NOT NULL,
    metadata JSONB,
    ingested_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (content_tsv);
CREATE UNIQUE INDEX ON chunks (source, content_hash);
```
The `content_tsv` generated column *is* your BM25-equivalent index,
maintained automatically by Postgres — no separate rebuild step, ever.

### 1.3 One SQL query replaces your hand-rolled hybrid retrieval
```sql
WITH vector_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_embedding)s) AS rank
    FROM chunks ORDER BY embedding <=> %(query_embedding)s LIMIT 20
),
keyword_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, plainto_tsquery(%(query_text)s)) DESC) AS rank
    FROM chunks WHERE content_tsv @@ plainto_tsquery(%(query_text)s) LIMIT 20
)
SELECT c.*, 
       COALESCE(1.0/(60+v.rank), 0) + COALESCE(1.0/(60+k.rank), 0) AS rrf_score
FROM chunks c
LEFT JOIN vector_ranked v ON c.id = v.id
LEFT JOIN keyword_ranked k ON c.id = k.id
WHERE v.id IS NOT NULL OR k.id IS NOT NULL
ORDER BY rrf_score DESC LIMIT 10;
```
This is the *exact same RRF math* currently in `_reciprocal_rank_fusion()`
— `k=60`, unchanged — just executed by Postgres instead of Python. You
keep your existing LLM-based reranking stage on top, unchanged.

### 1.4 Code changes
- `app/ingest.py`: replace `Chroma.from_documents(...)` with `INSERT`
  statements against `chunks`, keyed on `(source, content_hash)` for the
  existing freshness-check logic to slot into naturally (an `ON CONFLICT`
  clause handles the "changed file → replace" case cleanly).
- `app/retrieval.py`: replace `_get_vector_store()` and
  `_build_bm25_retriever()` with a single function running the query
  above via `asyncpg` (async, matches your FastAPI handlers) or the
  Cloud SQL Python Connector (handles secure connection + IAM auth for
  you, the more idiomatic GCP-native choice).
- `app/cache.py`: same migration — a second, smaller table
  (`semantic_cache`), same threshold logic, same fail-open pattern
  already there.
- New secret: `DATABASE_URL`, via Secret Manager, same pattern as
  `groq_api_key`.
- **Connection pooling matters here:** Cloud Run can spin up many
  concurrent instances; without pooling, each could open many DB
  connections and exhaust Cloud SQL's connection limit. Use
  `asyncpg.create_pool()` with a conservative max size, or the Cloud SQL
  Connector's built-in pooling.

### 1.5 Cutover strategy
For a portfolio project — no real users to protect — a clean cutover
during a short maintenance window is the right-sized approach: run the
migration, re-ingest into Postgres, redeploy pointing at the new store,
verify, done. (If this were genuinely live with real traffic, the honest
answer would be a dual-write period and gradual read cutover — worth
being able to say that distinction out loud in an interview, even though
you won't build it here.)

### 1.6 Test suite updates
`tests/conftest.py`'s fixtures currently mock the Chroma/BM25 layer —
update them to mock the new Postgres query function instead. Same
mocking philosophy, different target.

**Effort:** 3-5 focused days. **Definition of done:** `/ready` checks
row count in Postgres instead of a Chroma collection; two Cloud Run
instances running simultaneously both see an upload made through either
one; existing test suite passes against the new store.

---

## Phase 2 — External Session State (Memory + Metrics)

**Goal:** the second half of the statelessness fix — `memory.py` and
`metrics.py` are still plain in-process dicts after Phase 1.

**Decision: Firestore, not Memorystore for Redis.** Reasoning: Firestore
has a genuine free tier and pay-per-operation pricing, which matters
given your remaining credit; Memorystore's smallest instance has a fixed
hourly cost with no free tier. Redis is the more "textbook" answer at
real-world scale (lower latency, purpose-built for this), but Firestore
is the right-sized choice for this project's actual traffic and budget.

### 2.1 Conversation memory
Replace the in-process `_histories` dict with a Firestore collection:
document ID = `session_id`, fields = last 5 turns (unchanged limit),
plus a TTL policy (Firestore supports native TTL fields) so abandoned
sessions clean themselves up automatically instead of growing forever.

### 2.2 Metrics
Two real options, pick based on how much you want to learn:
- **Simpler:** keep the in-process counters as a fast local cache, but
  add a background task that periodically pushes aggregates to
  **Cloud Monitoring custom metrics** — you get centralized, durable
  metrics without a full rewrite.
- **More correct:** replace the hand-rolled `/metrics` endpoint with
  real **OpenTelemetry** instrumentation and the Cloud Monitoring
  exporter — the standard, portable observability pattern, and a
  stronger resume line than "custom in-memory metrics."

I'd do the OpenTelemetry version if you have the time — it's a genuinely
valuable, widely-transferable skill beyond this project specifically.

**Effort:** 2-3 days. **Definition of done:** a multi-turn conversation
survives being load-balanced across instances (test by forcing
`min-instances=2` temporarily and confirming follow-up questions still
have context); `/metrics` reflects activity from all instances, not just
whichever one answered the request.

---

## Phase 3 — Async Ingestion

**Goal:** stop blocking a request thread on document processing;
support "upload → get a job ID → poll status" instead.

### 3.1 Architecture
```
POST /upload  -->  validate + save file  -->  enqueue Cloud Task  -->  return 202 + job_id
                                                      |
                                                      v
                                         Cloud Task calls an internal
                                         endpoint (or separate worker
                                         service) that runs ingest.run()
                                                      |
                                                      v
                                         Job status written to a small
                                         Firestore "jobs" collection
```

### 3.2 Implementation notes
- **Cloud Tasks** over Celery: GCP-native, no separate broker to run and
  pay for (Celery needs Redis or RabbitMQ as a broker — extra
  infrastructure for a portfolio project's traffic level isn't worth it).
- Job status collection: `{job_id, status: pending|processing|done|failed,
  files, error, created_at}` — simple, queryable from a new
  `GET /jobs/{job_id}` endpoint, and from the UI (poll it after upload
  instead of waiting on the original request).
- **Bundle the manifest-write-timing fix here** while you're already
  touching ingestion: write the manifest incrementally after each file,
  not once at the end of the run — same root cause category (durability
  of partial progress), same fix location, no reason to do it twice.
- Consider whether ingestion should run in the *same* Cloud Run service
  or a separate one sized for CPU-heavy bursts rather than low-latency
  serving. For this project's scale, same-service-different-endpoint is
  fine; call out the separate-service option in your README as the
  documented next step if asked about it.

**Effort:** 2-3 days. **Definition of done:** uploading a large document
returns immediately with a job ID; the UI polls and shows progress;
`ingest_manifest.json` (or its Postgres equivalent) is updated
incrementally, verified by killing the process mid-batch and confirming
no duplicate re-processing on restart.

---

## Phase 4 — Evals in CI + Real CD Pipeline

**Goal:** stop letting prompt/retrieval changes silently degrade answer
quality, and stop deploying by a human running commands by hand.

### 4.1 Evals as a CI gate
Add a GitHub Actions job (separate from your existing mocked-pytest job,
since this one needs real API calls):
```yaml
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --frozen
      - env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          uv run python eval_ragas.py --output results.json
          uv run python scripts/check_thresholds.py results.json
```
`check_thresholds.py` (new, small script) fails the build if
faithfulness/precision/recall drop below a threshold you set from your
current baseline scores. **Keep the golden eval set small** (10-20
questions) to control CI cost, since this hits real Groq API calls on
every PR, unlike your fully-mocked unit tests.

### 4.2 Real staging environment
Create a second Cloud Run service, `rag-capstone-staging`, same image
pipeline, separate from production. Cheapest honest version for a
portfolio project: same GCP project, different service name (a fully
separate GCP project per environment is the more rigorous real-company
answer, but is overkill here).

### 4.3 CD pipeline
On merge to `main`: build → push → deploy to staging → run a smoke test
(a script hitting `/health`, `/ready`, and one real `/ask` call) → if
that passes, deploy to production using **Cloud Run traffic splitting**
rather than an instant 100% cutover:
```bash
gcloud run deploy rag-capstone --image=... --no-traffic --tag=canary
gcloud run services update-traffic rag-capstone --to-tags=canary=10
# watch error rate for a few minutes
gcloud run services update-traffic rag-capstone --to-latest
```
Document the rollback command right next to the deploy command in your
README — `gcloud run services update-traffic rag-capstone
--to-revisions=PREVIOUS_REVISION=100`.

**Effort:** 3-4 days. **Definition of done:** a PR that intentionally
degrades retrieval quality (e.g., dropping the reranker) fails CI; a
merge to main deploys itself through staging to production without a
manual `gcloud` command.

---

## Phase 5 — Real Authentication

**Goal:** replace the single shared `API_KEY` string with per-user
identity — the prerequisite for audit logs, per-user rate limits, and
any future multi-tenancy.

### 5.1 Firebase Auth
Simplest real path on GCP, genuine free tier. Replace
`app/middleware.py`'s `APIKeyMiddleware` with JWT verification against
Firebase's public keys. `ui.html` adds Firebase's JS SDK for a sign-in
flow (Google sign-in is a two-line integration).

### 5.2 Identity propagation
Once a verified user ID exists per request, thread it into your existing
structured logs (`app/main.py`'s `logger.info(json.dumps({...}))` calls)
— "who asked this" becomes answerable for the first time.

**Effort:** 2-3 days. **Definition of done:** unauthenticated requests
to `/ask` are rejected; the UI has a working sign-in flow; logs show a
real user identifier per request.

---

## Phase 6 — Circuit Breaker & Provider Failover

**Goal:** stop letting every request individually discover a Groq outage
the slow way (full retries + full timeout, repeated per request).

### 6.1 Circuit breaker
Hand-rolled is genuinely fine at this scale (a library like `pybreaker`
is also reasonable if you'd rather not maintain it yourself): track
consecutive failures in `app/providers.py`; after N consecutive failures
within a window, trip the circuit — fail fast for a cooldown period
instead of attempting the full retry sequence.

### 6.2 Automatic failover
You already have the provider abstraction (`get_llm()` switches on
`MODEL_PROVIDER`) — extend it so that when Groq's circuit is open,
`get_llm()` automatically returns a Vertex AI client instead, rather
than requiring a manual redeploy with a different config value. This is
the payoff for having built the abstraction cleanly in the first place.

**Effort:** 2 days. **Definition of done:** a test that forces
`get_llm()`'s underlying call to fail repeatedly and confirms the
circuit opens and subsequent calls route to the fallback provider.

---

## Phase 7 — Security Hardening

**Goal:** close the prompt-injection, PII-logging, and UI-sanitization
gaps.

- **Prompt injection:** add an input-screening step before a question
  reaches retrieval (a lightweight heuristic/pattern check, or an
  LLM-based classifier call for suspicious inputs) and an output check
  before returning an answer.
- **PII redaction:** route logged questions/answers through **Cloud
  Data Loss Prevention (DLP) API** before writing to structured logs —
  the GCP-native, real answer for this, not a hand-rolled regex.
- **Log retention:** configure a Cloud Logging retention policy instead
  of the current indefinite default.
- **UI XSS fix:** run LLM/markdown output through **DOMPurify** before
  inserting into `innerHTML` in `ui.html` — small, contained fix.

**Effort:** 2-3 days. **Definition of done:** a deliberately crafted
prompt-injection attempt is flagged rather than silently followed; a
question containing a fake SSN-shaped string is redacted in logs.

---

## Phase 8 — Cost Tracking

**Goal:** answer "what is this actually costing, per request, per
feature" — currently unanswerable.

Both Groq's and Vertex AI's SDK responses include token usage counts.
Capture input/output tokens per LLM call, multiply by a small
per-provider pricing table in `config.py`, and add the resulting cost
estimate to your existing structured logs and metrics. Optional stretch:
a budget-based circuit breaker — if daily spend crosses a threshold,
degrade to a cheaper model rather than refusing outright.

**Effort:** 1-2 days. **Definition of done:** logs show an estimated
dollar cost per request; `/metrics` includes a running daily total.

---

## Phase 9 — Load Testing

**Goal:** find this system's actual breaking point — currently unknown.

**Do this after Phase 1-2, not before** — load-testing the current
architecture would just confirm the known statelessness bug under load,
which tells you nothing new. Use **Locust** or **k6** to script a
realistic traffic mix (mostly `/ask`, some `/ask-agentic`, occasional
`/upload`), run against the staging environment from Phase 4, and use
the results to actually tune `containerConcurrency` and `maxScale`
instead of the current unvalidated defaults.

**Effort:** 1-2 days. **Definition of done:** a documented p50/p95/p99
latency curve at increasing concurrent-user counts, and a known request
rate where the system starts degrading.

---

## Phase 10 — Multi-Tenancy (only if you actually need this)

Not detailed in depth here deliberately — this is the one item from the
original review that's genuinely optional for a portfolio project.
Real implementation would mean per-tenant row-level security in
Postgres (a `tenant_id` column plus Postgres RLS policies) and
per-tenant Firestore collections for memory/jobs. Only build this if
you have a specific reason to (e.g., you're extending this into an
actual multi-customer product) — for demonstrating skill to an
employer, Phases 1-9 already cover the concepts an interviewer would
probe.

---

## Section 10 — What's Actually Worth Your Time

Be honest with yourself about why you're building this: **the goal is
landing a job, not running a live product with paying customers.** Not
every phase above is equally valuable against that actual goal, and
treating them as equally urgent would be a mistake given your limited
time.

**Do these regardless of time constraints — highest interview value per
hour invested:**
- **Phase 0** (10 minutes, no excuse not to).
- **Phase 1** (external vector store). This is the one architectural
  fix that actually changes what you can honestly claim in an interview
  — "I identified that my vector store violated Cloud Run's
  stateless-instance model and migrated to Postgres+pgvector, which also
  let me delete my hand-rolled BM25 rebuild" is a genuinely strong,
  specific story. This is the highest-value phase on the whole list.
- **Phase 4's eval-gate piece specifically** (not the full CD pipeline
  necessarily, just wiring evals into CI). "My evals run automatically
  and block a merge that regresses answer quality" is a strong, concise
  answer to the JD's "regression tests" language, and it's genuinely
  fast to build once Phase 1 is done.

**Do these if you have real runway (a few more weeks):**
- Phase 2 (external session state) — pairs naturally with Phase 1, same
  underlying story.
- Phase 6 (circuit breaker + failover) — you already built the provider
  abstraction that makes this cheap to add, and "automatic failover
  between providers" is a strong, concrete reliability story.
- Phase 9 (load testing) — a real, documented p95 latency number under
  load is more credible than any architecture diagram alone.

**Lower priority for your actual situation — genuinely important for a
real company, less critical for a job search:**
- Phase 3 (async ingestion), Phase 5 (real auth), Phase 7 (security
  hardening), Phase 8 (cost tracking) — all real, all things a senior
  engineer would ask about, but each is also something you can **speak
  to as a known, well-understood gap** ("here's exactly what I'd build
  and why, and here's the design") nearly as credibly as having actually
  built it, if time runs out. Being able to describe Phase 3's Cloud
  Tasks architecture correctly in an interview gets you most of the
  credit even unbuilt.
- Phase 10 (multi-tenancy) — skip unless you have a specific reason.

**My honest recommendation given everything you've told me about your
timeline:** do Phase 0 today, commit to Phase 1 as your next real
project block (3-5 days), then Phase 4's eval-gate piece. That's a
concentrated, coherent story — "I found the architectural flaw, fixed
the highest-impact one properly, and now have quality regressions caught
automatically" — that's worth more in an interview than a shallow pass
across all ten phases.
