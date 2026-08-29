# TODOs

Deferred decisions and open items that aren't tracked anywhere else. The
08-22 pre-review audit flagged that none of this was written down anywhere,
which is part of how the CORS gap below went unnoticed for as long as it did.
Keep entries short; link to the commit or review doc that has the detail.

## Open

- **Cloud SQL does not enforce SSL.** `sslMode=ALLOW_UNENCRYPTED_AND_ENCRYPTED`
  with `ipv4Enabled=true` (checked 2026-08-29). Not an active exposure: the
  service connects through the Cloud SQL Auth Proxy sidecar, which encrypts
  regardless. But the public IP means anyone holding the password could
  connect *unencrypted* from the internet. Tightening `sslMode` on a live
  database risks connectivity, so it wants a maintenance window rather than
  a drive-by change — and consider private-IP-only if nothing needs the
  public address.
- **`rag-capstone-sa` is both the runtime identity and the deploy identity.**
  It holds `roles/viewer`, `run.admin`, `storage.admin` and
  `cloudbuild.builds.editor` on top of the runtime roles it actually needs
  (`aiplatform.user`, `cloudsql.client`, `datastore.user`, `dlp.user`,
  `cloudtasks.enqueuer`, `storage.objectAdmin`, `monitoring.metricWriter`).
  The broad ones are there because `cd.yml` impersonates this same account
  to deploy, so simply removing them breaks CI/CD. The fix is splitting it
  into two service accounts, not trimming roles.
- **`containerConcurrency: 80` is not backed by the pools underneath it.**
  *(Reconfirmed by the 08-29 four-lens review, independently, from both the
  developer and DevOps lenses. Still open: the fix is a load test plus a
  sizing decision, not an edit.)*
  `DATABASE_POOL_MAX` defaults to 10 and no Cloud Run config overrides it,
  and `psycopg2`'s `ThreadedConnectionPool` raises `PoolError` immediately
  on exhaustion rather than queueing — `database.get_conn()` doesn't catch
  it. Blocking work also funnels through `asyncio.to_thread`, whose default
  executor is `min(32, cpu+4)` ≈ 6 threads at `cpu: '2'`. So effective safe
  concurrency for anything touching Postgres is nearer 10 than 80. Raising
  `DATABASE_POOL_MAX` multiplies by `maxScale` against Cloud SQL's own
  connection limit — size it against the instance tier and load-test, don't
  guess. Low urgency: demo traffic is nowhere near it.
- **The `livenessProbe` can restart a busy container.** `timeoutSeconds: 1`
  on `/health` with `failureThreshold: 3` / `periodSeconds: 15`, against
  concurrency 80 on 2 vCPU: three slow replies inside 45s restart the
  instance and drop every in-flight `/ask`. That is a load amplifier under
  exactly the conditions a liveness probe should survive. 3–5s is the safer
  timeout. Left at 1 so `cloudrun-vertexai.yaml` keeps describing what is
  actually deployed; changing it is a production change, not a config-file
  edit.
- **CSP ships `Report-Only`, not enforcing.** The allowlist covers fonts,
  jsDelivr, cdnjs, gstatic/Firebase and the popup endpoints, but a wrong
  entry in an *enforcing* policy silently breaks sign-in — and the Firebase
  popup flow cannot be exercised from CI. Promotion path: deploy, sign in,
  ask a question, upload a file, confirm zero violation reports in the
  DevTools console, then rename the header to `Content-Security-Policy` in
  `app/main.py`. One-line diff.
- **GCP free-trial credits lapse around 2026-11-10** ($300 / 90 days;
  project created 2026-08-12). Credit expiry is the likeliest cause of a
  dead demo link, and no code change prevents it — it needs a billing
  decision (upgrade to a paid account, or accept the demo has an end date).
  It is at least no longer *silent*: the uptime check added in `6bae1bd`
  fails and emails on the next 15-minute tick once the service stops
  answering. Whether the account has already been upgraded is not
  determinable from the billing API — check the Cloud Console.

## Watch for (coupling traps, not action items)

- Changing `GROQ_CHAT_MODEL` without adding a matching row to
  `app/llm/cost.py`'s `_DEFAULT_PRICING` silently zero-prices every call,
  which silently disables `DAILY_BUDGET_USD` even when it's set. Same trap
  applies to `VERTEX_CHAT_MODEL`/`VERTEX_EMBEDDING_MODEL`.
- `INGEST_TARGET_URL` and `UPLOAD_BUCKET` (and any other Cloud Run env var)
  set only via `gcloud run services update` is silently reverted by the next
  `gcloud run services replace` against the checked-in YAML. If it matters,
  it belongs in the YAML.
- `FIREBASE_WEB_API_KEY` must be a **real value or absent — never a
  placeholder**. `ui.html`'s `initFirebase()` bails only on an *empty*
  api_key, so a placeholder is truthy: it loads the SDK and renders a
  sign-in button that cannot work. This is the one env var where the repo's
  usual `REPLACE-WITH-...` habit is worse than the gap it fills.
- A `gcloud run services update` creates a traffic-carrying revision outside
  `cd.yml`. If a `canary` tag is still assigned from a previous rollout, the
  new revision can land on **10% of traffic** rather than 100% — seen for
  real on 2026-08-29 while enabling Firebase. Check `status.traffic` after
  any imperative update.

## Deferred by design (documented trade-offs — do not "fix" these)

- **Guest uploads require no account, no email, no data.** The per-visitor
  file cap is the mitigation, not identity collection.
- **Session IDs are unauthenticated.** Anyone can claim any `session_id`
  string and manage "their" documents. Acceptable while sessions hold only
  demo uploads; if they ever hold sensitive content, bind the session to the
  Firebase uid when one is present.
- **`X-Forwarded-For` keying is spoofable** — inherent to Cloud Run, whose
  GFE appends rather than replaces. The limiter is a speed bump by design;
  the alternative was one shared bucket for every visitor.
- **`check_hallucination` opt-out is honored only under `API_KEY`.** On the
  public tier the verdict is always computed.
- **The groundedness judge and reranker share the generator's model family**,
  so they inherit its blind spots. A different provider via the existing
  `stage`-labelled `get_llm()`, or a local NLI cross-encoder, is the known
  follow-up.
- **Indirect prompt injection via uploaded documents is mitigated, not
  solved** — context delimiting, output screening and prompt fingerprinting.
  Screening document content at ingest was considered and rejected: a
  security corpus is full of sentences that look like payloads.
- **`/docs` and `/openapi.json` are public**, and admin routes 404 rather
  than 401. Both deliberate; preserve them.
- **`maxScale=2`** deliberately bounds the in-process rate limiter and the
  per-process budget ceiling. Don't raise it before a load test (k6/Locust
  against staging: `/ask` hit and miss, `/upload` → job polling, sustained
  SSE; define p95 target and the Cloud SQL connection ceiling first).
- **Multi-tenancy** — only revisit if the demo becomes a product. Every
  upload is already keyed by session, so a tenant column is the natural
  extension.

## Resolved

### Four-lens review (2026-08-29)

Code, testing, front-end and CI/CD passes, with Codex as an independent
second voice. Full report: `~/.gstack/projects/.../main-four-lens-review-20260829.md`.
Codex's top finding was wrong on the deployed service and the probe that
disproved it is recorded there -- claims were verified, not relayed.

- **CD could deploy past a failed quality gate.** cd.yml, ci.yml and
  eval.yml were independent workflows on the same push with nothing
  ordering them, and the human at the production approval gate is shown
  neither result. The workflow argued PR checks made this safe, which is
  true for merges and false for direct pushes -- how this repo is worked.
  A `require-green-checks` job now waits for both to pass for the exact
  SHA, failing closed, with an audited `workflow_dispatch` override so a
  red eval cannot block a hotfix. See `0bd109d`.
- **The canary asserted nothing** -- `sleep 120` then 100% traffic, so a
  revision that booted and then 500'd was promoted automatically. It now
  probes the canary revision's own URL for readiness and a real answer.
- **Config drift was invisible.** CD preserves live config by design, so
  the YAMLs are not the source of truth; a post-deploy step now reports the
  live values rather than applying the file (applying it takes the demo
  offline).
- **The semantic cache was unbounded** -- no TTL, no size cap, no eviction,
  on a public endpoint. Now bounded on both axes and pruned by the existing
  sweep, and invalidated whenever the corpus changes, since a cached answer
  could outlive the document it cited and cache hits return no sources.
- **The daily budget was check-then-spend**, so concurrent requests all
  passed a ceiling none had paid for. Now an atomic reserve/release.
- **Conversation history was a read-modify-write with no transaction**, so
  simultaneous turns dropped each other.
- **Firebase verification blocked the event loop** ahead of rate limiting.
- **`/upload` accepted requests it could never ingest**, returning 202 and
  then failing the job with a misleading error; and it buffered whole files
  before checking their size.
- **Four a11y defects** in the new UI: contrast 0.06 above the floor, no
  `<h1>`, no `<main>`, no `aria-live` on results.
- **Three dead CSS variables and a stale `<title>`** left by the Ledger
  rewrite, live in production.
- **The agentic loop had one test** covering only the happy path; its
  retry, rewrite and fallback branches now have ten.


### Security audit (2026-08-29)

Found against a stale second checkout at `.zcode/workspace/default`
(ten commits behind); every finding was re-verified against this repo and
ported hunk-by-hunk in `07519e0`. See `SECURITY_AUDIT.md` / `FIX_PLAN.md`.

- **Unpinned CDN scripts with no SRI.** `ui.html` loaded `marked` from a
  floating jsDelivr URL; DOMPurify had no integrity either. Pinned
  `marked@15.0.12` + `dompurify@3.0.6` with SRI hashes **recomputed from the
  served bytes here** rather than trusted from the report (they matched).
  `renderAnswer()` falls back to `textContent` if either fails to load —
  which is also the XSS-safe direction.
- **`pypdf` 6.14.2 carried PYSEC-2026-3655/3656** and parses PDFs anonymous
  visitors upload. Bumped to 6.15.0; `uv.lock` regenerated, only pypdf moved.
- **The rate limiter was one shared bucket.** `get_remote_address` keys on
  the socket peer, which behind Cloud Run's GFE is the same address for
  every visitor — a single noisy client could 429 everyone (verified live:
  an 11-request burst). Now keys on the leftmost `X-Forwarded-For`.
- **No security headers on any response.** Added `nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy`, and a Report-Only CSP.
- **`/upload` read before it checked.** The per-file cap fired only after
  `await file.read()` had pulled the whole part into memory. A
  `Content-Length` gate now rejects oversized batches before any read.
- **Secret comparisons used `==`** on all three tiers — now
  `secrets.compare_digest`.
- **`.env.example` still shipped `MAX_UPLOAD_SIZE_MB=50`** with no
  `MAX_UPLOAD_FILES` — my own miss in `7dcb54d`, which hardened `config.py`'s
  default to 2 and never touched the file a deployment is copied from.
- **`check_hallucination` was client-controlled** on the open tier. Now
  honored only where `API_KEY` gates the deployment (`711f369`).
- **`ci.yml` wired a live `GROQ_API_KEY` into a fully-mocked job.** Removed:
  the suite patches `config.GROQ_API_KEY` directly and `_IS_CI` keys off
  `GITHUB_ACTIONS`. A live secret in a job that cannot use it is how an
  integration test added later starts making real paid calls.

### Upload abuse

- **No per-visitor cap on total uploaded files** — `MAX_UPLOAD_FILES` capped
  one request, so a guest could accumulate unlimited documents by repeating
  3-file uploads (observed live: 6+ files without signing in). Fixed with
  `MAX_SESSION_FILES` (guest, 6) / `MAX_SESSION_FILES_AUTHED` (30, floored at
  the guest value in `auth.session_file_limit()`), enforced in `/upload`
  against live chunks **plus staged-but-unprocessed files** (closes the
  back-to-back request race), mirrored in `ui.html` with a sign-in CTA,
  declared in all four Cloud Run YAMLs, and covered by
  `tests/test_session_file_cap.py`.
- **Signing in lowered the upload ceiling.** `upload_limits()` swapped
  between anon and authed config values with no floor, and the shipped
  defaults were 50MB anon vs 10MB authed. Defaults are now 2MB/3 files
  (matching production), and the authed values are floored at the anonymous
  ones in code so the inversion can't return via a single env var. See
  `7dcb54d`.
- **Upload/rate abuse bounds were imperative-only in every Cloud Run YAML.**
  `MAX_UPLOAD_FILES`, `MAX_UPLOAD_SIZE_MB` and `RATE_LIMIT` were declared in
  0 of 4 configs, so a `gcloud run services replace` dropped them to
  `config.py`'s defaults — 50MB x 5 files, a 250MB request on an endpoint
  anonymous visitors reach. Not hypothetical: **live staging was already in
  that state**. All four YAMLs now declare them (plus
  `MAX_CORPUS_CHUNKS`/`MAX_SESSION_CHUNKS`, absent entirely from both groq
  configs), and both live services were updated to match.
- **A failed upload becoming a permanent public document** — verified
  closed, not assumed. The local-disk path is guarded in `ingest.run()`'s
  ownership gate and covered by
  `test_ingest_refuses_an_upload_with_no_owning_session`. The production
  path stages to GCS, which `ingest.run()` never globs, on a bucket with
  public access prevention enforced and a 7-day delete rule (both confirmed
  live via `gcloud storage buckets describe`).
- **C1/C2/C3** (path traversal → stored XSS, cross-session upload leak,
  wrong-instance job execution) — closed 2026-08-23, see `2601373`.

### Configuration and deployment

- **`cloudrun-vertexai.yaml` had never been applied to production** — live
  ran `containerConcurrency: 80` against the file's 10, plus a
  `targetBurstCapacity` and `livenessProbe` this file did not declare at
  all. (An earlier note said those were unique to `cloudrun-groq.yaml`;
  they are not — all four configs set them.) Reconciled file-to-live on
  2026-08-29. That does **not** make `services replace` safe to run
  casually: it still re-mounts `API_KEY`, resets `INGEST_TARGET_URL` to a
  placeholder, and omits `TASKS_SERVICE_ACCOUNT_EMAIL`.
- **`startup-cpu-boost` was declared in the YAML but not enabled on the
  live service** — the 08-22 review marked it done from the YAML line
  without checking the running config. Enabled 2026-08-29.
- **`DAILY_BUDGET_USD` shipped disabled (`0`).** Set to `0.25` in all four
  YAMLs and applied live to both services. Grounded in the project's real
  Cloud Billing budget (₹2000/month ≈ $0.70/day across everything), leaving
  headroom for the non-LLM share.
- **CORS reflected any origin with credentials on** — closed by turning off
  `allow_credentials` (nothing here uses cookie-based auth).
- **Spend ceiling was blind to `/upload`** — embedding calls now go through
  `_CostTrackingEmbeddings` and feed the same `add_usage()`/budget path.
- **`GROQ_CHAT_MODEL` default 404'd** (Groq stopped serving Llama). Now
  `openai/gpt-oss-20b`, with a matching price row.

### Performance

- **"Cold start ~25s to first answer" — measured, and it was mostly not
  cold start.** `/health` answered in 0.35s while the next `/ask` took
  20.6s and everything after was under 0.7s. A container serving `/health`
  that fast is already up; the cost was lazy provider init. `lifespan` now
  warms it on a background thread (`ENABLE_STARTUP_WARMUP`, default true).
  Post-deploy a novel question returned in **2.85s**. The warmup narrows
  that window rather than closing it — a request landing seconds after
  container start still races it (observed once at 19.9s after a revision
  switch).
- **~249MB of eval-only deps** (`ragas`, `datasets`, and transitive
  `pandas`/`scipy`/`openai`) shipped in the runtime image. Moved to an
  `eval` dependency group excluded from the Dockerfile's `uv sync --frozen`;
  `eval.yml` now runs `uv sync --frozen --group eval`. Verified by building
  the image and confirming those packages are absent.

### Identity

- **Firebase sign-in is LIVE** (enabled 2026-08-29). Web app registered,
  Google provider enabled with its OAuth client, and **both** Cloud Run
  hostnames in Authorization → Settings → Authorized domains:
  `rag-capstone-jjinz2egfq-uc.a.run.app` and
  `rag-capstone-1057080140820.us-central1.run.app` — note the second keeps
  the `rag-capstone-` prefix; the bare `<project-number>.us-central1.run.app`
  is not a real host and would have silently failed sign-in on that zone.
  `FIREBASE_WEB_API_KEY`/`FIREBASE_AUTH_DOMAIN` are set live and declared in
  `cloudrun-vertexai.yaml`. Sign-in remains **additive**: it raises upload
  ceilings and never gates access. Deliberately not enabled on staging —
  only the production hostnames are authorized, so a staging sign-in would
  fail the domain check.

### Housekeeping

- **Nine dead `app/*.py` paths** left over from `ebf0b35`'s package
  reorganisation — 7 in `.env.example`, 2 in the Cloud Run YAMLs that the
  review itself missed. `tests/test_doc_paths.py` now fails on any
  unresolvable `app/...py` reference outside `notes/`.
- **Nothing checked the advertised demo URL is alive.** Added
  `.github/workflows/uptime.yml`: a scheduled Action hits `/health` on the
  canonical hostname every 15 minutes and fails (emailing the owner) if it
  doesn't answer. Chosen over Cloud Scheduler + Cloud Monitoring because it
  needs no GCP setup beyond the file.
- **Blocking Cloud DLP call in `redact_log_fields()`** ran on the event loop
  — now wrapped in `asyncio.to_thread`.
- **`python -m app.ingestion.ingest --help` ran a real ingestion** and `-f`
  silently no-op'd — moved to `argparse`.
- **`.github/codeql/`** — untracked and unwired to any workflow. Deleted
  rather than wired up; revisit if CodeQL scanning is wanted, it would need
  a `codeql.yml` authored from scratch.
- **`AGENTS.md`** — was untracked; now committed, same footing as
  `CLAUDE.md`.
- **`DEMO.md`** — intentionally untracked per the repository-voice rule, and
  now in `.gitignore` alongside `AUTOPLAN-REVIEW-*.md` so a `git add -A`
  can't commit them.
