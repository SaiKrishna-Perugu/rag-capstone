# TODOs

Deferred decisions and open items that aren't tracked anywhere else. The
08-22 pre-review audit flagged that none of this was written down anywhere,
which is part of how the CORS gap below went unnoticed for as long as it did.
Keep entries short; link to the commit or review doc that has the detail.

## Open

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

## Resolved

- **`cloudrun-vertexai.yaml` had never been applied to production** — live
  ran `containerConcurrency: 80` against the file's 10, plus a
  `targetBurstCapacity` and `livenessProbe` only `cloudrun-groq.yaml` sets,
  i.e. the service was created from the groq config with Vertex AI env vars
  layered on imperatively. Reconciled file-to-live on 2026-08-29: the file
  now describes the configuration actually serving traffic, so filling in
  the placeholders and applying it yields the tested setup. Also corrected
  the declared `startupProbe`, which claimed a ~32s budget that has never
  been in force (production runs Cloud Run's default tcpSocket probe at
  240s) — and the comment in `main.py::_warm_providers` that had cited that
  32s figure. If `containerConcurrency: 10` was deliberate, it is a real
  change to make and load-test, not one to leave written where it never
  took effect.
- **"Cold start ~25s to first answer" — measured, and it was mostly not
  cold start.** On the live service `/health` answered in 0.35s, the next
  `/ask` took 20.6s, and everything after was under 0.7s (including a novel
  question through the full three-call pipeline). A container serving
  `/health` that fast is already up; the cost was lazy provider init —
  `get_embeddings()` building its client, acquiring credentials and opening
  a connection on first use. `lifespan` now warms it on a background thread
  (`ENABLE_STARTUP_WARMUP`, default true). See `f7f7597`.
- **`startup-cpu-boost` was declared in `cloudrun-vertexai.yaml` but not
  actually enabled on the live service** — the 08-22 review marked it done
  from the YAML line without checking the running config. Enabled
  2026-08-29 (revision `rag-capstone-00064-xh9`); live annotations now
  carry `run.googleapis.com/startup-cpu-boost: "true"`. The broader
  file-vs-live divergence it exposed is listed under Open.
- **Signing in lowered the upload ceiling.** `upload_limits()` swapped
  between anon and authed config values with no floor, and the shipped
  defaults were 50MB anon vs 10MB authed. Defaults are now 2MB/3 files
  (matching production), and the authed values are floored at the anonymous
  ones in code so the inversion can't return via a single env var. See
  `7dcb54d`.
- **Nine dead `app/*.py` paths** left over from `ebf0b35`'s package
  reorganisation — 7 in `.env.example`, 2 in the Cloud Run YAMLs that the
  review itself missed. `tests/test_doc_paths.py` now fails on any
  unresolvable `app/...py` reference outside `notes/`.
- **A failed upload becoming a permanent public document** — verified
  closed, not assumed. The local-disk path is guarded in `ingest.run()`'s
  ownership gate and covered by
  `test_ingest_refuses_an_upload_with_no_owning_session`. The production
  path stages to GCS, which `ingest.run()` never globs, on a bucket with
  public access prevention enforced and a 7-day delete rule (both confirmed
  live via `gcloud storage buckets describe`).
- **Upload/rate abuse bounds were imperative-only in every Cloud Run YAML.**
  `MAX_UPLOAD_FILES`, `MAX_UPLOAD_SIZE_MB` and `RATE_LIMIT` were declared in
  0 of 4 configs, so a `gcloud run services replace` dropped them to
  `config.py`'s defaults — 50MB x 5 files, a 250MB request on an endpoint
  anonymous visitors reach. Not hypothetical: **live staging was already in
  that state**, running the code defaults rather than 2MB x 3, because the
  imperative update was never re-run after its last replace. All four YAMLs
  now declare them (plus `MAX_CORPUS_CHUNKS`/`MAX_SESSION_CHUNKS`, absent
  entirely from both groq configs — same defect, same group), and both live
  services were updated to match. Verified via `/config` on each.

- CORS reflected any origin with credentials on — closed by turning off
  `allow_credentials` (nothing in this app uses cookie-based auth). See
  `app/main.py`.
- C1/C2/C3 (path traversal → stored XSS, cross-session upload leak,
  wrong-instance job execution) — closed 2026-08-23, see `2601373`.
- `GROQ_CHAT_MODEL` default 404'd (Groq stopped serving Llama). Default is
  now `openai/gpt-oss-20b`, with a matching price row.
- Spend ceiling was blind to `/upload` — embedding calls now go through
  `_CostTrackingEmbeddings` and feed the same `add_usage()`/budget path as
  chat calls.
- Blocking Cloud DLP call in `redact_log_fields()` ran on the event loop —
  now wrapped in `asyncio.to_thread`.
- `python -m app.ingestion.ingest --help` ran a real ingestion instead of
  printing help, and `-f` silently no-op'd — moved to `argparse`.
- ~249MB of eval-only deps (`ragas`, `datasets`, and their transitive deps
  like `pandas`/`scipy`/`openai`) shipped in the runtime image. Moved to a
  new `eval` dependency group in `pyproject.toml`, excluded from the
  Dockerfile's plain `uv sync --frozen`; `.github/workflows/eval.yml` now
  runs `uv sync --frozen --group eval`. Verified: `uv lock` regenerated
  cleanly, both sync paths resolve, `docker build` succeeds and the built
  image genuinely lacks ragas/datasets/pandas/scipy/openai, and the full
  test suite + ruff still pass against the leaner default `.venv`.
- `.github/codeql/` — untracked, unwired to any workflow. Deleted rather
  than wired up (decision made 2026-08-24; revisit if CodeQL scanning is
  wanted later, it'd need a `codeql.yml` workflow authored from scratch).
- **Nothing checked the advertised demo URL is alive.** Added
  `.github/workflows/uptime.yml`: a scheduled Action hits `/health` on the
  canonical `*-uc.a.run.app` hostname every 15 minutes and fails (GitHub
  emails the repo owner/watchers by default) if it doesn't answer. Chosen
  over Cloud Scheduler + Cloud Monitoring specifically because it needs no
  GCP setup beyond this file.
- **Firebase authorized-domains assumes one hostname — checked, not
  currently live.** Neither `cloudrun-*.yaml` sets `FIREBASE_WEB_API_KEY`/
  `FIREBASE_AUTH_DOMAIN` in production or staging, and `main.py`'s `/config`
  deliberately blanks `firebase.project_id` whenever `FIREBASE_WEB_API_KEY`
  is unset (see `app/config.py`) specifically so `ui.html` hides sign-in
  rather than show a button that can't work. No client-side Firebase flow
  runs today, so the authorized-domains list has nothing to be wrong
  about yet. Revisit this the day Firebase sign-in is actually turned on
  (both hostnames will need to be in the console's authorized-domains list
  at that point, not just the canonical one).
- **`DAILY_BUDGET_USD` ships disabled (`0`).** Set to `0.25` in all four
  `cloudrun-*.yaml` files and applied live to both production and staging
  via `gcloud run services update` (cd.yml deploys by image tag only and
  never applies the YAML, so the YAML edit alone wouldn't have taken
  effect). Figure grounded in the project's real Cloud Billing budget —
  found an existing "rag-capstone monthly guard" of ₹2000/month
  (≈$0.70/day total at the time, across the whole project) and picked
  $0.25/day for the LLM+embedding slice specifically, leaving headroom for
  the rest. Enabling the Cloud Billing API to read that budget was itself
  a one-time step (`gcloud services enable cloudbilling.googleapis.com`).
- `AGENTS.md` — was untracked; now committed, same footing as `CLAUDE.md`.
- `DEMO.md` — confirmed intentionally untracked per CLAUDE.md's repository
  voice rule (job-search/presentation material stays out of git). Not a
  gap; left as-is.
