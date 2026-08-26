# TODOs

Deferred decisions and open items that aren't tracked anywhere else. The
08-22 pre-review audit flagged that none of this was written down anywhere,
which is part of how the CORS gap below went unnoticed for as long as it did.
Keep entries short; link to the commit or review doc that has the detail.

## Open

(nothing currently open from the 08-22 review)

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
