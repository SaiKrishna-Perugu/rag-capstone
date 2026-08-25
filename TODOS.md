# TODOs

Deferred decisions and open items that aren't tracked anywhere else. The
08-22 pre-review audit flagged that none of this was written down anywhere,
which is part of how the CORS gap below went unnoticed for as long as it did.
Keep entries short; link to the commit or review doc that has the detail.

## Open

- **Firebase authorized-domains assumes one hostname.** Cloud Run serves
  each service under two hostnames (legacy `*-uc.a.run.app` and newer
  `*.<region>.run.app`, see CLAUDE.md's Deployment section) but the Firebase
  Auth console's authorized-domains list may only have one of them. Needs a
  check in the Firebase console — not fixable from this repo.
- **Nothing checks the advertised demo URL is alive.** No uptime check, no
  alert. Candidate: a Cloud Scheduler job hitting `/health` plus a Cloud
  Monitoring alerting policy, or a scheduled GitHub Action as a cheaper
  alternative.
- **`DAILY_BUDGET_USD` ships disabled (`0`).** Deliberately not set this
  session — `/metrics` is per-process and pull-only, so a real ceiling needs
  a number pulled from the Cloud Billing report, not guessed. When it is
  set: put it in the `cloudrun-*.yaml` files, not just `gcloud run services
  update` — an imperative-only value resets on the next
  `gcloud run services replace` (see `UPLOAD_BUCKET`'s commit for the same
  pattern already fixed once).
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
- `AGENTS.md` — was untracked; now committed, same footing as `CLAUDE.md`.
- `DEMO.md` — confirmed intentionally untracked per CLAUDE.md's repository
  voice rule (job-search/presentation material stays out of git). Not a
  gap; left as-is.
