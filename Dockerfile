# Deliberately a floating patch tag, NOT a digest pin. A review flagged all
# three images here as mutable supply-chain inputs, which is true -- the same
# git SHA can rebuild on different bytes. But this repo has no dependabot or
# renovate, so a digest pin would never be bumped, and the base image would
# stop receiving CVE fixes for a service that is on the public internet.
# Floating on 3.13-slim trades build reproducibility for patch delivery,
# which is the right way round here. Pin by digest the day something exists
# to update the pin.
FROM python:3.13-slim

WORKDIR /app

# Install uv directly from the official image (avoids pip entirely)
# Version-pinned, unlike the base image above, and for the opposite reason:
# `:latest` here is unbounded, so a uv major release could change what
# `uv sync --frozen` does between two builds of the same commit. uv is a
# build-time tool with no runtime attack surface in the shipped image, so
# pinning it costs no security patching -- it only removes a way for the
# build to change under us. Bump deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

# Install OS-level dependencies for psycopg2 (libpq) -- psycopg2-binary
# bundles its own libpq, but having the system one avoids edge cases in
# some slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
# --no-dev, not a bare sync: uv treats `dev` as a DEFAULT dependency group, so
# a plain `uv sync --frozen` installed the test toolchain into the runtime
# image. Measured by diffing `uv export` with and without the flag: 140 -> 134
# packages, dropping pytest, pytest-asyncio, ruff and their transitive
# iniconfig/pluggy/pygments. (httpx is NOT in that set -- it stays either way
# as a runtime dependency of the google/langchain clients.)
#
# Six packages is small, and size is not the argument: a test runner and a
# linter have no job inside a container on the public internet, and every CVE
# in either is otherwise a CVE in production. The `eval` group is already
# excluded for exactly this reason (see pyproject.toml), which is why this one
# was easy to miss -- that comment says the image "doesn't ship eval-only deps"
# and is silent on dev, which uv installs unless told not to.
RUN uv sync --frozen --no-dev

# --- Model weights, baked in BEFORE the source copy -------------------------
# Ordering is deliberate and load-bearing for build time. These two downloads
# depend only on what `uv sync` installed, never on application source, so
# placing them above `COPY . .` keeps them in a layer that survives every
# ordinary commit. Below it they re-ran on every single push -- ~98MB of model
# weights re-downloaded per CD build for a one-line change.

# Pre-download the FastEmbed embedding model (used for MODEL_PROVIDER=groq
# -- Groq has no embeddings API) into the image at build time. Without
# this, every fresh container instance downloads it from Hugging Face on
# first use into /tmp (fastembed's default cache, which never persists
# across instances anyway) -- and Cloud Run's shared outbound IP range
# routinely hits HF's anonymous-API rate limit (429), which surfaces as a
# hard crash on the first request a cold instance handles. Baking it in
# removes the runtime network dependency entirely. Must match
# GROQ_EMBEDDING_MODEL's default in app/config.py -- if you override that
# env var, rebuild with a matching model name here too.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN uv run --frozen --no-dev python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Pre-download the FlashRank reranker model into the image at build time
# so containers never hit Hugging Face / external network at runtime.
# Must match RERANKER_PROVIDER=flashrank's default model in app/config.py.
ENV FLASHRANK_CACHE_DIR=/app/.flashrank_cache
RUN uv run --frozen --no-dev python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/.flashrank_cache')"

# Application source last, so a code change invalidates only this layer and
# everything expensive above it is reused.
#
# NOTE: documents are ingested into PostgreSQL (Cloud SQL + pgvector), not
# into anything baked in here. Run `uv run python -m app.ingestion.ingest`
# against a configured DATABASE_URL after deploying -- ingestion is
# deliberately decoupled from the image and from the deploy pipeline.
COPY . .

# Cloud Run injects $PORT (default 8080) and requires the container to
# listen on it. Shell form (not exec-form array) so the env var actually
# expands at container start instead of being read as a literal string.
# Run as non-root for defense-in-depth (Cloud Run best practice).
# `docs` and `logs` are created rather than copied: .dockerignore excludes
# both, and ingest.py globs DOCS_DIR while the request logger writes to logs/.
RUN adduser --disabled-password appuser \
    && mkdir -p logs docs \
    && chown -R appuser:appuser logs docs .fastembed_cache .flashrank_cache
USER appuser

EXPOSE 8080
# --frozen --no-dev on the runtime invocation too, not just at build time:
# `uv run` re-syncs the environment before executing, so a bare `uv run` here
# would reinstall the dev group the build just excluded AND could attempt lock
# resolution -- i.e. a network round trip -- during a cold start.
CMD exec uv run --frozen --no-dev uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
