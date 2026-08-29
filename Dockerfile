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
# Install project dependencies
RUN uv sync --frozen

COPY . .

# NOTE: documents are now ingested into PostgreSQL (Cloud SQL + pgvector),
# not a local chroma_db/ directory. Run `uv run python -m app.ingestion.ingest`
# after deploying with DATABASE_URL configured, or as part of a CI/CD
# step that has access to the database.

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
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Cloud Run injects $PORT (default 8080) and requires the container to
# listen on it. Shell form (not exec-form array) so the env var actually
# expands at container start instead of being read as a literal string.
# Run as non-root for defense-in-depth (Cloud Run best practice).
# Provide a writable home directory for FastEmbed model caching,
# and ensure runtime directories are owned by the non-root user.
RUN adduser --disabled-password appuser \
    && mkdir -p logs docs chroma_db \
    && chown -R appuser:appuser logs docs chroma_db .fastembed_cache
USER appuser

EXPOSE 8080
CMD exec uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
