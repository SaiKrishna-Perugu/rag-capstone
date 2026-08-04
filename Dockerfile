FROM python:3.13-slim

WORKDIR /app

# Install uv directly from the official image (avoids pip entirely)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
# Install project dependencies
RUN uv sync --frozen

COPY . .

# NOTE: this expects chroma_db/ to already exist in the build context --
# i.e. you've run `uv run python -m app.ingest` locally BEFORE building this
# image. We deliberately do NOT run ingest.py during the image build:
# that would require baking an API key into the build (visible in Cloud
# Build logs/layers), which is worth avoiding on principle even for a
# portfolio project. See README "Deploying to GCP (Cloud Run + Vertex AI)"
# for the full ingest-then-build-then-deploy flow.

# Cloud Run injects $PORT (default 8080) and requires the container to
# listen on it. Shell form (not exec-form array) so the env var actually
# expands at container start instead of being read as a literal string.
# Run as non-root for defense-in-depth (Cloud Run best practice).
# Provide a writable home directory for FastEmbed model caching,
# and ensure runtime directories are owned by the non-root user.
RUN adduser --disabled-password appuser \
    && mkdir -p logs docs chroma_db \
    && chown -R appuser:appuser logs docs chroma_db
USER appuser

EXPOSE 8080
CMD exec uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
