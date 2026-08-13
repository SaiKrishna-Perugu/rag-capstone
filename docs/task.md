
# Task List — Production RAG Upgrade

## Phase 1: Critical Production Fixes (Security + Stability)

### 1.1 API Security & Rate Limiting
- [/] Create `app/middleware.py` — API key auth middleware
- [/] Add security config to `app/config.py` (API_KEY, CORS_ORIGINS, RATE_LIMIT, etc.)
- [/] Wire CORS + rate limiter + API key middleware into `app/main.py`

### 1.2 Input Validation & Sanitization
- [/] Sanitize `file.filename` in `/upload` (path traversal fix)
- [/] Validate file extension against supported types
- [/] Add file size limit check
- [/] Add `max_length` to question field in `AskRequest`

### 1.3 Async Request Handling
- [/] Convert `/ask` to `async def` with `asyncio.to_thread()`
- [/] Convert `/ask-agentic` to `async def` with `asyncio.to_thread()`
- [/] Convert `/upload` ingestion to `asyncio.to_thread()`

### 1.4 Graceful Error Handling & Retries
- [/] Add `max_retries` to LLM providers in `app/providers.py`
- [/] Add `LLM_MAX_RETRIES` to `app/config.py`

### 1.5 Health Check Enhancement
- [/] Enhance `/health` with vector store dependency check
- [/] Add `/ready` endpoint (readiness probe for Cloud Run)

### 1.6 Dependencies & Config
- [/] Add `slowapi` to `pyproject.toml`
- [/] Add dev dependencies (`pytest`, `ruff`, etc.)
- [ ] Update `.env.example` with new settings
- [ ] Run `uv lock`

## Phase 2: Production Observability & Monitoring
- [x] Create `app/metrics.py` — request/groundedness counters
- [x] Add `GET /metrics` endpoint
- [x] Add request ID to all log entries

## Phase 3: RAG Pipeline Improvements
- [ ] Create `app/memory.py` — conversation memory
- [ ] Create `app/cache.py` — semantic caching
- [ ] Create `app/streaming.py` — SSE streaming responses
- [ ] Add `/ask-stream` endpoint
- [ ] Add multi-document upload

## Phase 4: Testing & Quality Assurance
- [ ] Create `tests/conftest.py` — shared fixtures
- [ ] Create `tests/test_ingest.py`
- [ ] Create `tests/test_retrieval.py`
- [ ] Create `tests/test_rag.py`
- [ ] Create `tests/test_agent.py`
- [ ] Create `tests/test_api.py`
- [ ] Create `.github/workflows/ci.yml`

## Phase 5: Cloud Infrastructure
- [ ] Create `cloud-run-service.yaml`
- [ ] Add Secret Manager integration to `config.py`
- [ ] Migrate from GCR to Artifact Registry in README

## Phase 6: UI Enhancement
- [ ] Add chat history to `ui.html`
- [ ] Add streaming display
- [ ] Improve mobile responsiveness
- [ ] Add copy answer button

## Phase 7: Documentation & Portfolio Polish
- [ ] Add Mermaid architecture diagram to README
- [ ] Update project structure in README
- [ ] Update `.env.example` with all new settings
