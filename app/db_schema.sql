-- Schema for the RAG Capstone PostgreSQL + pgvector database.
-- Executed idempotently by database.init_db() on app startup.

CREATE EXTENSION IF NOT EXISTS vector;

-- Document chunks: the primary store for retrieval.
-- content_tsv is a generated column that Postgres maintains automatically —
-- no separate BM25 rebuild step ever needed.
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSION}),
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    content_hash TEXT NOT NULL,
    metadata JSONB,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW, not IVFFLAT. IVFFLAT clusters existing rows into `lists` centroids
-- at BUILD time, and init_db() runs before anything is ingested -- so the
-- index was always built on an empty table, leaving degenerate centroids.
-- With the default ivfflat.probes=1 a query then scans a single nearly-empty
-- list: measured directly, a 12-candidate request against a 30-row table
-- returned 2 rows, silently starving the vector half of hybrid retrieval and
-- making answers depend on whether full-text search alone happened to hit.
-- HNSW builds an incrementally-maintained graph with no training step, so
-- creating it before the data exists is fine -- which is exactly what this
-- idempotent create-on-startup schema does.
DROP INDEX IF EXISTS idx_chunks_embedding;
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (content_tsv);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_source_hash ON chunks (source, content_hash);

-- Session scoping. Added by ALTER rather than in the CREATE above on
-- purpose: CREATE TABLE IF NOT EXISTS is a no-op against the existing table,
-- so columns declared there would never reach a deployed database. ADD COLUMN
-- IF NOT EXISTS keeps init_db() idempotent AND actually migrates.
--
--   session_id IS NULL  -> curated docs/ corpus, visible to everyone, forever
--   session_id = '<id>' -> a visitor's own upload, visible only to them
--
-- Without this every upload landed in one shared corpus, so any visitor could
-- change what every later visitor saw -- including leaving a prompt-injection
-- payload in place for the next person.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks (session_id);
-- Partial: only uploaded chunks have an expiry, curated ones never do.
CREATE INDEX IF NOT EXISTS idx_chunks_expires ON chunks (expires_at)
    WHERE expires_at IS NOT NULL;

-- Semantic cache: stores previously answered Q&A pairs for similarity lookup.
CREATE TABLE IF NOT EXISTS semantic_cache (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    groundedness TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSION}),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Same reasoning as idx_chunks_embedding_hnsw above. It matters more here
-- if anything: the cache starts empty by definition, so an IVFFLAT index
-- over it was guaranteed to be built with no rows to cluster.
DROP INDEX IF EXISTS idx_cache_embedding;
CREATE INDEX IF NOT EXISTS idx_cache_embedding_hnsw ON semantic_cache USING hnsw (embedding vector_cosine_ops);

-- Ingest manifest: per-file content hash + last-ingested timestamp, used
-- for incremental re-ingestion (skip unchanged files). Lives here rather
-- than in a local JSON file so it can never desync from the data it
-- describes -- a local manifest file has no way of knowing it's pointed
-- at a different (e.g. freshly created) database than the one it was
-- last written against, and would wrongly skip files that were never
-- actually ingested into the database currently in use.
CREATE TABLE IF NOT EXISTS ingest_manifest (
    source TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    num_chunks INT NOT NULL,
    last_ingested TIMESTAMPTZ DEFAULT now()
);
