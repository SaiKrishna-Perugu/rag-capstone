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

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (content_tsv);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_source_hash ON chunks (source, content_hash);

-- Semantic cache: stores previously answered Q&A pairs for similarity lookup.
CREATE TABLE IF NOT EXISTS semantic_cache (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    groundedness TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSION}),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cache_embedding ON semantic_cache USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
