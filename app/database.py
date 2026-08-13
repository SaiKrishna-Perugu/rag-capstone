"""
Centralized database layer for PostgreSQL + pgvector.

Replaces the local Chroma vector store with an external, shared PostgreSQL
database using the pgvector extension for vector similarity search and
Postgres's built-in tsvector/tsquery for full-text (BM25-equivalent) search.

Why this exists: Cloud Run instances are stateless — each gets its own
filesystem, so a local Chroma DB means every instance has a different,
divergent copy of the index. An external Postgres instance is shared by
all instances, making the system actually correct under autoscaling.

Connection pooling is critical: Cloud Run can spin up many concurrent
instances, each running many concurrent requests. Without pooling, the
connection count would blow past Cloud SQL's limit. We use psycopg2's
ThreadedConnectionPool with conservative defaults.

All functions follow the existing fail-open pattern: transient DB errors
in non-critical paths (cache) degrade gracefully rather than crashing
the request.
"""
import logging
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from psycopg2 import pool

from app import config

logger = logging.getLogger(__name__)

_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool:
    """Lazy-initialise the connection pool on first use."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = pool.ThreadedConnectionPool(
            minconn=config.DATABASE_POOL_MIN,
            maxconn=config.DATABASE_POOL_MAX,
            dsn=config.DATABASE_URL,
        )
    return _pool


@contextmanager
def get_conn():
    """Context manager that checks out a connection from the pool,
    registers the pgvector type, and returns it on exit."""
    p = _get_pool()
    conn = p.getconn()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables and extensions idempotently. Safe to call on every
    startup, and from ingest.py's standalone entrypoint -- uses IF NOT
    EXISTS throughout, so a database that's already initialised is a
    fast no-op.

    The VECTOR column width is templated from config.EMBEDDING_DIMENSION
    (384 for FastEmbed/Groq, 768 for Vertex AI's text-embedding-005) so a
    fresh database matches whichever provider is configured. This only
    takes effect on first creation -- CREATE TABLE IF NOT EXISTS won't
    widen an existing column, so switching providers on a database that's
    already been initialised at the old dimension requires dropping and
    re-creating the `chunks`/`semantic_cache` tables (and re-ingesting),
    same as any other cross-provider embedding-space switch.
    """
    schema_path = Path(__file__).parent / "db_schema.sql"
    schema_sql = schema_path.read_text().replace(
        "{EMBEDDING_DIMENSION}", str(config.EMBEDDING_DIMENSION)
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
    logger.info("Database schema initialised (idempotent).")


# ---------------------------------------------------------------------------
# Chunk operations (used by ingest.py and retrieval.py)
# ---------------------------------------------------------------------------

def get_chunk_count() -> int:
    """Return the total number of chunks in the store. Used by /ready."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]


def upsert_chunks(
    source: str,
    contents: list[str],
    embeddings: list[list[float]],
    content_hashes: list[str],
    metadatas: list[dict],
) -> int:
    """Bulk upsert chunks for a given source file.

    Uses ON CONFLICT on (source, content_hash) to handle the freshness
    logic: unchanged chunks are skipped, changed chunks are updated.
    Returns the number of rows affected.
    """
    import numpy as np

    sql = """
        INSERT INTO chunks (source, content, embedding, content_hash, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source, content_hash) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            ingested_at = now()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            rows = []
            for content, emb, chash, meta in zip(
                contents, embeddings, content_hashes, metadatas
            ):
                rows.append((
                    source,
                    content,
                    np.array(emb, dtype=np.float32),
                    chash,
                    psycopg2.extras.Json(meta),
                ))
            cur.executemany(sql, rows)
            return cur.rowcount


def delete_chunks_by_source(source: str) -> int:
    """Delete all chunks for a given source file (used when a file has
    changed and needs full re-ingestion)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
            return cur.rowcount


# ---------------------------------------------------------------------------
# Hybrid search (used by retrieval.py)
# ---------------------------------------------------------------------------

def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    k: int = 10,
    candidate_pool: int = 20,
) -> list[dict]:
    """Single-query hybrid retrieval: vector similarity + full-text search,
    fused with Reciprocal Rank Fusion (RRF, k=60).

    This is the exact same RRF math previously in retrieval.py's
    _reciprocal_rank_fusion() — k=60, unchanged — just executed by Postgres
    instead of Python. Returns dicts with keys: id, source, content, metadata.
    """
    import numpy as np

    sql = """
        WITH vector_ranked AS (
            SELECT id, ROW_NUMBER() OVER (
                ORDER BY embedding <=> %s
            ) AS rank
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        ),
        keyword_ranked AS (
            SELECT id, ROW_NUMBER() OVER (
                ORDER BY ts_rank(content_tsv, plainto_tsquery('english', %s)) DESC
            ) AS rank
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            LIMIT %s
        )
        SELECT c.id, c.source, c.content, c.metadata,
               COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
        FROM chunks c
        LEFT JOIN vector_ranked v ON c.id = v.id
        LEFT JOIN keyword_ranked k ON c.id = k.id
        WHERE v.id IS NOT NULL OR k.id IS NOT NULL
        ORDER BY rrf_score DESC
        LIMIT %s
    """
    emb = np.array(query_embedding, dtype=np.float32)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (emb, emb, candidate_pool, query_text, query_text, candidate_pool, k))
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Semantic cache operations (used by cache.py)
# ---------------------------------------------------------------------------

def cache_get(
    question_embedding: list[float],
    threshold: float = 0.95,
) -> dict | None:
    """Find the most similar cached question. Returns the cached entry if
    similarity >= threshold, else None.

    pgvector's <=> operator returns cosine *distance* (0 = identical),
    so similarity = 1 - distance.
    """
    import numpy as np

    sql = """
        SELECT question, answer, groundedness,
               1 - (embedding <=> %s) AS similarity
        FROM semantic_cache
        ORDER BY embedding <=> %s
        LIMIT 1
    """
    emb = np.array(question_embedding, dtype=np.float32)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (emb, emb))
            row = cur.fetchone()

    if row is None:
        return None

    if row["similarity"] >= threshold:
        return {
            "answer": row["answer"],
            "groundedness": row["groundedness"],
            "sources": [],
            "cached": True,
            "similarity_score": float(row["similarity"]),
        }
    return None


def cache_set(
    question: str,
    answer: str,
    groundedness: str,
    question_embedding: list[float],
) -> None:
    """Store a Q&A pair in the semantic cache."""
    import numpy as np

    sql = """
        INSERT INTO semantic_cache (question, answer, groundedness, embedding)
        VALUES (%s, %s, %s, %s)
    """
    emb = np.array(question_embedding, dtype=np.float32)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (question, answer, groundedness, emb))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def close_pool() -> None:
    """Close all connections in the pool. Called on app shutdown."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.closeall()
        _pool = None
