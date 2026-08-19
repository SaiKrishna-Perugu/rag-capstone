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
def get_conn(register_types: bool = True):
    """Context manager that checks out a connection from the pool,
    registers the pgvector type, and returns it on exit.

    register_types=False skips registering the vector type adapter --
    needed by init_db() specifically, since register_vector() looks up
    the `vector` type's OID in pg_type, which doesn't exist yet on a
    brand-new database until init_db()'s own `CREATE EXTENSION IF NOT
    EXISTS vector` has run. Every other caller needs the adapter (they
    bind numpy arrays as query parameters) and uses the default.
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        if register_types:
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

    with get_conn(register_types=False) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
    logger.info("Database schema initialised (idempotent).")


# ---------------------------------------------------------------------------
# Chunk operations (used by ingest.py and retrieval.py)
# ---------------------------------------------------------------------------

def get_chunk_count(session_id: str | None = None) -> int:
    """Chunks in the store. Used by /ready (total) and by /upload's caps.

    With `session_id`, counts only that visitor's own uploaded chunks, which
    is what the per-session ceiling is enforced against -- so one visitor
    cannot consume the whole global budget.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if session_id is None:
                cur.execute("SELECT COUNT(*) FROM chunks")
            else:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE session_id = %s", (session_id,))
            return cur.fetchone()[0]


def upsert_chunks(
    source: str,
    contents: list[str],
    embeddings: list[list[float]],
    content_hashes: list[str],
    metadatas: list[dict],
    session_id: str | None = None,
    expires_at=None,
) -> int:
    """Bulk upsert chunks for a given source file.

    Uses ON CONFLICT on (source, content_hash) to handle the freshness
    logic: unchanged chunks are skipped, changed chunks are updated.
    Returns the number of rows affected.

    `session_id`/`expires_at` default to None, which is the curated corpus:
    visible to everyone, never expires. Visitor uploads pass both.
    """
    import numpy as np

    sql = """
        INSERT INTO chunks (source, content, embedding, content_hash, metadata,
                            session_id, expires_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source, content_hash) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            session_id = EXCLUDED.session_id,
            expires_at = EXCLUDED.expires_at,
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
                    session_id,
                    expires_at,
                ))
            cur.executemany(sql, rows)
            return cur.rowcount


def list_session_documents(session_id: str) -> list[dict]:
    """The documents this visitor uploaded, one row per source file.

    Backs the "your documents" list in the UI. Curated corpus files
    (session_id IS NULL) never appear -- a visitor manages their own uploads,
    not the shared sample set.
    """
    sql = """
        SELECT source,
               COUNT(*)            AS chunks,
               MIN(ingested_at)    AS ingested_at,
               MAX(expires_at)     AS expires_at
        FROM chunks
        WHERE session_id = %s
        GROUP BY source
        ORDER BY MIN(ingested_at)
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_id,))
            return [dict(row) for row in cur.fetchall()]


def delete_session_document(session_id: str, source: str) -> int:
    """Delete one visitor's document. Returns rows removed.

    Scoped by session_id as well as source, deliberately. `source` is derived
    from a client-supplied filename, and this is the second of two independent
    checks (main.py rebuilds the path server-side from the caller's own
    session). Either layer alone is one bug away from letting somebody delete
    the curated corpus or another visitor's file.

    Kept separate from delete_chunks_by_source() rather than adding an
    optional filter to it: that one is used by ingest.py for the curated
    corpus, where session_id IS NULL, and an optional parameter there invites
    passing None and deleting far more than intended.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE source = %s AND session_id = %s",
                (source, session_id),
            )
            return cur.rowcount


def delete_manifest_entry(source: str) -> int:
    """Forget that a file was ever ingested.

    Load-bearing, not tidy-up. ingest.run() skips any file whose manifest row
    still matches its content hash, so deleting a document's chunks while
    leaving this row behind means re-uploading the SAME file is silently
    treated as "unchanged" -- ingestion skips it and the chunks are never
    recreated. The document becomes permanently unrecoverable with no error
    raised anywhere.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ingest_manifest WHERE source = %s", (source,))
            return cur.rowcount


def delete_chunks_by_source(source: str) -> int:
    """Delete all chunks for a given source file (used when a file has
    changed and needs full re-ingestion)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
            return cur.rowcount


# ---------------------------------------------------------------------------
# Ingest manifest (used by ingest.py for incremental re-ingestion)
# ---------------------------------------------------------------------------

def get_manifest() -> dict:
    """Return the full ingest manifest as {source: {hash, num_chunks,
    last_ingested}}, matching the shape the old local-JSON manifest used."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT source, content_hash, num_chunks, last_ingested FROM ingest_manifest")
            rows = cur.fetchall()
    return {
        row["source"]: {
            "hash": row["content_hash"],
            "num_chunks": row["num_chunks"],
            "last_ingested": row["last_ingested"].isoformat(),
        }
        for row in rows
    }


def upsert_manifest_entry(source: str, content_hash: str, num_chunks: int) -> None:
    """Record a file as ingested. Written per-file (not batched at the end
    of a run) so an interrupted ingest run doesn't lose progress -- a
    restart re-checks only the files that never got a manifest entry."""
    sql = """
        INSERT INTO ingest_manifest (source, content_hash, num_chunks, last_ingested)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (source) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            num_chunks = EXCLUDED.num_chunks,
            last_ingested = now()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (source, content_hash, num_chunks))


# ---------------------------------------------------------------------------
# Hybrid search (used by retrieval.py)
# ---------------------------------------------------------------------------

# The visibility predicate, in one place. Repeated inline in each CTE rather
# than factored into a `visible` CTE so Postgres can still push it down to the
# HNSW / GIN indexes instead of scanning a materialised intermediate.
#
#   session_id IS NULL -> curated corpus, everyone sees it
#   session_id = :sid  -> this visitor's own upload
#
# A caller with no session (sid IS NULL) matches only the first branch, so an
# anonymous visitor sees the curated corpus and nobody's uploads -- including
# their own from a previous browser. That is the intended default.
_VISIBLE = """
    (session_id IS NULL OR session_id = %(session_id)s)
    AND (expires_at IS NULL OR expires_at > now())
"""


def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    k: int = 10,
    candidate_pool: int = 20,
    session_id: str | None = None,
) -> list[dict]:
    """Single-query hybrid retrieval: vector similarity + full-text search,
    fused with Reciprocal Rank Fusion (RRF, k=60).

    This is the exact same RRF math previously in retrieval.py's
    _reciprocal_rank_fusion() — k=60, unchanged — just executed by Postgres
    instead of Python. Returns dicts with keys: id, source, content, metadata.

    Scoped to what `session_id` may see (see _VISIBLE). Named parameters
    throughout, deliberately: this query binds the same values several times
    and the visibility predicate appears three times, which is exactly the
    shape where positional placeholders get silently mis-ordered.

    Known pgvector caveat: an HNSW scan with a filter post-filters, so a
    heavily-filtered search can return fewer than `candidate_pool` rows. Not a
    practical concern at this corpus size (MAX_CORPUS_CHUNKS is in the
    hundreds), but it would be at scale.
    """
    import numpy as np

    sql = f"""
        WITH vector_ranked AS (
            SELECT id, ROW_NUMBER() OVER (
                ORDER BY embedding <=> %(emb)s
            ) AS rank
            FROM chunks
            WHERE {_VISIBLE}
            ORDER BY embedding <=> %(emb)s
            LIMIT %(pool)s
        ),
        keyword_ranked AS (
            SELECT id, ROW_NUMBER() OVER (
                ORDER BY ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) DESC
            ) AS rank
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('english', %(q)s)
              AND {_VISIBLE}
            LIMIT %(pool)s
        )
        SELECT c.id, c.source, c.content, c.metadata, c.session_id,
               COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
        FROM chunks c
        LEFT JOIN vector_ranked v ON c.id = v.id
        LEFT JOIN keyword_ranked k ON c.id = k.id
        WHERE (v.id IS NOT NULL OR k.id IS NOT NULL)
          AND {_VISIBLE}
        ORDER BY rrf_score DESC
        LIMIT %(k)s
    """
    params = {
        "emb": np.array(query_embedding, dtype=np.float32),
        "q": query_text,
        "pool": candidate_pool,
        "k": k,
        "session_id": session_id,
    }

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def delete_expired_chunks() -> int:
    """Drop uploaded chunks past their TTL. Curated chunks (expires_at NULL)
    are never touched. Called by the internal cleanup endpoint that Cloud
    Scheduler hits daily -- Postgres has no native TTL, unlike Firestore."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE expires_at IS NOT NULL AND expires_at < now()"
            )
            return cur.rowcount


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
