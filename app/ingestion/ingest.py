"""
Ingestion pipeline: load documents from DOCS_DIR -> split into chunks ->
embed -> persist into PostgreSQL + pgvector.

Run standalone to (re)build the index:
    python -m app.ingestion.ingest

Covers three JD requirements directly:
  - "ingestion" (end-to-end) -- multi-format loaders + per-file error
    isolation, so one bad file doesn't kill the whole batch.
  - "freshness pipelines" -- incremental re-ingestion via content hashing:
    unchanged files are skipped entirely (no re-embedding cost), changed
    files have their old chunks deleted and replaced, new files are added.
    A manifest (the `ingest_manifest` table, see app/database.py) tracks
    per-file hash and last-ingested timestamp -- it lives in the same
    database as the chunks it describes, not a local file, so it can't
    silently desync from whichever database DATABASE_URL currently points
    at (e.g. a fresh Cloud SQL instance).

This is deliberately separate from the API. In a real system you'd run
ingestion as its own job (triggered on document upload, on a schedule,
or via a CLI) rather than inside the request path of a chat endpoint.
"""
import glob
import hashlib
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.db import database
from app.llm.providers import get_embeddings

# Extension -> loader class. Add new formats here -- everything else
# (chunking, hashing, freshness tracking) works unchanged for any new type.
LOADER_BY_EXTENSION = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": CSVLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".docx": Docx2txtLoader,
}

def _discover_files(docs_dir: str) -> list:
    """Find every file under docs_dir whose extension we have a loader for."""
    files = []
    for ext in LOADER_BY_EXTENSION:
        files.extend(glob.glob(os.path.join(docs_dir, "**", f"*{ext}"), recursive=True))
    return sorted(set(files))


def _load_one_file(path: str) -> list:
    ext = Path(path).suffix.lower()
    loader_factory = LOADER_BY_EXTENSION[ext]
    loader = loader_factory(path)
    return loader.load()


def _file_hash(path: str) -> str:
    """SHA-256 of file contents -- used to detect whether a file has
    actually changed since it was last ingested (freshness check)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _content_hash(text: str) -> str:
    """SHA-256 of chunk content — used as part of the upsert key to
    detect whether a chunk has changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(force: bool = False, session_id: str | None = None, expires_at=None) -> dict:
    """
    Incremental ingestion. Set force=True to re-embed every file
    regardless of whether its content hash changed (e.g. after switching
    MODEL_PROVIDER, since embeddings from different providers aren't
    compatible with each other in the same collection).

    `session_id`/`expires_at` scope a visitor's uploaded documents so they
    are retrievable only by that visitor and expire on their own. They are
    applied **only to files under DOCS_DIR/uploads/**, never to the curated
    corpus. That distinction is load-bearing: this function rescans the whole
    docs tree on every run, so tagging indiscriminately would quietly convert
    the shared sample corpus into one visitor's private documents and make it
    invisible to everyone else.

    Returns a summary dict: {"added": [...], "updated": [...],
    "skipped_unchanged": [...], "failed": [{"file": ..., "error": ...}]}.
    """
    uploads_root = str(Path(config.DOCS_DIR) / "uploads")
    files = _discover_files(config.DOCS_DIR)
    if not files:
        supported = ", ".join(sorted(LOADER_BY_EXTENSION))
        raise FileNotFoundError(
            f"No supported files found in '{config.DOCS_DIR}'. "
            f"Supported extensions: {supported}"
        )

    # Ensure the schema exists -- this is commonly the first command run
    # against a fresh database (before the API has ever started, so its
    # lifespan-hook init_db() call hasn't run yet). Idempotent, so this
    # is a fast no-op on an already-initialised database.
    database.init_db()

    manifest = database.get_manifest()
    embeddings = get_embeddings()

    summary = {"added": [], "updated": [], "skipped_unchanged": [], "failed": []}

    for path in files:
        try:
            current_hash = _file_hash(path)
        except OSError as exc:
            summary["failed"].append({"file": path, "error": f"could not read file: {exc}"})
            continue

        previous = manifest.get(path)
        is_new = previous is None
        is_changed = previous is not None and (force or previous.get("hash") != current_hash)

        if not force and not is_new and not is_changed:
            summary["skipped_unchanged"].append(path)
            continue

        try:
            docs = _load_one_file(path)
            chunks = chunk_documents(docs)
        except Exception as exc:
            # Per-file isolation: one bad file (corrupt PDF, malformed
            # CSV, etc.) is recorded and skipped, not a fatal crash for
            # the whole ingestion run.
            summary["failed"].append({"file": path, "error": str(exc)})
            continue

        if is_changed:
            # Remove this file's old chunks before adding the new ones,
            # so re-ingesting a changed file doesn't leave stale
            # duplicates alongside the fresh content.
            database.delete_chunks_by_source(path)

        # Extract text content and metadata, then embed in batch
        contents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        content_hashes = [_content_hash(c) for c in contents]

        # Batch embed all chunks for this file
        chunk_embeddings = embeddings.embed_documents(contents)

        # Only visitor uploads carry a session and an expiry; curated docs
        # stay global and permanent. See the note in this function's docstring.
        is_upload = path.startswith(uploads_root)

        # Upsert into PostgreSQL
        database.upsert_chunks(
            source=path,
            contents=contents,
            embeddings=chunk_embeddings,
            content_hashes=content_hashes,
            metadatas=metadatas,
            session_id=session_id if is_upload else None,
            expires_at=expires_at if is_upload else None,
        )

        # Written immediately, not batched until the end of the run, so
        # an interrupted run (killed mid-batch, crashed on a later file)
        # doesn't lose progress already committed to the chunks table --
        # a restart re-checks only files that never got a manifest entry.
        database.upsert_manifest_entry(path, current_hash, len(chunks))
        (summary["updated"] if is_changed else summary["added"]).append(path)

    # Manifest entries for files that no longer exist on disk are left in
    # place deliberately (not auto-deleted from the vector store) -- a
    # missing file is ambiguous (moved? temporarily unavailable? deleted
    # on purpose?) and silently deleting embeddings on a glob miss is a
    # worse failure mode than a stale manifest entry. Handle removals
    # explicitly if you need that -- see README.
    return summary


def print_summary(summary: dict) -> None:
    print(f"[ingest] Added:            {len(summary['added'])}")
    print(f"[ingest] Updated:          {len(summary['updated'])}")
    print(f"[ingest] Skipped (fresh):  {len(summary['skipped_unchanged'])}")
    print(f"[ingest] Failed:           {len(summary['failed'])}")
    for failure in summary["failed"]:
        print(f"  - {failure['file']}: {failure['error']}", file=sys.stderr)


if __name__ == "__main__":
    force = "--force" in sys.argv
    try:
        summary = run(force=force)
        print_summary(summary)
        if summary["failed"] and not (summary["added"] or summary["updated"]):
            sys.exit(1)  # every file failed -- treat as a hard failure
    except Exception as exc:
        print(f"[ingest] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
