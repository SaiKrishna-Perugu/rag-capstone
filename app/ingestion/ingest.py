"""
Ingestion pipeline: load documents from DOCS_DIR -> split into chunks ->
embed -> persist into PostgreSQL + pgvector.

Run standalone to (re)build the index:
    python -m app.ingestion.ingest

Two properties matter here beyond "it loads files":
  - **Per-file error isolation** -- one unreadable or malformed document is
    recorded and skipped rather than failing the whole batch, because a
    single bad file in a corpus of hundreds should not block the rest.
  - **Incremental re-ingestion** via content hashing:
    unchanged files are skipped entirely (no re-embedding cost), changed
    files have their old chunks deleted and replaced, new files are added.
    A manifest (the `ingest_manifest` table, see app/db/database.py) tracks
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
import logging
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

logger = logging.getLogger(__name__)

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

def _embed_in_batches(embeddings, contents: list[str]) -> list:
    """Embed `contents` in pieces small enough for one provider request.

    A single embed_documents() call over a whole file's chunks is the obvious
    way to write this, and it is what this module did until a 151KB HTML
    upload hit Vertex AI's per-request ceiling: "input token count is 33360
    but the model supports up to 20000". The document was fine and the chunks
    were fine -- only the batch was too big, and the job failed with a
    provider error that named nothing the visitor could act on.

    Batching on characters rather than tokens is deliberate: counting tokens
    means either a tokenizer dependency or a CountTokens round trip per file,
    and the ceiling only has to be *safe*, not tight. See
    config.EMBED_BATCH_MAX_CHARS for the margin and why it is that wide.

    A single chunk over the budget is still sent alone rather than dropped --
    CHUNK_SIZE is far below the limit, so that can only happen if someone
    raises it past the model's ceiling, and failing loudly on the real cause
    beats silently skipping content.
    """
    max_chars = config.EMBED_BATCH_MAX_CHARS
    max_items = config.EMBED_BATCH_MAX_ITEMS
    out: list = []
    batch: list[str] = []
    batch_chars = 0
    for text in contents:
        over_chars = batch and batch_chars + len(text) > max_chars
        over_items = len(batch) >= max_items
        if over_chars or over_items:
            out.extend(embeddings.embed_documents(batch))
            batch, batch_chars = [], 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        out.extend(embeddings.embed_documents(batch))
    return out


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


def run(
    force: bool = False,
    session_id: str | None = None,
    expires_at=None,
    only: list[str] | None = None,
) -> dict:
    """
    Incremental ingestion. Set force=True to re-embed every file
    regardless of whether its content hash changed (e.g. after switching
    MODEL_PROVIDER, since embeddings from different providers aren't
    compatible with each other in the same collection).

    `session_id`/`expires_at` scope a visitor's uploaded documents so they are
    retrievable only by that visitor and expire on their own. They are applied
    **only to files under DOCS_DIR/uploads/<session_id>/**, never to the curated
    corpus and never to another visitor's uploads.

    That second exclusion is the one that bites. This function rescans the whole
    docs tree on every run, so a guard of merely "is this under uploads/?" tags
    EVERY visitor's pending upload with whichever session happens to be running
    -- visitor B loses their document and visitor A silently gains it. The write
    side already separates sessions into their own directories; this is the read
    side agreeing with it.

    `only` restricts ingestion to a specific list of basenames within this
    session's directory. Callers that know exactly which files a job owns
    (jobs.process_job) should pass it: a concurrent visitor's upload sitting in
    the same tree is then never even considered, rather than merely being tagged
    correctly.

    Returns a summary dict: {"added": [...], "updated": [...],
    "skipped_unchanged": [...], "failed": [{"file": ..., "error": ...}]}.
    """
    # Resolved Path objects, compared with is_relative_to() rather than string
    # prefixes. _discover_files() goes through glob.glob(os.path.join(...)), which
    # on Windows returns whatever separator mix DOCS_DIR was written with
    # ("C:/x/y\\uploads\\s\\f.md"), while str(Path(...)) normalises to all
    # backslashes. A startswith() between those two silently matches nothing, so
    # every upload falls through and is written as curated corpus -- caught only
    # by running this against a real database, since tmp_path in the unit tests
    # produces consistent separators and hides it.
    uploads_root = (Path(config.DOCS_DIR) / "uploads").resolve()
    # The directory whose contents this run is allowed to claim ownership of.
    # None when no session is in play (the curated-corpus CLI path).
    session_root = (uploads_root / session_id) if session_id else None
    only_names = set(only) if only else None
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
        resolved = Path(path).resolve()
        is_upload = resolved.is_relative_to(uploads_root)

        # --- Ownership gate -------------------------------------------------
        # Everything below decides whether THIS run is entitled to touch THIS
        # file. It runs before the hash read so an unowned file costs nothing.
        if is_upload:
            if session_id is None:
                # An upload with no owning session. Reachable from the plain CLI
                # (`python -m app.ingestion.ingest`) whenever a /upload wrote
                # files but its job never ran -- Firestore unreachable, enqueue
                # failed. Ingesting it here would write session_id=NULL,
                # expires_at=NULL: a private document silently promoted to the
                # curated corpus, globally visible and never expiring.
                summary["skipped_unchanged"].append(path)
                logger.warning("Skipping %s: upload with no owning session.", path)
                continue
            if not resolved.is_relative_to(session_root):
                # Another visitor's upload, sitting in the same tree. Not ours
                # to tag, and tagging it is exactly the ownership leak.
                continue
            if only_names is not None and resolved.name not in only_names:
                # Ours by directory, but not part of this job's file list.
                continue

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

        # Embedding and persistence are inside the per-file guard for the same
        # reason loading is. They were not, and the asymmetry was invisible
        # until it mattered: a file whose chunks exceeded the embedding
        # provider's per-request ceiling raised out of run() entirely, so a
        # batch of five documents lost the four that were fine, and the job
        # reported a provider error rather than naming the file that caused
        # it. A per-file failure is data about that file.
        try:
            # Batch embed all chunks for this file, in request-sized pieces.
            chunk_embeddings = _embed_in_batches(embeddings, contents)

            # `is_upload` was resolved by the ownership gate at the top of the
            # loop. Reaching here means: either a curated file, or an upload
            # this run is entitled to claim. Only visitor uploads carry a
            # session and an expiry; curated docs stay global and permanent.
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
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", path, exc, exc_info=True)
            summary["failed"].append({"file": path, "error": str(exc)})
            continue
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Incrementally (re-)ingest DOCS_DIR into the vector store."
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Re-embed every file regardless of content hash "
             "(e.g. after switching MODEL_PROVIDER).",
    )
    args = parser.parse_args()

    try:
        summary = run(force=args.force)
        print_summary(summary)
        if summary["failed"] and not (summary["added"] or summary["updated"]):
            sys.exit(1)  # every file failed -- treat as a hard failure
    except Exception as exc:
        print(f"[ingest] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
