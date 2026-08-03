"""
Ingestion pipeline: load documents from DOCS_DIR -> split into chunks ->
embed -> persist into a local Chroma vector store.

Run standalone to (re)build the index:
    python -m app.ingest

Covers three JD requirements directly:
  - "ingestion" (end-to-end) -- multi-format loaders + per-file error
    isolation, so one bad file doesn't kill the whole batch.
  - "freshness pipelines" -- incremental re-ingestion via content hashing:
    unchanged files are skipped entirely (no re-embedding cost), changed
    files have their old chunks deleted and replaced, new files are added.
    A manifest (chroma_db/ingest_manifest.json) tracks per-file hash and
    last-ingested timestamp.

This is deliberately separate from the API. In a real system you'd run
ingestion as its own job (triggered on document upload, on a schedule,
or via a CLI) rather than inside the request path of a chat endpoint.
"""
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app import config
from app.providers import get_embeddings

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

MANIFEST_FILENAME = "ingest_manifest.json"


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


def _manifest_path() -> Path:
    return Path(config.CHROMA_DIR) / MANIFEST_FILENAME


def _load_manifest() -> dict:
    path = _manifest_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def run(force: bool = False) -> dict:
    """
    Incremental ingestion. Set force=True to re-embed every file
    regardless of whether its content hash changed (e.g. after switching
    MODEL_PROVIDER, since embeddings from different providers aren't
    compatible with each other in the same collection).

    Returns a summary dict: {"added": [...], "updated": [...],
    "skipped_unchanged": [...], "failed": [{"file": ..., "error": ...}]}.
    """
    files = _discover_files(config.DOCS_DIR)
    if not files:
        supported = ", ".join(sorted(LOADER_BY_EXTENSION))
        raise FileNotFoundError(
            f"No supported files found in '{config.DOCS_DIR}'. "
            f"Supported extensions: {supported}"
        )

    manifest = _load_manifest()
    vector_store = _get_vector_store()

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
            vector_store.delete(where={"source": path})

        vector_store.add_documents(chunks)

        manifest[path] = {
            "hash": current_hash,
            "last_ingested": datetime.now(timezone.utc).isoformat(),
            "num_chunks": len(chunks),
        }
        (summary["updated"] if is_changed else summary["added"]).append(path)

    # Manifest entries for files that no longer exist on disk are left in
    # place deliberately (not auto-deleted from the vector store) -- a
    # missing file is ambiguous (moved? temporarily unavailable? deleted
    # on purpose?) and silently deleting embeddings on a glob miss is a
    # worse failure mode than a stale manifest entry. Handle removals
    # explicitly if you need that -- see README.
    _save_manifest(manifest)
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
