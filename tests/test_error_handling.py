"""Ingestion failures must be actionable to the visitor and detailed in logs.

Everything here guards the same rule from a different angle: `/jobs/{id}`
returns the job record verbatim and ui.html renders `job.error` directly, so
whatever process_job() writes there is read by an anonymous browser. It used
to be `str(exc)` -- a Vertex client error complete with request URL, project
id and the upload bucket's name, for a file whose only problem was length.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.errors import classify

# The real message from the live service, verbatim.
_VERTEX_TOKEN_ERROR = (
    "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Unable to submit "
    "request because the input token count is 33360 but the model supports up to "
    "20000. Reduce the input token count and try again."
)
_GCS_404 = (
    "404 GET https://storage.googleapis.com/download/storage/v1/b/"
    "rag-capstone-uploads-hybrid-rag-505311/o/uploads%2Fabc%2Ffile.html?alt=media: "
    "No such object: rag-capstone-uploads-hybrid-rag-505311/uploads/abc/file.html"
)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (_VERTEX_TOKEN_ERROR, "document_too_large"),
        (_GCS_404, "storage_unavailable"),
        ("429 RESOURCE_EXHAUSTED: quota exceeded for embeddings", "provider_busy"),
        ("503 Service Unavailable", "provider_unavailable"),
        ("psycopg2.OperationalError: connection refused", "database_unavailable"),
        ("PdfReadError: EOF marker not found", "unsupported_content"),
        ("something nobody has seen before", "internal"),
    ],
)
def test_known_failures_are_classified(raw, code):
    assert classify(raw).code == code


@pytest.mark.parametrize("raw", [_VERTEX_TOKEN_ERROR, _GCS_404])
def test_classified_messages_leak_no_infrastructure(raw):
    """The whole point: no bucket names, hostnames, project ids or URLs."""
    message = classify(raw).message
    lowered = message.lower()
    for leak in ("http", "googleapis", "rag-capstone", "bucket", "psycopg",
                 "invalid_argument", "traceback", "0x"):
        assert leak not in lowered, f"{leak!r} leaked into a visitor-facing message"


def test_classify_never_raises():
    """A classifier that throws would replace the real error at the exact
    moment someone is trying to read it."""
    class Hostile(Exception):
        def __str__(self):
            raise ValueError("no string for you")

    assert classify(Hostile()).code == "internal"


def _job(**over):
    base = {"session_id": None, "files": [], "status": "pending", "error": None}
    base.update(over)
    return base


def test_provider_error_reaches_the_browser_classified_not_raw():
    from app.ingestion import jobs

    with patch.object(jobs, "get_job", return_value=_job()), \
         patch.object(jobs, "update_job_status") as mock_update, \
         patch.object(jobs, "_cleanup_files"), \
         patch.object(jobs.ingest, "run", side_effect=RuntimeError(_VERTEX_TOKEN_ERROR)):
        with pytest.raises(RuntimeError):
            jobs.process_job("j1")

    failed = [c for c in mock_update.call_args_list if c.args[1] == "failed"]
    assert failed, "the job must be recorded as failed"
    message = failed[-1].kwargs["error"]
    assert "too large" in message
    assert "33360" not in message and "INVALID_ARGUMENT" not in message
    assert failed[-1].kwargs["error_code"] == "document_too_large"


def test_a_run_that_indexed_nothing_is_not_reported_as_done():
    """The worst available outcome was telling the visitor their upload
    succeeded, then answering their question without the document."""
    from app.ingestion import jobs

    summary = {
        "added": [], "updated": [], "skipped_unchanged": [],
        "failed": [{"file": "docs/uploads/s1/big.html", "error": _VERTEX_TOKEN_ERROR}],
    }
    with patch.object(jobs, "get_job", return_value=_job(session_id=None, files=[])), \
         patch.object(jobs, "update_job_status") as mock_update, \
         patch.object(jobs, "_cleanup_files"), \
         patch.object(jobs.ingest, "run", return_value=summary):
        with pytest.raises(RuntimeError):
            jobs.process_job("j2")

    statuses = [c.args[1] for c in mock_update.call_args_list]
    assert "done" not in statuses
    assert statuses[-1] == "failed"
    assert mock_update.call_args_list[-1].kwargs["error_code"] == "document_too_large"


def test_partial_success_is_done_but_names_the_missing_file():
    from app.ingestion import jobs

    summary = {
        "added": ["docs/uploads/s1/good.txt"], "updated": [], "skipped_unchanged": [],
        "failed": [{"file": "docs/uploads/s1/bad.html", "error": _VERTEX_TOKEN_ERROR}],
    }
    with patch.object(jobs, "get_job", return_value=_job()), \
         patch.object(jobs, "update_job_status") as mock_update, \
         patch.object(jobs, "_cleanup_files"), \
         patch.object(jobs.ingest, "run", return_value=summary), \
         patch("app.db.database.invalidate_cache"):
        jobs.process_job("j3")

    last = mock_update.call_args_list[-1]
    assert last.args[1] == "done"
    assert "bad.html" in last.kwargs["warning"]
    # The path is not the filename -- the visitor never saw docs/uploads/<id>/.
    assert "docs/uploads" not in last.kwargs["warning"]


def test_one_bad_file_does_not_lose_the_good_ones():
    """Embedding and upsert are inside the per-file guard. They were not, so a
    provider error on file 3 discarded files 1 and 2 as well."""
    from app.ingestion import ingest

    def _embed(_embeddings, contents):
        if any("boom" in c for c in contents):
            raise RuntimeError(_VERTEX_TOKEN_ERROR)
        return [[0.0] for _ in contents]

    docs = {"a.txt": "fine", "b.txt": "boom", "c.txt": "fine too"}
    chunks = {k: [MagicMock(page_content=v, metadata={})] for k, v in docs.items()}

    with patch.object(ingest, "_discover_files", return_value=list(docs)), \
         patch.object(ingest, "_file_hash", side_effect=lambda p: "h-" + p), \
         patch.object(ingest, "_load_one_file", side_effect=lambda p: chunks[p]), \
         patch.object(ingest, "chunk_documents", side_effect=lambda d: d), \
         patch.object(ingest, "_embed_in_batches", side_effect=_embed), \
         patch.object(ingest, "get_embeddings", return_value=MagicMock()), \
         patch.object(ingest.database, "init_db"), \
         patch.object(ingest.database, "get_manifest", return_value={}), \
         patch.object(ingest.database, "upsert_chunks"), \
         patch.object(ingest.database, "upsert_manifest_entry"):
        summary = ingest.run()

    assert sorted(summary["added"]) == ["a.txt", "c.txt"]
    assert [f["file"] for f in summary["failed"]] == ["b.txt"]
