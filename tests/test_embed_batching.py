"""Embedding batches must fit one provider request.

ingest.py embedded every chunk of a file in a single embed_documents() call
until a 151KB HTML upload hit Vertex AI's ceiling: "input token count is
33360 but the model supports up to 20000". Nothing about the document was
wrong -- only the batch.
"""
from unittest.mock import MagicMock, patch

from app.ingestion.ingest import _embed_in_batches


class _Embeddings:
    """Records the batches it was handed, and returns one vector per input."""

    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t))] for t in texts]


def test_large_file_is_split_into_several_requests():
    emb = _Embeddings()
    # 60 chunks x 800 chars = 48000 chars, over the 40000 default.
    contents = ["x" * 800 for _ in range(60)]

    result = _embed_in_batches(emb, contents)

    assert len(emb.batches) > 1, "a file over the budget must not go in one request"
    assert all(sum(len(t) for t in b) <= 40_000 for b in emb.batches)
    # Every chunk embedded exactly once, order preserved.
    assert len(result) == 60
    assert sum(len(b) for b in emb.batches) == 60


def test_small_file_still_goes_in_one_request():
    """Batching must not cost an extra round trip on the common case."""
    emb = _Embeddings()
    result = _embed_in_batches(emb, ["x" * 800 for _ in range(5)])

    assert len(emb.batches) == 1
    assert len(result) == 5


def test_item_count_ceiling_applies_independently_of_size():
    """Vertex caps instances per request as well as tokens, so many tiny
    chunks must still be split."""
    emb = _Embeddings()
    contents = ["x" for _ in range(450)]  # trivial chars, 450 items

    _embed_in_batches(emb, contents)

    assert len(emb.batches) > 1
    assert all(len(b) <= 200 for b in emb.batches)


def test_order_is_preserved_across_batches():
    """Embeddings are zipped against contents/hashes by position downstream;
    reordering would attach every vector to the wrong chunk."""
    emb = _Embeddings()
    contents = ["y" * (i + 1) for i in range(300)]

    result = _embed_in_batches(emb, contents)

    assert [v[0] for v in result] == [float(i + 1) for i in range(300)]


def test_single_oversized_chunk_is_sent_rather_than_dropped():
    emb = _Embeddings()
    result = _embed_in_batches(emb, ["z" * 100_000])

    assert len(result) == 1
    assert emb.batches == [["z" * 100_000]]


def test_empty_input_makes_no_request():
    emb = _Embeddings()
    assert _embed_in_batches(emb, []) == []
    assert emb.batches == []


def test_retry_of_a_failed_job_keeps_the_original_error():
    """Cleanup runs on failure, so a Cloud Tasks retry always finds the files
    gone. Reporting that would replace the real cause -- which is what the
    visitor reads -- with a storage message that explains nothing."""
    from app.ingestion import jobs

    job = {
        "session_id": "s1",
        "files": ["big.html"],
        "status": "failed",
        "error": "400 INVALID_ARGUMENT ... input token count is 33360",
    }
    with patch.object(jobs, "get_job", return_value=job), \
         patch.object(jobs, "update_job_status") as mock_update, \
         patch.object(jobs.storage, "enabled", return_value=True), \
         patch.object(jobs.storage, "fetch_to", return_value=[]), \
         patch.object(jobs, "ingest", MagicMock()):
        try:
            jobs.process_job("job-1")
        except RuntimeError as exc:
            assert "input token count" in str(exc)
        else:
            raise AssertionError("expected the original error to be re-raised")

    # "processing" is still set, but the error must never be overwritten.
    assert not any(
        call.args[1] == "failed" for call in mock_update.call_args_list
    ), "a retry must not rewrite the terminal error"
