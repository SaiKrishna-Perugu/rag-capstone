from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


def _candidates(n: int) -> list:
    return [Document(page_content=f"chunk {i}", metadata={"source": f"s{i}"}) for i in range(n)]


@pytest.mark.parametrize(
    "llm_behavior",
    [
        pytest.param("raise", id="llm_call_fails"),
        pytest.param('{"ranking": [1, 2]}', id="json_object_not_array"),
        pytest.param('"[1, 2]"', id="json_string_not_array"),
        pytest.param("5", id="json_number_not_array"),
        pytest.param("not json at all", id="unparseable"),
    ],
)
def test_rerank_falls_back_to_rrf_order_on_bad_llm_response(llm_behavior):
    """rerank() is a quality optimization, never a correctness requirement:
    every failure mode must degrade to the pre-rerank RRF order rather than
    propagate. Regression test -- llm.invoke() used to sit outside the try,
    so a provider timeout took down the whole retrieval call, and a non-list
    JSON response raised TypeError past the except-tuple."""
    from app.retrieval import rerank

    candidates = _candidates(12)

    with patch("app.retrieval.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        if llm_behavior == "raise":
            mock_llm.invoke.side_effect = TimeoutError("provider timed out")
        else:
            mock_llm.invoke.return_value = MagicMock(content=llm_behavior)
        mock_get_llm.return_value = mock_llm

        out = rerank("question", candidates, top_k=4)

    # Falls back to the first top_k candidates in their original RRF order,
    # and critically returns a full result set rather than an empty one.
    assert out == candidates[:4]


def test_rerank_applies_valid_ordering_and_backfills_omissions():
    """A well-formed response reorders; candidates the LLM omitted are
    appended in original order rather than silently dropped."""
    from app.retrieval import rerank

    candidates = _candidates(12)

    with patch("app.retrieval.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="[3, 1]")
        mock_get_llm.return_value = mock_llm

        out = rerank("question", candidates, top_k=4)

    assert out[0] is candidates[2]  # rank 3 -> index 2
    assert out[1] is candidates[0]  # rank 1 -> index 0
    assert len(out) == 4  # backfilled, not truncated to the 2 ranked


def test_retrieve_with_hybrid_and_rerank():
    """Verify hybrid_retrieve -> rerank pipeline works end-to-end
    with the new database.hybrid_search backend."""
    from app.retrieval import retrieve_with_hybrid_and_rerank

    with patch("app.retrieval.get_embeddings") as mock_get_emb:
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_emb

        with patch("app.retrieval.database") as mock_db:
            # hybrid_search returns empty list -> no candidates
            mock_db.hybrid_search.return_value = []

            res = retrieve_with_hybrid_and_rerank("query")
            assert res == []
            mock_db.hybrid_search.assert_called_once()


def test_hybrid_retrieve_converts_rows_to_documents():
    """Verify that database rows are converted to LangChain Documents."""
    from app.retrieval import hybrid_retrieve

    with patch("app.retrieval.get_embeddings") as mock_get_emb:
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_emb

        with patch("app.retrieval.database") as mock_db:
            mock_db.hybrid_search.return_value = [
                {
                    "id": 1,
                    "source": "test.pdf",
                    "content": "The refund policy is 30 days.",
                    "metadata": {"page": 1},
                }
            ]

            docs = hybrid_retrieve("refund policy", k=4)
            assert len(docs) == 1
            assert docs[0].page_content == "The refund policy is 30 days."
            assert docs[0].metadata["source"] == "test.pdf"
            assert docs[0].metadata["page"] == 1
