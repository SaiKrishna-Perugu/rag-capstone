import json
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
def test_rerank_falls_back_to_rrf_order_on_bad_llm_response(llm_behavior, monkeypatch):
    """rerank() is a quality optimization, never a correctness requirement:
    every failure mode must degrade to the pre-rerank RRF order rather than
    propagate. Regression test -- llm.invoke() used to sit outside the try,
    so a provider timeout took down the whole retrieval call, and a non-list
    JSON response raised TypeError past the except-tuple."""
    from app import config
    from app.retrieval.hybrid import rerank

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "llm")
    candidates = _candidates(12)

    with patch("app.retrieval.hybrid.get_llm") as mock_get_llm:
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


def test_rerank_applies_valid_ordering_and_backfills_omissions(monkeypatch):
    """A well-formed response reorders; candidates the LLM omitted are
    appended in original order rather than silently dropped."""
    from app import config
    from app.retrieval.hybrid import rerank

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "llm")
    candidates = _candidates(12)

    with patch("app.retrieval.hybrid.get_llm") as mock_get_llm:
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
    from app.retrieval.hybrid import retrieve_with_hybrid_and_rerank

    with patch("app.retrieval.hybrid.get_embeddings") as mock_get_emb:
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_emb

        with patch("app.retrieval.hybrid.database") as mock_db:
            # hybrid_search returns empty list -> no candidates
            mock_db.hybrid_search.return_value = []

            res = retrieve_with_hybrid_and_rerank("query")
            assert res == []
            mock_db.hybrid_search.assert_called_once()


def test_hybrid_retrieve_converts_rows_to_documents():
    """Verify that database rows are converted to LangChain Documents."""
    from app.retrieval.hybrid import hybrid_retrieve

    with patch("app.retrieval.hybrid.get_embeddings") as mock_get_emb:
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_emb

        with patch("app.retrieval.hybrid.database") as mock_db:
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


# --- Reranker response parsing ------------------------------------------
# The eval gate logged three "rerank failed (JSONDecodeError...)" lines per
# run, every run, silently discarding the most expensive stage in the
# pipeline. Both shapes below fail json.loads() at character 0 with the
# identical, unhelpful "Expecting value: line 1 column 1 (char 0)".

@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param("[3, 1, 2]", [3, 1, 2], id="bare_array"),
        pytest.param("```json\n[3, 1, 2]\n```", [3, 1, 2], id="fenced_with_lang"),
        pytest.param("```\n[3, 1, 2]\n```", [3, 1, 2], id="fenced_bare"),
        pytest.param("  [3, 1, 2]  ", [3, 1, 2], id="surrounding_whitespace"),
        pytest.param("Ranked: [3, 1, 2] -- most relevant first.", [3, 1, 2], id="wrapped_in_prose"),
    ],
)
def test_parse_rank_order_recovers_real_llm_shapes(raw, expected):
    from app.retrieval.hybrid import _parse_rank_order
    assert _parse_rank_order(raw) == expected


@pytest.mark.parametrize(
    "raw,expected_exc",
    [
        pytest.param("", ValueError, id="empty"),
        pytest.param("   \n  ", ValueError, id="whitespace_only"),
        pytest.param("no array here at all", json.JSONDecodeError, id="no_array"),
        pytest.param('{"ranking": [1, 2]}', TypeError, id="object_not_array"),
        pytest.param("42", TypeError, id="bare_number"),
    ],
)
def test_parse_rank_order_raises_so_rerank_falls_back(raw, expected_exc):
    """Anything unusable must raise -- rerank() turns that into RRF order."""
    from app.retrieval.hybrid import _parse_rank_order
    with pytest.raises(expected_exc):
        _parse_rank_order(raw)


def test_rerank_now_honours_a_fenced_response(monkeypatch):
    """End-to-end: the shape that used to be discarded now reorders."""
    from app import config
    from app.retrieval import hybrid

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "llm")
    candidates = _candidates(6)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="```json\n[6, 5, 4, 3, 2, 1]\n```")

    with patch("app.retrieval.hybrid.get_llm", return_value=mock_llm):
        result = hybrid.rerank("q", candidates, top_k=3)

    # Reversed order means the LLM ranking was actually applied, not the
    # RRF passthrough it previously fell back to.
    assert [d.page_content for d in result] == ["chunk 5", "chunk 4", "chunk 3"]


def test_rerank_flashrank_success(monkeypatch):
    """Verify FlashRank reorders candidates according to model score."""
    from app import config
    from app.retrieval import hybrid

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "flashrank")
    candidates = [
        Document(page_content="Hardware warranty details"),
        Document(page_content="Returns and refund policy is 30 days"),
        Document(page_content="Standard shipping takes 5 days"),
    ]

    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": 1, "score": 0.99},
        {"id": 0, "score": 0.10},
        {"id": 2, "score": 0.05},
    ]

    with patch("app.retrieval.hybrid._get_flashrank", return_value=mock_ranker):
        result = hybrid.rerank("refund policy", candidates, top_k=2)

    assert len(result) == 2
    assert result[0].page_content == "Returns and refund policy is 30 days"
    assert result[1].page_content == "Hardware warranty details"


def test_rerank_flashrank_fails_open_to_rrf(monkeypatch):
    """On FlashRank exception, rerank falls back to RRF candidate order."""
    from app import config
    from app.retrieval import hybrid

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "flashrank")
    candidates = _candidates(6)

    mock_ranker = MagicMock()
    mock_ranker.rerank.side_effect = RuntimeError("FlashRank model corrupted")

    with patch("app.retrieval.hybrid._get_flashrank", return_value=mock_ranker):
        result = hybrid.rerank("q", candidates, top_k=3)

    assert result == candidates[:3]


def test_rerank_provider_none(monkeypatch):
    """When RERANKER_PROVIDER is none, rerank returns top_k in RRF order directly."""
    from app import config
    from app.retrieval import hybrid

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "none")
    candidates = _candidates(6)
    result = hybrid.rerank("q", candidates, top_k=3)
    assert result == candidates[:3]
