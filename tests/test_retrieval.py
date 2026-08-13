from unittest.mock import MagicMock, patch


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
