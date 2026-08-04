from unittest.mock import patch, MagicMock

def test_retrieve_with_hybrid_and_rerank():
    from app.retrieval import retrieve_with_hybrid_and_rerank
    
    with patch("app.retrieval._get_vector_store") as mock_get_vs:
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_get_vs.return_value = mock_vs
        
        with patch("app.retrieval._build_bm25_retriever") as mock_get_bm25:
            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = []
            mock_get_bm25.return_value = mock_bm25
            
            # Since both return [], it should return []
            res = retrieve_with_hybrid_and_rerank("query")
            assert res == []
