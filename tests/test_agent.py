from unittest.mock import patch, MagicMock

def test_run_agentic_rag_hit():
    from app.agent import run_agentic_rag
    
    with patch("app.agent.rag.retrieve_with_hybrid_and_rerank") as mock_retrieve:
        mock_retrieve.return_value = [MagicMock(page_content="Mock content")]
        with patch("app.agent.rag.generate_answer") as mock_gen:
            mock_gen.return_value = "Mock answer"
            with patch("app.agent.rag.check_groundedness") as mock_check:
                mock_check.return_value = "GROUNDED"
                with patch("app.providers.get_llm") as mock_get_llm:
                    mock_llm = MagicMock()
                    mock_resp = MagicMock()
                    mock_resp.content = "yes" # relevant
                    mock_llm.invoke.return_value = mock_resp
                    mock_get_llm.return_value = mock_llm
                    
                    state = run_agentic_rag("Test query")
                    assert state["answer"] == "Mock answer"
                    assert state["groundedness"] == "GROUNDED"
                    assert state["retry_count"] == 0
