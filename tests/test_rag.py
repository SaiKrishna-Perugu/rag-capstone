from unittest.mock import patch, MagicMock

def test_check_groundedness_supported(mock_groundedness):
    # This tests that our check_groundedness function works and can parse LLM output
    from app.rag import check_groundedness
    mock_chunk = MagicMock()
    mock_chunk.page_content = "The sky is blue."
    
    with patch("app.providers.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "GROUNDED"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = check_groundedness("The sky is blue.", [mock_chunk])
        assert result == "GROUNDED"

def test_generate_answer(mock_llm_answer):
    from app.rag import generate_answer
    mock_chunk = MagicMock()
    mock_chunk.page_content = "Context"
    
    ans = generate_answer("Question?", [mock_chunk])
    assert ans == "The refund policy is 30 days."
