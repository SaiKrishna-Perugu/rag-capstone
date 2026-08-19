from unittest.mock import MagicMock, patch


def _chunk(text: str = "The sky is blue."):
    chunk = MagicMock()
    chunk.page_content = text
    chunk.metadata = {"source": "sky.txt"}
    return chunk


# NOTE: these patch `app.retrieval.rag.get_llm`, not `app.llm.providers.get_llm` -- rag.py
# binds the name at import time, so patching the source module would leave the
# real function in place. They also deliberately skip the `mock_groundedness`
# fixture, which replaces `check_groundedness` itself: a test using it would
# assert against the mock rather than the function under test.

def test_check_groundedness_supported():
    # This tests that our check_groundedness function works and can parse LLM output
    from app.retrieval.rag import check_groundedness

    with patch("app.retrieval.rag.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "GROUNDED"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = check_groundedness("The sky is blue.", [_chunk()])
        assert result == "GROUNDED"

def test_check_groundedness_llm_failure_returns_not_checked():
    # A transient failure on the *verification* call must not propagate --
    # the caller already has a usable answer.
    from app.retrieval.rag import check_groundedness

    with patch("app.retrieval.rag.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("upstream LLM 503")
        mock_get_llm.return_value = mock_llm

        assert check_groundedness("The sky is blue.", [_chunk()]) == "NOT_CHECKED"

def test_answer_question_survives_groundedness_failure(mock_retrieval, mock_llm_answer):
    # The end-to-end guarantee: a dead groundedness check degrades the verdict,
    # it does not lose the answer.
    from app.retrieval.rag import answer_question

    with patch("app.retrieval.rag.get_llm", side_effect=RuntimeError("upstream LLM 503")):
        result = answer_question("What is the refund policy?")

    assert result.answer == "The refund policy is 30 days."
    assert result.groundedness == "NOT_CHECKED"

def test_generate_answer(mock_llm_answer):
    from app.retrieval.rag import generate_answer
    mock_chunk = MagicMock()
    mock_chunk.page_content = "Context"
    
    ans = generate_answer("Question?", [mock_chunk])
    assert ans == "The refund policy is 30 days."
