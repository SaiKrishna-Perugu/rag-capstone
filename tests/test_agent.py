from unittest.mock import MagicMock, patch


def test_run_agentic_rag_hit():
    from app.retrieval.agent import run_agentic_rag
    
    with patch("app.retrieval.agent.retrieve") as mock_retrieve:
        mock_retrieve.return_value = [MagicMock(page_content="Mock content")]
        with patch("app.retrieval.agent.generate_answer") as mock_gen:
            mock_gen.return_value = "Mock answer"
            with patch("app.retrieval.agent.check_groundedness") as mock_check:
                mock_check.return_value = "GROUNDED"
                with patch("app.retrieval.agent.get_llm") as mock_get_llm:
                    mock_llm = MagicMock()
                    mock_resp = MagicMock()
                    mock_resp.content = "SUFFICIENT" # relevant
                    mock_llm.invoke.return_value = mock_resp
                    mock_get_llm.return_value = mock_llm
                    
                    state = run_agentic_rag("Test query")
                    assert state["answer"] == "Mock answer"
                    assert state["groundedness"] == "GROUNDED"
                    assert state["retry_count"] == 0


# --- routing and node behaviour ---------------------------------------------
# The self-correcting loop had exactly one test (the happy path) despite
# being the endpoint /ask-agentic exists to demonstrate. The retry, rewrite
# and fallback branches -- the whole reason the graph is a graph -- were
# never exercised.

import pytest

from app.retrieval import agent


@pytest.mark.parametrize(
    "grade,retries,expected",
    [
        ("SUFFICIENT", 0, "generate"),
        ("SUFFICIENT", agent.MAX_RETRIES, "generate"),   # good context wins over budget
        ("INSUFFICIENT", 0, "rewrite"),
        ("INSUFFICIENT", agent.MAX_RETRIES - 1, "rewrite"),
        ("INSUFFICIENT", agent.MAX_RETRIES, "fallback"),  # budget exhausted
        ("INSUFFICIENT", agent.MAX_RETRIES + 1, "fallback"),
    ],
)
def test_route_after_grade(grade, retries, expected):
    state = {"grade": grade, "retry_count": retries}
    assert agent.route_after_grade(state) == expected


def test_grade_short_circuits_on_empty_chunks():
    """No retrieved context cannot be SUFFICIENT, and grading it would spend
    an LLM call to learn that."""
    with patch("app.retrieval.agent._get_grading_llm") as llm:
        out = agent.node_grade({"chunks": [], "original_question": "q"})
    assert out["grade"] == "INSUFFICIENT"
    llm.assert_not_called()


def test_grade_treats_an_unparseable_verdict_as_insufficient():
    """Fail toward retrying rather than answering on context the judge did
    not actually approve."""
    doc = MagicMock()
    doc.page_content = "ctx"
    doc.metadata = {}
    resp = MagicMock()
    resp.content = "probably fine?"
    with patch("app.retrieval.agent._get_grading_llm") as llm:
        llm.return_value.invoke.return_value = resp
        out = agent.node_grade({"chunks": [doc], "original_question": "q"})
    assert out["grade"] == "INSUFFICIENT"


def test_rewrite_increments_the_retry_budget():
    """If this ever stopped incrementing, the loop would never reach
    fallback and would rewrite forever."""
    resp = MagicMock()
    resp.content = "  a sharper query  "
    with patch("app.retrieval.agent._get_grading_llm") as llm:
        llm.return_value.invoke.return_value = resp
        out = agent.node_rewrite_query(
            {"original_question": "q", "current_query": "q", "retry_count": 1}
        )
    assert out["retry_count"] == 2
    assert out["current_query"] == "a sharper query"
    assert out["original_question"] == "q", "the original question must survive rewriting"


def test_fallback_makes_no_claims():
    """The honest-refusal path: no sources, and GROUNDED only because it
    asserts nothing about the documents."""
    out = agent.node_fallback({"chunks": [], "retry_count": agent.MAX_RETRIES})
    assert out["answer"] == agent.FALLBACK_MESSAGE
    assert out["sources"] == []
    assert out["groundedness"] == "GROUNDED"


def test_node_generate_deduplicates_sources():
    """Verify that multiple chunks from the same document and page are deduplicated."""
    chunk1 = MagicMock(page_content="excerpt 1", metadata={"source": "doc.pdf", "page": 1})
    chunk2 = MagicMock(page_content="excerpt 2", metadata={"source": "doc.pdf", "page": 1})
    chunk3 = MagicMock(page_content="excerpt 3", metadata={"source": "other.pdf", "page": 2})

    state = {
        "original_question": "q",
        "chunks": [chunk1, chunk2, chunk3],
    }

    with patch("app.retrieval.agent.generate_answer", return_value="answer"):
        with patch("app.retrieval.agent.check_groundedness", return_value="GROUNDED"):
            out = agent.node_generate(state)

    assert len(out["sources"]) == 2
    assert out["sources"][0]["source"] == "doc.pdf"
    assert out["sources"][0]["page"] == 1
    assert out["sources"][1]["source"] == "other.pdf"
    assert out["sources"][1]["page"] == 2
