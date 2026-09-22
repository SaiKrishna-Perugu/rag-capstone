"""TypeSafe judgments: typed probabilities in, the old path whenever they fail.

The agentic grader parsed a one-word LLM reply as text, so any reply other
than exactly SUFFICIENT/INSUFFICIENT counted as insufficient and cost a
retry. With config.TYPESAFE_GRADER on, the grade comes from a probability
and a threshold instead. Every failure mode of the TypeSafe call -- no key,
an exception, an unusable answer, an open breaker -- must land back on the
original grader rather than surface as an error.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from typesafe_sdk import SystemOneResponse

from app import config
from app.llm import circuit, cost, typesafe
from app.retrieval import agent


def _response(p: float, input_tokens: int = 1000) -> SystemOneResponse:
    return SystemOneResponse.model_validate({
        "model": "jev-1.13.0",
        "usage": {"input_tokens": input_tokens, "output_tokens": 1},
        "answers": {"q": {"type": "noul", "noul": p}},
    })


@pytest.fixture
def client():
    """A configured TypeSafe client whose system_one() the test controls."""
    fake = MagicMock()
    circuit.reset_all()
    with patch.object(config, "TYPESAFE_API_KEY", "test-key"), \
         patch("app.llm.typesafe._client", return_value=fake):
        yield fake
    circuit.reset_all()


def _ask():
    return typesafe.noul({"question": "q"}, "Is it?", None, stage="grade")


# --- app/llm/typesafe.py ---------------------------------------------------

def test_no_api_key_falls_back_without_calling_the_service():
    with patch.object(config, "TYPESAFE_API_KEY", ""), \
         patch("app.llm.typesafe._client") as make_client, \
         patch("app.llm.typesafe.metrics.record_typesafe_call") as record:
        assert _ask() is None
    make_client.assert_not_called()
    record.assert_called_once_with("grade", "fallback")


def test_answer_is_priced_under_its_stage(client):
    client.system_one.return_value = _response(0.8, input_tokens=1_000_000)
    cost.start_request()
    with patch("app.llm.typesafe.metrics.record_typesafe_call") as record:
        assert _ask() == pytest.approx(0.8)
    usage = cost.current()
    assert usage.by_stage["grade"] == pytest.approx(0.042)
    record.assert_called_once_with("grade", "ok")


def test_any_exception_falls_back(client):
    client.system_one.side_effect = TimeoutError("slow")
    assert _ask() is None


def test_an_out_of_range_probability_is_not_trusted(client):
    client.system_one.return_value = _response(1.7)
    assert _ask() is None


def test_a_sustained_outage_stops_calling_the_service(client):
    """Without the breaker every request would wait out the timeout."""
    client.system_one.side_effect = ConnectionError("down")
    for _ in range(config.LLM_CIRCUIT_FAILURE_THRESHOLD):
        assert _ask() is None
    calls_before = client.system_one.call_count
    assert _ask() is None
    assert client.system_one.call_count == calls_before


def test_jev_has_a_price():
    """An unpriced model reports $0, so TypeSafe spend would never reach
    DAILY_BUDGET_USD."""
    assert cost.estimate_cost(config.TYPESAFE_MODEL, 1_000_000, 0) > 0


# --- app/retrieval/agent.py::node_grade -------------------------------------

_CHUNKS = [Document(page_content="Returns are accepted within 30 days.",
                    metadata={"source": "docs/returns.md", "_session_id": "visitor-tag"})]


def _state():
    return {"chunks": _CHUNKS, "original_question": "How long is the return window?",
            "session_id": "visitor-tag"}


def _llm_replying(text):
    llm = MagicMock()
    llm.invoke.return_value.content = text
    return llm


@pytest.mark.parametrize("p,expected", [(0.9, "SUFFICIENT"), (0.1, "INSUFFICIENT")])
def test_grade_comes_from_the_probability(p, expected):
    with patch.object(config, "TYPESAFE_GRADER", True), \
         patch("app.retrieval.agent.typesafe.noul", return_value=p), \
         patch("app.retrieval.agent._get_grading_llm") as get_llm:
        out = agent.node_grade(_state())
    assert out["grade"] == expected
    assert out["sufficiency_p"] == p
    get_llm.assert_not_called()


def test_threshold_is_config_not_prompt_wording():
    with patch.object(config, "TYPESAFE_GRADER", True), \
         patch.object(config, "TYPESAFE_GRADER_THRESHOLD", 0.3), \
         patch("app.retrieval.agent.typesafe.noul", return_value=0.4):
        assert agent.node_grade(_state())["grade"] == "SUFFICIENT"


def test_typesafe_unavailable_falls_back_to_the_llm_grader():
    with patch.object(config, "TYPESAFE_GRADER", True), \
         patch("app.retrieval.agent.typesafe.noul", return_value=None), \
         patch("app.retrieval.agent._get_grading_llm", return_value=_llm_replying("SUFFICIENT")):
        out = agent.node_grade(_state())
    assert out["grade"] == "SUFFICIENT"
    assert out["sufficiency_p"] is None


def test_switch_off_never_calls_typesafe():
    with patch.object(config, "TYPESAFE_GRADER", False), \
         patch("app.retrieval.agent.typesafe.noul") as noul, \
         patch("app.retrieval.agent._get_grading_llm", return_value=_llm_replying("INSUFFICIENT")):
        out = agent.node_grade(_state())
    noul.assert_not_called()
    assert out["grade"] == "INSUFFICIENT"


def test_only_question_and_passages_are_sent():
    """The visitor's session tag scopes their uploads; it has no bearing on
    sufficiency and must not leave the process in the judgment's state."""
    with patch.object(config, "TYPESAFE_GRADER", True), \
         patch("app.retrieval.agent.typesafe.noul", return_value=0.9) as noul:
        agent.node_grade(_state())
    sent = noul.call_args.kwargs["state"]
    assert set(sent) == {"question", "passages"}
    assert sent["passages"] == [{"source": "docs/returns.md",
                                 "text": "Returns are accepted within 30 days."}]
    assert "visitor-tag" not in str(sent)


def test_the_suite_runs_with_typesafe_off():
    """A developer .env with a TypeSafe switch on must not make the mocked
    suite send real requests; conftest.py switches them off."""
    assert config.TYPESAFE_GRADER is False
