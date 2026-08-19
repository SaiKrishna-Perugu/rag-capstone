"""Every /ask* endpoint must report what it cost.

Phase 8 originally wired cost.start_request() into /ask only, so
/ask-agentic and /ask-stream logged no cost fields at all -- their tokens
were spent and metered but never attributed to a request. These tests read
the actual structured log line each endpoint writes, because that log line
IS the deliverable ("logs show an estimated dollar cost per request"), not
an internal detail.

Mocking follows the repo convention: patch at the module-function boundary
(app.rag.*, app.main.run_agentic_rag) rather than mocking LLM clients.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from app import cost

COST_FIELDS = {"llm_calls", "input_tokens", "output_tokens", "cost_usd", "cost_by_stage"}


@pytest.fixture
def captured_log():
    """Capture what the service writes to its structured request log."""
    lines = []
    with patch("app.main.logger") as main_log, patch("app.streaming.logger") as stream_log:
        main_log.info.side_effect = lambda m: lines.append(m)
        stream_log.info.side_effect = lambda m: lines.append(m)
        yield lines


def _events(lines, event):
    out = []
    for line in lines:
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if payload.get("event") == event:
            out.append(payload)
    return out


def _spend(model="gemini-2.5-flash-lite", stage="generate", in_tok=1000, out_tok=100):
    """Simulate an LLM call billing into the active request accumulator,
    the way _CostTrackingLLM does from inside providers.py."""
    cost.add_usage(model, in_tok, out_tok, stage=stage)


def test_ask_reports_cost(client, mock_cache, mock_retrieval, mock_llm_answer, mock_groundedness, captured_log):
    mock_llm_answer.side_effect = lambda *a, **k: (_spend(stage="generate"), "answer")[1]

    assert client.post("/ask", json={"question": "q"}).status_code == 200

    (entry,) = _events(captured_log, "ask")
    assert COST_FIELDS <= entry.keys()
    assert entry["cost_usd"] > 0
    assert entry["cost_by_stage"]["generate"] > 0


def test_ask_agentic_reports_cost(client, mock_cache, captured_log):
    """The regression this closes: /ask-agentic never called start_request(),
    so it logged no cost fields at all."""
    def fake_agentic(question, session_id=None):
        _spend(stage="grade", in_tok=500, out_tok=10)
        _spend(stage="generate", in_tok=1200, out_tok=150)
        return {
            "current_query": question, "answer": "a", "groundedness": "GROUNDED",
            "sources": [], "retry_count": 1, "chunks": [],
        }

    with patch("app.main.run_agentic_rag", side_effect=fake_agentic):
        assert client.post("/ask-agentic", json={"question": "q"}).status_code == 200

    (entry,) = _events(captured_log, "ask-agentic")
    assert COST_FIELDS <= entry.keys()
    assert entry["llm_calls"] == 2
    # The per-stage split is the point: the agentic loop's extra grade/rewrite
    # calls are exactly the cost that a single total would hide.
    assert set(entry["cost_by_stage"]) == {"grade", "generate"}


def test_ask_stream_reports_cost(client, mock_cache, captured_log):
    """Guards the async-generator context question: cost.py accumulates in a
    ContextVar, and stream_answer() is an async generator resumed by the
    response task. If start_request() landed in a different context than the
    later add_usage()/current() calls, this comes back zero."""
    chunk = MagicMock()
    chunk.page_content = "ctx"
    chunk.metadata = {"source": "s.txt"}

    async def fake_astream(*_a, **_k):
        _spend(stage="generate_stream", in_tok=900, out_tok=40)
        for token in ("he", "llo"):
            out = MagicMock()
            out.content = token
            yield out

    llm = MagicMock()
    llm.astream = fake_astream

    with patch("app.streaming.retrieve", return_value=[chunk]), \
         patch("app.streaming.get_llm", return_value=llm), \
         patch("app.streaming.check_groundedness", return_value="GROUNDED"):
        resp = client.post("/ask-stream", json={"question": "q"})
        assert resp.status_code == 200
        assert "hello" in resp.text.replace('"', "")

    (entry,) = _events(captured_log, "ask-stream")
    assert COST_FIELDS <= entry.keys()
    assert entry["cost_usd"] > 0, "streamed usage did not reach the request accumulator"
    assert entry["cost_by_stage"]["generate_stream"] > 0


def test_cache_hit_still_reports_contextualization_cost(client, mock_retrieval, captured_log):
    """A cache hit is not free when a session_id is supplied:
    contextualize_question() runs an LLM call BEFORE the cache is consulted.
    Logging no cost there reported those tokens as free."""
    def fake_contextualize(session_id, question):
        _spend(stage="contextualize", in_tok=300, out_tok=20)
        return question

    hit = {"answer": "cached", "groundedness": "GROUNDED", "similarity_score": 0.99}

    with patch("app.memory.contextualize_question", side_effect=fake_contextualize), \
         patch("app.cache.get_cached_answer", return_value=hit), \
         patch("app.memory.add_to_history"):
        resp = client.post("/ask", json={"question": "q", "session_id": "s1"})
        assert resp.status_code == 200

    (entry,) = _events(captured_log, "ask")
    assert entry["cache"] == "HIT"
    assert entry["cost_by_stage"]["contextualize"] > 0
