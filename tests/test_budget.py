"""Daily LLM spend ceiling (app/llm/budget.py) and its enforcement in main.py."""
import datetime
from unittest.mock import patch

import pytest

from app import config
from app.llm import budget


@pytest.fixture(autouse=True)
def _clean_budget():
    """The accumulator is process-global, so leakage between tests is real."""
    budget.reset()
    yield
    budget.reset()


def test_disabled_by_default():
    """DAILY_BUDGET_USD=0 must never refuse -- this ships inert."""
    with patch.object(config, "DAILY_BUDGET_USD", 0.0):
        budget.record_spend(100.0)
        assert budget.is_exceeded() is False


def test_negative_limit_also_disables():
    with patch.object(config, "DAILY_BUDGET_USD", -1.0):
        budget.record_spend(5.0)
        assert budget.is_exceeded() is False


def test_accumulates_and_trips():
    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        budget.record_spend(0.4)
        assert budget.is_exceeded() is False
        budget.record_spend(0.4)
        assert budget.is_exceeded() is False
        budget.record_spend(0.4)
        assert budget.is_exceeded() is True


def test_trips_exactly_at_the_limit():
    """>= not >: spending exactly the budget has consumed it."""
    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        budget.record_spend(1.0)
        assert budget.is_exceeded() is True


def test_resets_when_the_utc_day_rolls_over():
    """The window is the UTC calendar day, so a new date starts from zero."""
    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        budget.record_spend(2.0)
        assert budget.is_exceeded() is True

        tomorrow = datetime.datetime.now(datetime.UTC).date() + datetime.timedelta(days=1)
        with patch.object(budget.DailyBudget, "_today", staticmethod(lambda: tomorrow)):
            assert budget.spent_today() == 0.0
            assert budget.is_exceeded() is False


def test_cost_calls_feed_the_budget():
    """add_usage() is the single funnel; nothing else should need to know."""
    from app.llm import cost

    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        cost.add_usage("gemini-2.5-flash-lite", 1_000_000, 1_000_000, stage="generate")
        # 0.10 in + 0.40 out per 1M = 0.50
        assert budget.spent_today() == pytest.approx(0.50)


def test_ask_refuses_with_503_when_exceeded(client, mock_retrieval, mock_llm_answer,
                                            mock_groundedness, mock_cache):
    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        budget.record_spend(5.0)
        resp = client.post("/ask", json={"question": "What is the refund policy?"})
    assert resp.status_code == 503
    # A visitor should be told this is a cap, not a crash.
    assert "daily usage limit" in resp.json()["detail"]["error"]


def test_ask_refuses_before_spending_anything(client, mock_retrieval, mock_llm_answer,
                                              mock_groundedness, mock_cache):
    """A refused request must not run retrieval or generation."""
    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        budget.record_spend(5.0)
        client.post("/ask", json={"question": "What is the refund policy?"})
    mock_retrieval.assert_not_called()
    mock_llm_answer.assert_not_called()


def test_ask_still_works_under_the_ceiling(client, mock_retrieval, mock_llm_answer,
                                           mock_groundedness, mock_cache):
    with patch.object(config, "DAILY_BUDGET_USD", 100.0):
        budget.record_spend(0.01)
        resp = client.post("/ask", json={"question": "What is the refund policy?"})
    assert resp.status_code == 200


def test_stream_endpoint_is_also_gated(client):
    with patch.object(config, "DAILY_BUDGET_USD", 1.0):
        budget.record_spend(5.0)
        resp = client.post("/ask-stream", json={"question": "What is the refund policy?"})
    assert resp.status_code == 503
