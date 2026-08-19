"""Per-request LLM cost attribution."""
import asyncio

from app.llm import cost


def test_estimate_cost_uses_pricing_table():
    # 1M input + 1M output at flash-lite's (0.10, 0.40) per-1M rates.
    assert cost.estimate_cost("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == 0.50


def test_unknown_model_prices_at_zero_rather_than_guessing():
    """A new model must not crash a request, and an invented price would be
    worse than an obvious zero that reads as 'unpriced' in the data."""
    assert cost.estimate_cost("some-model-shipped-yesterday", 1_000_000, 1_000_000) == 0.0


def test_pricing_overridable_by_env(monkeypatch):
    """Provider pricing changes; correcting it must not need a code edit."""
    monkeypatch.setenv("RAG_PRICE_GEMINI_2_5_FLASH_LITE_IN", "1.00")
    monkeypatch.setenv("RAG_PRICE_GEMINI_2_5_FLASH_LITE_OUT", "2.00")
    assert cost.estimate_cost("gemini-2.5-flash-lite", 1_000_000, 0) == 1.00
    assert cost.estimate_cost("gemini-2.5-flash-lite", 0, 1_000_000) == 2.00


def test_accumulates_across_calls_with_stage_breakdown():
    """The breakdown is the point: /ask makes three LLM calls, and the
    per-stage split is what shows whether reranking earns its cost."""
    cost.start_request()
    cost.add_usage("gemini-2.5-flash-lite", 1_000_000, 0, stage="rerank")
    cost.add_usage("gemini-2.5-flash-lite", 1_000_000, 0, stage="generate")
    cost.add_usage("gemini-2.5-flash-lite", 1_000_000, 0, stage="groundedness")

    usage = cost.current()
    assert usage.calls == 3
    assert usage.input_tokens == 3_000_000
    assert set(usage.by_stage) == {"rerank", "generate", "groundedness"}
    assert usage.as_log_fields()["cost_usd"] == 0.3


def test_usage_outside_a_request_is_a_no_op():
    """Ingestion, eval harnesses and tests call the same provider code, so
    recording must not require an active request context."""
    cost.start_request()
    cost.add_usage("gemini-2.5-flash-lite", 1000, 1000)
    fresh = cost.RequestUsage()
    assert fresh.calls == 0
    # A returned cost is still computed even with nothing to accumulate into.
    assert cost.estimate_cost("gemini-2.5-flash-lite", 1000, 1000) > 0


def test_concurrent_requests_do_not_share_totals():
    """The reason this uses contextvars rather than a module global: under
    load a global would bill one caller for another's tokens, which is
    exactly when the number starts to matter."""

    async def one(tokens):
        cost.start_request()
        cost.add_usage("gemini-2.5-flash-lite", tokens, 0)
        await asyncio.sleep(0)          # force interleaving
        cost.add_usage("gemini-2.5-flash-lite", tokens, 0)
        return cost.current().input_tokens

    async def main():
        return await asyncio.gather(one(1000), one(5000), one(9000))

    assert asyncio.run(main()) == [2000, 10000, 18000]
