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
    """The breakdown is the point, not the total: the per-stage split is what
    showed reranking eating ~47% of spend. Three stages are recorded here
    because that is the RERANKER_PROVIDER=llm shape -- the flashrank default
    makes no LLM rerank call, and this asserts the accounting, not the
    pipeline's current call count."""
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


# ---------------------------------------------------------------------------
# Per-request embedding memo (app/llm/providers.py).
#
# A cache-miss /ask embedded the SAME question three times -- cache lookup,
# retrieval, cache write -- because each call site asks the provider directly.
# All three are tagged stage="embedding", so the per-stage cost breakdown
# aggregated 3x into one bucket and it never showed up as waste.
# ---------------------------------------------------------------------------

class _CountingEmbeddings:
    """Stand-in for a provider client. Counts real calls."""

    def __init__(self):
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


def _wrapped():
    from app.llm import providers
    raw = _CountingEmbeddings()
    return raw, providers._CostTrackingEmbeddings(raw, "text-embedding-005")


def test_same_question_is_embedded_once_per_request():
    from app.llm import providers
    raw, emb = _wrapped()
    providers.start_request()
    for _ in range(3):
        emb.embed_query("How long is the refund window?")
    assert raw.query_calls == 1, (
        f"one question, one embedding -- got {raw.query_calls} provider calls"
    )


def test_memo_does_not_leak_between_requests():
    """Two visitors asking the identical question must not share a vector
    across request boundaries -- the memo is scoped, not a global cache."""
    from app.llm import providers
    raw, emb = _wrapped()
    providers.start_request()
    emb.embed_query("same question")
    providers.start_request()
    emb.embed_query("same question")
    assert raw.query_calls == 2


def test_memo_survives_asyncio_to_thread():
    """The property the whole design rests on.

    The three real call sites run inside separate asyncio.to_thread hops, and
    each thread gets a COPY of the context -- so a ContextVar.set() inside one
    thread is invisible to the next. The memo works only because start_request
    binds a MUTABLE dict in the request coroutine that every copy shares by
    reference. If someone later 'simplifies' this to set the var lazily inside
    embed_query, this test fails and that is exactly the point.
    """
    from app.llm import providers

    async def run():
        raw, emb = _wrapped()
        providers.start_request()
        await asyncio.to_thread(emb.embed_query, "q")
        await asyncio.to_thread(emb.embed_query, "q")
        return raw.query_calls

    assert asyncio.run(run()) == 1


def test_documents_are_never_memoised():
    """embed_documents carries large, caller-controlled input that is not
    repeated within a request -- memoising it would only grow memory."""
    from app.llm import providers
    raw, emb = _wrapped()
    providers.start_request()
    emb.embed_documents(["a", "b"])
    emb.embed_documents(["a", "b"])
    assert raw.document_calls == 2


def test_without_start_request_every_call_reaches_the_provider():
    """Absent memo degrades to the previous behaviour rather than breaking:
    a caller that forgets start_request() loses an optimisation, not
    correctness."""
    from app.llm import providers
    providers._QUERY_EMBEDDING_MEMO.set(None)
    raw, emb = _wrapped()
    emb.embed_query("q")
    emb.embed_query("q")
    assert raw.query_calls == 2
