"""
Per-request LLM cost attribution.

"What does a request cost?" was previously unanswerable. It matters here
for a concrete reason: `/ask` makes **three** LLM calls (rerank, generate,
groundedness check), so the interesting number is not the price of one
call but the sum -- and the breakdown tells you whether reranking earns
its keep.

Two pieces live here:

* A pricing table. These are USD per 1M tokens and are **estimates that go
  stale** -- provider pricing changes, and the authoritative number is
  always the Cloud Billing report. Every entry is overridable by env var
  so a correction never needs a code change. Cost figures produced here
  are for relative comparison and rough budgeting, not accounting.
* A request-scoped accumulator built on `contextvars`, so concurrent
  requests can't bleed usage into each other's totals. A plain module
  global would attribute one user's tokens to another under load, which
  is exactly the condition where the number starts mattering.
"""
import contextvars
import os
from dataclasses import dataclass, field

from app.llm import budget

# USD per 1,000,000 tokens. Override any of these via env rather than
# editing code -- e.g. RAG_PRICE_GEMINI_2_5_FLASH_LITE_IN=0.12
_DEFAULT_PRICING = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "openai/gpt-oss-20b": (0.075, 0.30),
    # Embedding models: input-only (output side is always 0, see
    # llm/providers.py::_CostTrackingEmbeddings). FastEmbed models are
    # absent here on purpose -- they run locally, so $0 is the correct
    # price, not an unpriced gap.
    "text-embedding-005": (0.10, 0.0),
}


def _env_key(model: str, direction: str) -> str:
    slug = model.upper().replace("-", "_").replace(".", "_")
    return f"RAG_PRICE_{slug}_{direction}"


def _price(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for `model`.

    Unknown models price at zero rather than raising or guessing: a new
    model should not crash a request, and a silently wrong invented price
    is worse than an obvious zero that shows up as "unpriced" in the data.
    """
    default_in, default_out = _DEFAULT_PRICING.get(model, (0.0, 0.0))
    return (
        float(os.getenv(_env_key(model, "IN"), str(default_in))),
        float(os.getenv(_env_key(model, "OUT"), str(default_out))),
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = _price(model)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class RequestUsage:
    """LLM usage accumulated over one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    # Per-call-site breakdown, e.g. {"rerank": 0.00012, "generate": 0.0004}.
    # This is the part worth reading: it answers whether a pipeline stage
    # justifies its cost.
    by_stage: dict = field(default_factory=dict)

    def as_log_fields(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            # 6dp: a single flash-lite call runs well under a cent, so
            # fewer decimals would render most requests as 0.0.
            "cost_usd": round(self.cost_usd, 6),
            "cost_by_stage": {k: round(v, 6) for k, v in self.by_stage.items()},
        }


_usage: contextvars.ContextVar[RequestUsage | None] = contextvars.ContextVar(
    "rag_request_usage", default=None
)


def start_request() -> None:
    """Begin accumulating for the current request context."""
    _usage.set(RequestUsage())


def add_usage(model: str, input_tokens: int, output_tokens: int, stage: str = "llm") -> float:
    """Record one LLM call. Returns its estimated cost.

    Safe to call outside a request (ingestion, eval harnesses, tests):
    with no active accumulator it just computes and returns the cost
    without storing anything, so provider code never needs to know
    whether it is running inside a request.
    """
    cost = estimate_cost(model, input_tokens, output_tokens)
    # Every priced call passes through here, which makes this the one place
    # a daily ceiling can be maintained without a second accounting path.
    # Unlike the contextvar above it is process-global on purpose: the
    # budget is about total spend, not any one request's share of it.
    budget.record_spend(cost)
    usage = _usage.get()
    if usage is not None:
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.cost_usd += cost
        usage.calls += 1
        usage.by_stage[stage] = usage.by_stage.get(stage, 0.0) + cost
    return cost


def current() -> RequestUsage:
    """Usage so far for this request; an empty record if none started."""
    return _usage.get() or RequestUsage()
