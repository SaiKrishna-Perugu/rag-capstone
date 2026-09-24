"""
TypeSafe System One judgments: typed yes/no answers instead of parsed text.

Several pipeline stages ask an LLM for a one-word verdict and then parse the
reply as a string -- ``.strip().upper() in ("SUFFICIENT", "INSUFFICIENT")``.
A reply with trailing punctuation or an extra word silently falls to the
default branch, and "how strict" can only be set by rewording the prompt.
A System One model instead returns the probability that the answer is yes,
so the cut-off is a number in config.py that can be tuned against data.

This is the only module that talks to TypeSafe, for the same reason
llm/providers.py is the only one that builds LLM clients: one place for the
client, its failure handling, and its accounting.

Every call here is optional by construction. ``noul()`` returns None when
TypeSafe is unconfigured, unreachable, slow, or answers something unusable,
and each caller treats None as "run the pre-TypeSafe path". So a TypeSafe
outage degrades a stage to how it behaved before this module existed, never
into an error -- and the breaker below keeps a sustained outage from adding
a timeout to every request while it lasts.
"""

import logging
from functools import lru_cache

from app import config, metrics
from app.llm import circuit, cost
from app.tracing import traced

logger = logging.getLogger(__name__)

_BREAKER = "typesafe"


@lru_cache(maxsize=1)
def _client():
    # Imported here so a deployment with every judgment off never imports
    # the SDK at all.
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    return TypeSafeClient(
        api_key=config.TYPESAFE_API_KEY,
        model=config.TYPESAFE_MODEL,
        timeout=config.TYPESAFE_TIMEOUT_SECONDS,
        retry=RetryPolicy(max_retries=config.TYPESAFE_MAX_RETRIES),
    )


def _fallback(stage: str, reason: str) -> None:
    metrics.record_typesafe_call(stage, "fallback")
    logger.warning(f"TypeSafe judgment '{stage}' unavailable ({reason}); using the fallback path.")


def noul(state: dict, instructions: str, criteria: dict | None, stage: str) -> float | None:
    """Probability (0-1) that the answer to `instructions` is yes, or None.

    `criteria` optionally defines what yes and no mean: {"true": ..., "false": ...}.
    `stage` labels the call in the per-request cost breakdown and in
    rag_typesafe_calls_total.
    """
    answers = nouls(state, {"q": (instructions, criteria)}, stage)
    return None if answers is None else answers["q"]


@traced("typesafe.nouls")
def nouls(
    state: dict, questions: dict[str, tuple[str, dict | None]], stage: str
) -> dict[str, float] | None:
    """Several yes/no questions over one state, in one request.

    `questions` maps an id of the caller's choosing to (instructions,
    criteria); the ids stay in code and are never sent as meaning, so each
    instruction must be complete on its own. Returns {id: probability} with
    every id present, or None -- all or nothing, so a caller never acts on a
    partial set of answers.
    """
    if not config.TYPESAFE_API_KEY:
        return _fallback(stage, "no API key")

    breaker = circuit.get_breaker(_BREAKER)
    if not breaker.allow_request():
        return _fallback(stage, "circuit open")

    from typesafe_sdk import Noul

    try:
        response = _client().system_one(
            state=state,
            questions={
                qid: Noul(instructions=instructions, criteria=criteria)
                for qid, (instructions, criteria) in questions.items()
            },
        )
        answers = {qid: float(response.nouls[qid].noul) for qid in questions}
        if not all(0.0 <= p <= 1.0 for p in answers.values()):
            raise ValueError(f"noul out of range: {answers}")
    except Exception as exc:
        if breaker.record_failure():
            metrics.record_circuit_opened(_BREAKER)
        return _fallback(stage, type(exc).__name__)

    breaker.record_success()
    # Priced by the configured alias rather than response.model, which names
    # whatever version the alias currently points at ("jev-1.13.0").
    cost.add_usage(
        config.TYPESAFE_MODEL,
        response.usage.input_tokens,
        response.usage.output_tokens,
        stage=stage,
    )
    metrics.record_typesafe_call(stage, "ok")
    return answers
