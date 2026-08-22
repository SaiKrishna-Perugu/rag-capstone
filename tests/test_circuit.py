"""Circuit breaker + cross-provider failover (app/llm/circuit.py, app/llm/providers.py).

The integration tests here patch `app.llm.providers._build_raw_client` rather
than the provider SDKs, keeping to the module-function boundary this repo
mocks at everywhere else. That boundary is also the useful one: it exercises
the real _ResilientLLM/_CostTrackingLLM stack, so a regression in the
breaker, the failover routing, or the cost attribution all show up.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from app import config
from app.llm import circuit, cost, providers
from app.llm.circuit import CircuitBreaker, CircuitOpenError, CircuitState

PRIMARY = "vertexai"
PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK = "groq"
FALLBACK_MODEL = "llama-3.3-70b-versatile"


@pytest.fixture(autouse=True)
def _isolate_provider_state():
    """Breaker state and the get_llm cache are both process-global; a test
    leaking either would silently change the next one's starting point."""
    circuit.reset_all()
    providers.get_llm.cache_clear()
    yield
    circuit.reset_all()
    providers.get_llm.cache_clear()


@pytest.fixture
def clock():
    """A controllable monotonic clock. Lets the cooldown be asserted at a
    realistic value -- at zero the circuit half-opens on the next read and
    is never observably open, which tests the degenerate case only."""
    now = {"t": 1_000.0}

    def read():
        return now["t"]

    read.advance = lambda seconds: now.update(t=now["t"] + seconds)
    return read


def _reply(content: str, in_tok: int = 10, out_tok: int = 5):
    """A response shaped like what _CostTrackingLLM reads off a real one."""
    return SimpleNamespace(
        content=content,
        usage_metadata={"input_tokens": in_tok, "output_tokens": out_tok},
    )


# --- CircuitBreaker state machine ----------------------------------------

def test_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker("p", failure_threshold=3, cooldown_seconds=300)

    assert breaker.record_failure() is False
    assert breaker.record_failure() is False
    assert breaker.allow_request() is True

    # The third one trips it, and says so exactly once.
    assert breaker.record_failure() is True
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow_request() is False
    assert breaker.record_failure() is False  # already open, not a new trip


def test_a_success_resets_the_failure_run():
    """The reason this counts consecutive failures rather than a rate: an
    isolated error inside healthy traffic must not be able to trip it,
    because this layer can't tell a bad request from a dead provider."""
    breaker = CircuitBreaker("p", failure_threshold=3, cooldown_seconds=300)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_stays_open_until_the_cooldown_elapses():
    breaker = CircuitBreaker("p", failure_threshold=1, cooldown_seconds=300)
    breaker.record_failure()
    assert breaker.allow_request() is False
    assert breaker.state is CircuitState.OPEN


def test_half_opens_after_cooldown_and_closes_on_a_successful_probe(clock):
    breaker = CircuitBreaker("p", failure_threshold=1, cooldown_seconds=30, now=clock)
    breaker.record_failure()

    clock.advance(29)
    assert breaker.allow_request() is False         # still inside the cooldown

    clock.advance(1)
    assert breaker.allow_request() is True          # cooldown elapsed -> probe
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_a_failed_probe_reopens_without_a_fresh_threshold(clock):
    breaker = CircuitBreaker("p", failure_threshold=5, cooldown_seconds=30, now=clock)
    for _ in range(5):
        breaker.record_failure()

    clock.advance(30)
    assert breaker.allow_request() is True          # half-open

    # One failure is enough -- the provider just proved it is still down.
    assert breaker.record_failure() is True
    assert breaker.state is CircuitState.OPEN
    # ...and the cooldown restarts from that failure, rather than staying
    # anchored to when the circuit first opened.
    clock.advance(29)
    assert breaker.allow_request() is False


# --- Failover configuration ----------------------------------------------

@pytest.mark.parametrize(
    ("fallback", "groq_key", "reason"),
    [
        ("", "k", "unset"),
        ("openai", "k", "not a supported provider"),
        (PRIMARY, "k", "same as MODEL_PROVIDER"),
        ("groq", "", "credentials missing"),
    ],
)
def test_failover_disables_itself_rather_than_raising(monkeypatch, fallback, groq_key, reason):
    """A bad value for an optional resilience feature must not be able to
    take the service down at startup."""
    monkeypatch.setattr(config, "MODEL_PROVIDER", PRIMARY)
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", fallback)
    monkeypatch.setattr(config, "GROQ_API_KEY", groq_key)

    assert providers._resolve_fallback_provider() is None, reason


def test_valid_fallback_resolves(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PROVIDER", PRIMARY)
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", FALLBACK)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")

    assert providers._resolve_fallback_provider() == FALLBACK


# --- Integration: get_llm() under a failing provider ----------------------

def _wire(monkeypatch, primary_client, fallback_client, *, fallback=FALLBACK, threshold=2):
    monkeypatch.setattr(config, "MODEL_PROVIDER", PRIMARY)
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", fallback)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_CIRCUIT_FAILURE_THRESHOLD", threshold)
    monkeypatch.setattr(config, "LLM_CIRCUIT_COOLDOWN_SECONDS", 300)

    def build(provider, temperature):
        if provider == PRIMARY:
            return primary_client, PRIMARY_MODEL
        if provider == FALLBACK:
            return fallback_client, FALLBACK_MODEL
        raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")

    return patch("app.llm.providers._build_raw_client", side_effect=build)


def test_repeated_failures_open_the_circuit_and_route_to_the_fallback(monkeypatch):
    """Phase 6's definition of done: force the underlying call to fail
    repeatedly, confirm the circuit opens and later calls go elsewhere."""
    primary = MagicMock()
    primary.invoke.side_effect = RuntimeError("provider 503")
    fallback = MagicMock()
    fallback.invoke.return_value = _reply("served by fallback")

    with _wire(monkeypatch, primary, fallback, threshold=2):
        llm = providers.get_llm(temperature=0.0, stage="generate")

        # Below the threshold the primary is still tried every time, and
        # each failure is covered by the fallback rather than raising.
        assert llm.invoke("q1").content == "served by fallback"
        assert llm.invoke("q2").content == "served by fallback"
        assert primary.invoke.call_count == 2

        # Threshold reached: the primary is now skipped entirely. This is
        # the whole point -- no retries, no timeout, no held thread.
        assert llm.invoke("q3").content == "served by fallback"
        assert primary.invoke.call_count == 2
        assert circuit.get_breaker(PRIMARY).state is CircuitState.OPEN
        assert fallback.invoke.call_count == 3


def test_open_circuit_fails_fast_when_no_fallback_is_configured(monkeypatch):
    """The breaker is useful on its own: callers that already fail open
    (rerank, groundedness) degrade immediately instead of after the full
    retry/timeout sequence."""
    primary = MagicMock()
    primary.invoke.side_effect = RuntimeError("provider 503")

    with _wire(monkeypatch, primary, MagicMock(), fallback="", threshold=2):
        llm = providers.get_llm(temperature=0.0, stage="generate")

        for _ in range(2):
            with pytest.raises(RuntimeError, match="provider 503"):
                llm.invoke("q")
        assert primary.invoke.call_count == 2

        with pytest.raises(CircuitOpenError):
            llm.invoke("q")
        assert primary.invoke.call_count == 2  # never called again


def test_recovery_closes_the_circuit_without_a_redeploy(monkeypatch):
    primary = MagicMock()
    primary.invoke.side_effect = RuntimeError("provider 503")
    fallback = MagicMock()
    fallback.invoke.return_value = _reply("served by fallback")

    with _wire(monkeypatch, primary, fallback, threshold=1):
        # Zero cooldown so the very next call is the recovery probe. (The
        # cooldown itself is asserted at a realistic value in the unit tests
        # above, where the clock is controllable.)
        monkeypatch.setattr(config, "LLM_CIRCUIT_COOLDOWN_SECONDS", 0)
        llm = providers.get_llm(temperature=0.0, stage="generate")

        # Trips the circuit; the fallback covers this call.
        assert llm.invoke("q").content == "served by fallback"

        # Provider comes back. The probe succeeds, so the circuit closes on
        # its own -- no redeploy, no config change.
        primary.invoke.side_effect = None
        primary.invoke.return_value = _reply("served by primary")

        assert llm.invoke("q").content == "served by primary"
        assert circuit.get_breaker(PRIMARY).state is CircuitState.CLOSED


def test_cost_is_billed_against_the_model_that_actually_served(monkeypatch):
    """Cost tracking sits inside failover, so a fallback-served call is
    priced with the fallback's model -- not the primary's."""
    primary = MagicMock()
    primary.invoke.side_effect = RuntimeError("provider 503")
    fallback = MagicMock()
    fallback.invoke.return_value = _reply("served by fallback", in_tok=1000, out_tok=200)

    with _wire(monkeypatch, primary, fallback, threshold=1):
        llm = providers.get_llm(temperature=0.0, stage="generate")
        cost.start_request()
        llm.invoke("q")

    usage = cost.current()
    assert usage.calls == 1
    assert usage.input_tokens == 1000
    assert usage.cost_usd == cost.estimate_cost(FALLBACK_MODEL, 1000, 200)
    assert usage.cost_usd != cost.estimate_cost(PRIMARY_MODEL, 1000, 200)


# --- Integration: streaming ----------------------------------------------

def _stream(*, tokens=(), fail_with=None):
    """Build an astream() stand-in yielding `tokens`, then optionally raising."""

    async def astream(*_args, **_kwargs):
        for token in tokens:
            yield AIMessageChunk(content=token)
        if fail_with is not None:
            raise fail_with

    return astream


def _collect(llm, *args):
    async def run():
        return [chunk.content async for chunk in llm.astream(*args)]

    return asyncio.run(run())


def test_stream_fails_over_when_it_breaks_before_the_first_token(monkeypatch):
    primary = MagicMock()
    primary.astream = _stream(fail_with=RuntimeError("provider 503"))
    fallback = MagicMock()
    fallback.astream = _stream(tokens=["fall", "back"])

    with _wire(monkeypatch, primary, fallback, threshold=5):
        llm = providers.get_llm(temperature=0.0, stage="generate_stream")
        assert _collect(llm, "q") == ["fall", "back"]


def test_stream_does_not_fail_over_once_tokens_have_been_sent(monkeypatch):
    """Re-routing mid-stream would restart the answer from the top and the
    reader would watch the response duplicate itself. streaming.py turns the
    raised error into an SSE error payload instead."""
    primary = MagicMock()
    primary.astream = _stream(tokens=["par", "tial"], fail_with=RuntimeError("provider 503"))
    fallback = MagicMock()
    fallback.astream = _stream(tokens=["fall", "back"])

    with _wire(monkeypatch, primary, fallback, threshold=5):
        llm = providers.get_llm(temperature=0.0, stage="generate_stream")
        with pytest.raises(RuntimeError, match="provider 503"):
            _collect(llm, "q")

        # The failure still counts toward the breaker even though it was
        # not recoverable here.
        assert circuit.get_breaker(PRIMARY).state is CircuitState.CLOSED


def test_stream_uses_the_fallback_while_the_circuit_is_open(monkeypatch):
    primary = MagicMock()
    primary.astream = _stream(fail_with=RuntimeError("provider 503"))
    fallback = MagicMock()
    fallback.astream = _stream(tokens=["fall", "back"])

    with _wire(monkeypatch, primary, fallback, threshold=1):
        llm = providers.get_llm(temperature=0.0, stage="generate_stream")

        assert _collect(llm, "q") == ["fall", "back"]      # trips the circuit
        assert circuit.get_breaker(PRIMARY).state is CircuitState.OPEN

        # Primary is not consulted at all now; swap in a stream that would
        # explode if it were.
        def _must_not_run(*_args, **_kwargs):
            raise AssertionError("primary was called while its circuit was open")

        primary.astream = _must_not_run
        assert _collect(llm, "q") == ["fall", "back"]
