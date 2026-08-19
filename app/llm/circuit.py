"""
Circuit breaker for LLM provider calls.

The problem: without one, every request individually rediscovers a provider
outage the slow way. ``config.LLM_MAX_RETRIES`` is 3 and
``config.LLM_REQUEST_TIMEOUT`` is 60s, so a single call against a dead
provider can burn ~3 minutes before it gives up -- and ``/ask`` makes three
LLM calls (rerank, generate, groundedness). Under a real outage every
concurrent request pays that in full while holding a thread from
``asyncio.to_thread``, which turns "the provider is down" into "the service
is wedged". Tripping a breaker after a few consecutive failures converts
that into an immediate, cheap error (or a failover -- see app/providers.py).

Standard three-state machine:

    CLOSED  --N consecutive failures-->  OPEN
    OPEN    --cooldown elapsed-------->  HALF_OPEN
    HALF_OPEN --success--------------->  CLOSED
    HALF_OPEN --failure--------------->  OPEN (cooldown clock restarts)

**Consecutive** failures, not a failure rate over a window, and that choice
does real work here: it means a one-off error mixed into healthy traffic can
never trip the breaker, because any success resets the count to zero. That
matters because this layer cannot reliably tell a provider outage from a
request-shaped problem -- an oversized context or a malformed prompt raises
from the same client as a 503 does, and the two would need brittle
provider-specific exception taxonomies to separate (Groq and Vertex AI raise
entirely different types). Requiring N in a row instead lets normal traffic
do that filtering for us: only a genuinely broken provider produces an
uninterrupted run of failures.

Note the interaction with LangChain's own retry: this breaker sits *outside*
it, so each failure counted here already represents ``LLM_MAX_RETRIES``
upstream attempts. The default threshold of 3 is therefore ~9 real attempts
before the circuit opens, not 3.

**Scope: one process.** State lives in module globals, so each Cloud Run
instance learns about an outage independently and pays its own first N
failures. Sharing it (Firestore, Redis) would make the trip global but adds
a network round-trip to the hot path of every LLM call and a new dependency
on the critical path -- a bad trade for a bounded, small cost at this scale.
"""
import logging
import threading
import time
from enum import Enum

from app import config

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised in place of calling a provider whose circuit is open.

    Callers that already fail open on any exception -- ``rerank()`` falling
    back to RRF order, ``check_groundedness()`` returning ``NOT_CHECKED`` --
    degrade exactly as they would on a real provider error, just
    immediately instead of after the full retry/timeout sequence.
    """


class CircuitBreaker:
    """Tracks the health of one named dependency. Thread-safe: LLM calls
    reach here from the worker threads ``asyncio.to_thread`` spawns, so
    several can be recording outcomes at once."""

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        cooldown_seconds: float,
        now=time.monotonic,
    ):
        """`now` is a seam for tests, so cooldown behaviour can be asserted
        at a realistic value instead of only at a degenerate zero (which
        half-opens on the very next read and never observably stays open)."""
        self.name = name
        # A threshold below 1 would open the circuit on a single failure and
        # defeat the consecutive-failure filtering described above.
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = cooldown_seconds
        self._now = now
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> CircuitState:
        """Current state, promoting OPEN -> HALF_OPEN once the cooldown has
        elapsed. Evaluated lazily on read rather than by a timer thread:
        Cloud Run only allocates CPU during request handling, so a
        background timer is not reliably scheduled between requests.

        Uses ``time.monotonic()`` -- a wall-clock adjustment (NTP step,
        DST) must not make a cooldown expire early or hang forever.
        """
        if (
            self._state is CircuitState.OPEN
            and (self._now() - self._opened_at) >= self._cooldown_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info(
                f"Circuit '{self.name}' half-open after {self._cooldown_seconds}s "
                "cooldown; next call is a recovery probe."
            )
        return self._state

    def allow_request(self) -> bool:
        """False means "don't call this provider" -- the fail-fast path.

        In HALF_OPEN this returns True for every caller rather than gating
        to a single probe. Admitting a small burst on recovery is the
        accepted trade: single-probe gating needs a slot that must be
        released on every exit path, and a probe that never returns would
        wedge the breaker half-open permanently. A burst is bounded here by
        the 20/minute rate limit in config.RATE_LIMIT.
        """
        with self._lock:
            return self._state_locked() is not CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            recovered = self._state is not CircuitState.CLOSED
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
        if recovered:
            logger.info(f"Circuit '{self.name}' closed -- provider recovered.")

    def record_failure(self) -> bool:
        """Record a failed call. Returns True only when *this* failure is
        what tripped the circuit, so the caller can log and count one event
        per trip instead of one per failure for as long as it stays open.
        """
        with self._lock:
            state = self._state_locked()
            self._consecutive_failures += 1
            # A failed recovery probe re-opens immediately -- it should not
            # have to accumulate another full threshold to prove the
            # provider is still down.
            if not (
                state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self._failure_threshold
            ):
                return False
            newly_opened = state is not CircuitState.OPEN
            failures = self._consecutive_failures
            self._state = CircuitState.OPEN
            self._opened_at = self._now()

        if newly_opened:
            logger.warning(
                f"Circuit '{self.name}' OPEN after {failures} consecutive "
                f"failures; failing fast for {self._cooldown_seconds}s."
            )
        return newly_opened


_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """Get (or create) the breaker for a provider.

    Keyed by provider name, deliberately: ``providers.get_llm()`` is
    ``lru_cache``d per ``(temperature, stage)``, so holding breaker state on
    the returned client would split one provider's health across up to 16
    independent copies -- each counting its own failures, none of them ever
    reaching the threshold. Health is a property of the provider, not of the
    call site.
    """
    with _registry_lock:
        breaker = _breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name,
                config.LLM_CIRCUIT_FAILURE_THRESHOLD,
                config.LLM_CIRCUIT_COOLDOWN_SECONDS,
            )
            _breakers[name] = breaker
        return breaker


def reset_all() -> None:
    """Drop all breaker state. For tests -- and it must be paired with
    ``providers.get_llm.cache_clear()``, since a cached client holds the
    provider names this state is keyed by."""
    with _registry_lock:
        _breakers.clear()
