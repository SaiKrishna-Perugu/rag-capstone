"""
Daily LLM spend ceiling.

The circuit breaker in app/llm/circuit.py stops calls to a provider that is
*broken*. This stops calls that are merely *expensive*. They are different
failures: a provider can be perfectly healthy while a scripted loop against
a publicly advertised demo URL runs up a bill, and nothing in the retry,
failover or rate-limit path notices that at all.

Enforcement sits at the request boundary (app/main.py), not inside
get_llm(). Refusing a whole request with one honest message is kinder than
letting it start and then failing partway through generation, and it means
a refused request costs zero tokens rather than the two calls it had
already made.

Two limitations, stated because they bound what this can promise:

* The number is app/llm/cost.py's ESTIMATE, derived from a hand-maintained
  price table that goes stale. Cloud Billing remains authoritative, and the
  billing budget alert remains the real backstop.
* State is PER PROCESS, the same trade app/llm/circuit.py makes: each Cloud
  Run instance tracks its own total, so the true worst case is the limit
  multiplied by maxScale. Making it exact would mean a network round-trip
  on every LLM call to keep a shared counter, which costs more than the
  precision is worth here. With maxScale=2 the ceiling is "about right",
  which is the correct amount of engineering for abuse mitigation.

The window is the UTC day, not a rolling 24h: a rolling window needs the
timestamp of every call retained to expire them, while a calendar day needs
one float and one date. Resets are therefore abrupt at 00:00 UTC by design.
"""
import datetime
import logging
import threading

from app import config

logger = logging.getLogger(__name__)


class DailyBudget:
    """Accumulates estimated spend for the current UTC day."""

    def __init__(self) -> None:
        # Guards the day-rollover read-modify-write. FastAPI runs handlers
        # across a thread pool (asyncio.to_thread throughout main.py), so
        # two concurrent requests can hit record() genuinely simultaneously.
        self._lock = threading.Lock()
        self._day: datetime.date | None = None
        self._spent_usd = 0.0

    @staticmethod
    def _today() -> datetime.date:
        return datetime.datetime.now(datetime.UTC).date()

    def record(self, cost_usd: float) -> float:
        """Add one call's estimated cost; returns the running daily total."""
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day = today
                self._spent_usd = 0.0
            self._spent_usd += cost_usd
            return self._spent_usd

    def spent_today(self) -> float:
        with self._lock:
            if self._today() != self._day:
                return 0.0
            return self._spent_usd

    def is_exceeded(self) -> bool:
        limit = config.DAILY_BUDGET_USD
        if limit <= 0:
            return False  # disabled
        return self.spent_today() >= limit

    def reset(self) -> None:
        """Clear state. For tests -- production relies on the date rollover."""
        with self._lock:
            self._day = None
            self._spent_usd = 0.0


_budget = DailyBudget()


def record_spend(cost_usd: float) -> float:
    return _budget.record(cost_usd)


def spent_today() -> float:
    return _budget.spent_today()


def is_exceeded() -> bool:
    return _budget.is_exceeded()


def reset() -> None:
    _budget.reset()
