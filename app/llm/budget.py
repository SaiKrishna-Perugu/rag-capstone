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
        # Estimated spend of requests admitted but not yet finished. Without
        # it the ceiling was check-then-spend: is_exceeded() only sees money
        # already recorded, so N requests arriving together all read the same
        # under-limit total, all pass, and all then make their paid calls.
        # At containerConcurrency 80 that is up to 80 requests past a ceiling
        # meant to stop the first one. Counting in-flight work closes the
        # window between admission and the first add_usage().
        self._reserved_usd = 0.0

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

    def try_admit(self, estimate_usd: float) -> bool:
        """Reserve `estimate_usd` for one request, or refuse it.

        Atomic check-and-reserve: the decision and the reservation happen
        under one lock, so concurrent callers cannot all observe the same
        pre-spend total. Every True must be paired with a release() in a
        finally block, or the reservation leaks and the process refuses
        traffic it could have served.

        Returns True when the budget is disabled -- an unset ceiling admits
        everything, same as is_exceeded().
        """
        limit = config.DAILY_BUDGET_USD
        if limit <= 0:
            return True
        with self._lock:
            if self._today() != self._day:
                self._day = self._today()
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
            if self._spent_usd + self._reserved_usd >= limit:
                return False
            self._reserved_usd += estimate_usd
            return True

    def release(self, estimate_usd: float) -> None:
        """Drop a reservation once the request is done.

        The real cost was recorded through record() by cost.add_usage() as
        each call completed; this only removes the placeholder. Floored at
        zero so a double release cannot drive the counter negative and
        silently raise the effective ceiling.
        """
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - estimate_usd)

    def reserved(self) -> float:
        with self._lock:
            return self._reserved_usd

    def reset(self) -> None:
        """Clear state. For tests -- production relies on the date rollover."""
        with self._lock:
            self._day = None
            self._spent_usd = 0.0
            self._reserved_usd = 0.0


_budget = DailyBudget()


def record_spend(cost_usd: float) -> float:
    return _budget.record(cost_usd)


def spent_today() -> float:
    return _budget.spent_today()


def is_exceeded() -> bool:
    return _budget.is_exceeded()


def try_admit(estimate_usd: float | None = None) -> bool:
    return _budget.try_admit(
        config.BUDGET_REQUEST_ESTIMATE_USD if estimate_usd is None else estimate_usd
    )


def release(estimate_usd: float | None = None) -> None:
    _budget.release(
        config.BUDGET_REQUEST_ESTIMATE_USD if estimate_usd is None else estimate_usd
    )


def reserved() -> float:
    return _budget.reserved()


def reset() -> None:
    _budget.reset()
