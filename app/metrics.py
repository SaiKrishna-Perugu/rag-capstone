"""
Lightweight in-memory metrics for production monitoring.

Tracks request counts, groundedness verdicts, latency percentiles, and
error rates. Exposed via ``GET /metrics`` as JSON -- scrapeable by Cloud
Monitoring custom metrics or any polling-based dashboard.

These reset on container restart, which is fine for Cloud Run (stateless
containers, autoscaled). For durable metrics across restarts, push to
Cloud Monitoring instead of serving them.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field

_lock = threading.Lock()


@dataclass
class _Metrics:
    """All counters and histograms, guarded by ``_lock``."""
    requests_total: dict = field(default_factory=lambda: defaultdict(int))
    groundedness_verdicts: dict = field(default_factory=lambda: defaultdict(int))
    errors_total: dict = field(default_factory=lambda: defaultdict(int))
    retrieval_empty_count: int = 0
    agent_retry_total: int = 0
    latencies_ms: list = field(default_factory=list)

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of all metrics."""
        latencies = sorted(self.latencies_ms) if self.latencies_ms else []
        n = len(latencies)

        def percentile(p: float) -> float:
            if not latencies:
                return 0.0
            idx = int(p / 100 * n)
            return latencies[min(idx, n - 1)]

        return {
            "requests_total": dict(self.requests_total),
            "groundedness_verdicts": dict(self.groundedness_verdicts),
            "errors_total": dict(self.errors_total),
            "retrieval_empty_count": self.retrieval_empty_count,
            "agent_retry_total": self.agent_retry_total,
            "latency_ms": {
                "count": n,
                "p50": percentile(50),
                "p95": percentile(95),
                "p99": percentile(99),
            },
        }


_metrics = _Metrics()


# --- Public API (thread-safe) ------------------------------------------------

def record_request(endpoint: str) -> None:
    with _lock:
        _metrics.requests_total[endpoint] += 1


def record_groundedness(verdict: str) -> None:
    with _lock:
        _metrics.groundedness_verdicts[verdict] += 1


def record_error(endpoint: str) -> None:
    with _lock:
        _metrics.errors_total[endpoint] += 1


def record_latency(latency_ms: int) -> None:
    with _lock:
        _metrics.latencies_ms.append(latency_ms)
        # Cap stored latencies to prevent unbounded memory growth.
        if len(_metrics.latencies_ms) > 10_000:
            _metrics.latencies_ms = _metrics.latencies_ms[-5_000:]


def record_empty_retrieval() -> None:
    with _lock:
        _metrics.retrieval_empty_count += 1


def record_agent_retry() -> None:
    with _lock:
        _metrics.agent_retry_total += 1


def get_snapshot() -> dict:
    with _lock:
        return _metrics.snapshot()
