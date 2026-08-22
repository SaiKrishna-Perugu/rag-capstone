"""
OpenTelemetry-instrumented metrics for production monitoring.

Tracks request counts, groundedness verdicts, latency, and error rates as
real OpenTelemetry instruments (Counter/Histogram) rather than a hand-rolled
in-process dataclass -- the standard, portable instrumentation pattern, and
usable by any OTel-compatible backend, not just this app's own /metrics
route.

Two MetricReaders are registered on the MeterProvider:
  1. PrometheusMetricReader -- always on, needs no GCP config. Backs
     GET /metrics (Prometheus text exposition format, pulled on request
     via prometheus_client.generate_latest() -- no separate HTTP server).
  2. PeriodicExportingMetricReader + CloudMonitoringMetricsExporter --
     opt-in (OTEL_GCP_EXPORT=true, requires GCP_PROJECT_ID), pushes
     aggregates to Cloud Monitoring every 60s so metrics are centralized
     and durable across Cloud Run instances instead of only visible
     per-instance. Off by default -- local dev/tests need zero GCP setup.

Known trade-offs, accepted deliberately:
  - opentelemetry-exporter-gcp-monitoring is pre-1.0 (alpha) and marked
    deprecated upstream in favor of native OTLP ingestion. Still the
    current, working, Google-documented path as of this writing.
  - PeriodicExportingMetricReader exports on a background timer thread.
    Cloud Run only allocates CPU during active request handling by
    default, so pushes may stall/batch between requests unless "CPU
    always allocated" is enabled on the service.
"""

from opentelemetry import metrics as otel_metrics_api  # this module IS app.metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider

from app import config

_readers = [PrometheusMetricReader()]

if config.OTEL_GCP_EXPORT and config.GCP_PROJECT_ID:
    from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    _readers.append(
        PeriodicExportingMetricReader(
            CloudMonitoringMetricsExporter(project_id=config.GCP_PROJECT_ID),
            export_interval_millis=60_000,
        )
    )

otel_metrics_api.set_meter_provider(MeterProvider(metric_readers=_readers))
_meter = otel_metrics_api.get_meter("rag_capstone")

_requests_total = _meter.create_counter(
    "rag_requests_total", description="Requests per endpoint."
)
_groundedness_verdicts = _meter.create_counter(
    "rag_groundedness_verdicts_total", description="Groundedness verdicts returned."
)
_errors_total = _meter.create_counter(
    "rag_errors_total", description="Errors per endpoint."
)
_latency_histogram = _meter.create_histogram(
    "rag_request_latency_ms", unit="ms", description="Request latency."
)
_retrieval_empty_total = _meter.create_counter(
    "rag_retrieval_empty_total", description="Requests where retrieval returned no chunks."
)
_agent_retry_total = _meter.create_counter(
    "rag_agent_retry_total", description="Agentic loop retries."
)
_llm_tokens_total = _meter.create_counter(
    "rag_llm_tokens_total", description="LLM tokens consumed, by model and direction."
)
# Counter, not a gauge: a monotonic total is what lets a dashboard derive
# spend rate over any window. Recorded in micro-USD as an integer because
# OpenTelemetry counters are integral and a single flash-lite call costs
# far less than one cent -- summing floats-rounded-to-zero would report
# nothing at all.
_llm_cost_micro_usd_total = _meter.create_counter(
    "rag_llm_cost_micro_usd_total",
    description="Estimated LLM spend in millionths of a USD, by model.",
)
# Counted once per trip, not once per failed call while open -- otherwise
# the number measures traffic during an outage rather than how often the
# provider actually broke.
_llm_circuit_opened_total = _meter.create_counter(
    "rag_llm_circuit_opened_total",
    description="Times a provider's circuit breaker tripped open.",
)
_llm_failover_total = _meter.create_counter(
    "rag_llm_failover_total",
    description="LLM calls served by the fallback provider instead of the primary.",
)


_budget_exceeded_total = _meter.create_counter(
    "rag_budget_exceeded_total",
    description="Requests refused because the daily LLM spend ceiling was reached.",
)
_injection_blocked_total = _meter.create_counter(
    "rag_injection_blocked_total",
    description="Requests refused by prompt-injection screening, by reason.",
)
_prompt_leak_total = _meter.create_counter(
    "rag_prompt_leak_total",
    description="Answers suppressed because they echoed this app's system prompt.",
)


# --- Public API ---------------------------------------------------------

def record_request(endpoint: str) -> None:
    _requests_total.add(1, {"endpoint": endpoint})


def record_groundedness(verdict: str) -> None:
    _groundedness_verdicts.add(1, {"verdict": verdict})


def record_error(endpoint: str) -> None:
    _errors_total.add(1, {"endpoint": endpoint})


def record_latency(latency_ms: int) -> None:
    _latency_histogram.record(latency_ms)


def record_empty_retrieval() -> None:
    _retrieval_empty_total.add(1)


def record_agent_retry() -> None:
    _agent_retry_total.add(1)


def record_llm_usage(
    model: str, input_tokens: int, output_tokens: int, cost_usd: float
) -> None:
    """Token counts and estimated spend for one LLM call."""
    _llm_tokens_total.add(input_tokens, {"model": model, "direction": "input"})
    _llm_tokens_total.add(output_tokens, {"model": model, "direction": "output"})
    _llm_cost_micro_usd_total.add(round(cost_usd * 1_000_000), {"model": model})


def record_circuit_opened(provider: str) -> None:
    """A provider's circuit breaker tripped (app/llm/circuit.py)."""
    _llm_circuit_opened_total.add(1, {"provider": provider})


def record_llm_failover(from_provider: str, to_provider: str) -> None:
    """A call was served by the fallback provider instead of the primary."""
    _llm_failover_total.add(1, {"from": from_provider, "to": to_provider})


def record_injection_blocked(reason: str) -> None:
    """A question was refused by input screening (app/api/security.py)."""
    _injection_blocked_total.add(1, {"reason": reason})


def record_prompt_leak() -> None:
    """An answer was suppressed for containing system-prompt text."""
    _prompt_leak_total.add(1)


def record_budget_exceeded() -> None:
    """A request was refused by the daily spend ceiling (app/llm/budget.py)."""
    _budget_exceeded_total.add(1)
