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
