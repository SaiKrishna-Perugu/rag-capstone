from app import metrics


def test_recording_functions_do_not_raise():
    metrics.record_request("ask")
    metrics.record_groundedness("GROUNDED")
    metrics.record_error("ask")
    metrics.record_latency(123)
    metrics.record_empty_retrieval()
    metrics.record_agent_retry()


def test_metrics_endpoint_returns_prometheus_format(client):
    metrics.record_request("ask")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    assert "rag_requests_total" in body
    assert "rag_request_latency_ms" in body
