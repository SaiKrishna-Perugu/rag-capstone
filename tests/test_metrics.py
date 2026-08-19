from app import metrics


def test_recording_functions_do_not_raise():
    metrics.record_request("ask")
    metrics.record_groundedness("GROUNDED")
    metrics.record_error("ask")
    metrics.record_latency(123)
    metrics.record_empty_retrieval()
    metrics.record_agent_retry()


def test_metrics_endpoint_returns_prometheus_format(client, monkeypatch):
    """/metrics moved to the admin tier -- it exposes token counts, spend and
    error rates, which must not be public on a demo anyone can reach."""
    from app import config
    monkeypatch.setattr(config, "ADMIN_KEY", "admin-secret")
    metrics.record_request("ask")

    response = client.get("/metrics", headers={"X-Admin-Key": "admin-secret"})
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    assert "rag_requests_total" in body
    assert "rag_request_latency_ms" in body


def test_metrics_is_not_public(client, monkeypatch):
    """404 rather than 401, deliberately: a 401 confirms the route exists to
    anyone probing the API."""
    from app import config
    monkeypatch.setattr(config, "ADMIN_KEY", "admin-secret")
    assert client.get("/metrics").status_code == 404
    assert client.get("/metrics", headers={"X-Admin-Key": "wrong"}).status_code == 404


def test_metrics_stays_closed_when_no_admin_key_is_configured(client, monkeypatch):
    """The failure mode that mattered: production runs with no keys set, and
    /metrics was reachable by anyone with the URL. Absent config must mean
    closed, not open."""
    from app import config
    monkeypatch.setattr(config, "ADMIN_KEY", "")
    assert client.get("/metrics").status_code == 404
