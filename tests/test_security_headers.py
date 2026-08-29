"""Regression tests for the audit fixes: security headers and the
rate-limit key function."""

from unittest.mock import MagicMock

from app.main import _rate_limit_key


def test_responses_carry_security_headers(client):
    """Every response -- including probe endpoints -- carries the
    hardening headers added by add_security_headers()."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    # Report-Only: enforcement is deliberately deferred until the Firebase
    # popup flow can be smoke-tested against the allowlist.
    assert resp.headers["Content-Security-Policy-Report-Only"].startswith("default-src 'self'")


def test_rate_limit_key_prefers_leftmost_forwarded_for():
    """Behind Cloud Run's GFE the socket peer is the proxy for everyone --
    the leftmost X-Forwarded-For entry is what distinguishes clients."""
    req = MagicMock()
    req.headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1, 169.254.8.129"}
    assert _rate_limit_key(req) == "203.0.113.7"


def test_rate_limit_key_falls_back_to_socket_address():
    """No X-Forwarded-For (direct local access, tests) -- same behaviour as
    the previous get_remote_address keying."""
    req = MagicMock()
    req.headers = {}
    req.client.host = "127.0.0.1"
    assert _rate_limit_key(req) == "127.0.0.1"


def test_rate_limit_key_ignores_whitespace_only_forwarded_for():
    """A malformed header value must not become an empty bucket key that
    every such client shares."""
    req = MagicMock()
    req.headers = {"x-forwarded-for": " , "}
    req.client.host = "127.0.0.1"
    assert _rate_limit_key(req) == "127.0.0.1"
