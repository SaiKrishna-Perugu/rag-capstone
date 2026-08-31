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


# --- groundedness opt-out is operator-controlled, not client-controlled -----

def test_hallucination_opt_out_ignored_on_the_public_tier(monkeypatch):
    """With no API_KEY the deployment is the open demo, where the
    groundedness verdict is the whole claim -- an anonymous caller must not
    be able to switch it off per request."""
    from app import config
    from app.main import _honor_hallucination_opt_out

    monkeypatch.setattr(config, "API_KEY", "")
    assert _honor_hallucination_opt_out(False) is True
    assert _honor_hallucination_opt_out(True) is True


def test_hallucination_opt_out_honored_when_api_key_gates_the_deployment(monkeypatch):
    """Staging and private integrations: the caller is known, and trading the
    verdict for latency is theirs to choose."""
    from app import config
    from app.main import _honor_hallucination_opt_out

    monkeypatch.setattr(config, "API_KEY", "a-real-key")
    assert _honor_hallucination_opt_out(False) is False
    assert _honor_hallucination_opt_out(True) is True


# --- CSP allowlist ----------------------------------------------------------
# Both assertions below pin violations MEASURED against the live service by
# driving the sign-in button in a real browser. Either would have broken
# sign-in silently the moment the policy was promoted to enforcing.

def _directive(policy: str, name: str) -> str:
    for part in policy.split("; "):
        if part.startswith(name + " "):
            return part
    return ""


def test_csp_allows_the_firebase_popup_helper_script(monkeypatch):
    """signInWithPopup loads apis.google.com/js/api.js. The host was in
    connect-src and frame-src but NOT script-src, which is the directive
    that actually governs loading it."""
    from app import config, main

    monkeypatch.setattr(config, "FIREBASE_WEB_API_KEY", "k")
    monkeypatch.setattr(config, "FIREBASE_AUTH_DOMAIN", "proj.firebaseapp.com")
    assert "https://apis.google.com" in _directive(main._csp_policy(), "script-src")


def test_csp_frames_the_deployments_own_auth_domain(monkeypatch):
    """The auth helper iframe comes from the project's authDomain, not from
    apis.google.com. Derived from config so it is right for whichever
    project is deployed, instead of one hostname baked into the source."""
    from app import config, main

    monkeypatch.setattr(config, "FIREBASE_WEB_API_KEY", "k")
    monkeypatch.setattr(config, "FIREBASE_AUTH_DOMAIN", "someone-else.firebaseapp.com")
    frame = _directive(main._csp_policy(), "frame-src")
    assert "https://someone-else.firebaseapp.com" in frame
    assert "hybrid-rag" not in frame, "a project hostname is hardcoded in the policy"


def test_csp_drops_auth_hosts_when_firebase_is_unconfigured(monkeypatch):
    """A deployment that cannot sign in should not advertise the endpoints
    for it. The UI hides the button in this case anyway."""
    from app import config, main

    monkeypatch.setattr(config, "FIREBASE_WEB_API_KEY", "")
    policy = main._csp_policy()
    assert _directive(policy, "frame-src") == "frame-src 'none'"
    assert "identitytoolkit" not in policy
    assert "apis.google.com" not in policy


def test_csp_is_still_report_only(client):
    """Deliberate. Promoting needs a completed Google sign-in, which no
    automated check here can perform -- and the post-redirect leg is exactly
    where an unmeasured violation would hide."""
    resp = client.get("/health")
    assert "Content-Security-Policy-Report-Only" in resp.headers
    assert "Content-Security-Policy" not in resp.headers
