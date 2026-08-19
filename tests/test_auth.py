"""Optional Firebase identity.

The governing rule for every test here: auth is ADDITIVE. Nothing in this
module may ever turn a working anonymous request into a rejected one --
this deployment is a public demo, and a sign-in wall would defeat its
purpose. Most of these tests exist to prove the failure modes degrade to
"anonymous" rather than to an error.
"""
from unittest.mock import patch

from app import config
from app.api import auth


def test_no_header_is_anonymous():
    assert auth.identity_from_header(None) is auth.ANONYMOUS
    assert auth.identity_from_header("") is auth.ANONYMOUS


def test_wrong_scheme_is_anonymous():
    assert auth.identity_from_header("Basic abc123") is auth.ANONYMOUS
    assert auth.identity_from_header("Bearer") is auth.ANONYMOUS
    assert auth.identity_from_header("Bearer   ") is auth.ANONYMOUS


def test_unconfigured_firebase_is_anonymous(monkeypatch):
    """With no project configured there is nothing to verify against. That
    is the default deployment shape, and it must not error."""
    monkeypatch.setattr(config, "FIREBASE_PROJECT_ID", "")
    assert auth.identity_from_header("Bearer sometoken") is auth.ANONYMOUS


def test_invalid_token_degrades_to_anonymous(monkeypatch):
    """A stale token in a browser tab must drop the visitor to the public
    experience, not lock them out."""
    monkeypatch.setattr(config, "FIREBASE_PROJECT_ID", "test-project")
    with patch("app.api.auth.id_token.verify_firebase_token", side_effect=ValueError("expired")):
        assert auth.identity_from_header("Bearer expired-token") is auth.ANONYMOUS


def test_valid_token_resolves_identity(monkeypatch):
    monkeypatch.setattr(config, "FIREBASE_PROJECT_ID", "test-project")
    claims = {"sub": "uid-123", "email": "someone@example.com"}
    with patch("app.api.auth.id_token.verify_firebase_token", return_value=claims):
        ident = auth.identity_from_header("Bearer good-token")
    assert ident.is_authenticated
    assert ident.uid == "uid-123"
    assert ident.email == "someone@example.com"


def test_log_value_uses_uid_not_email(monkeypatch):
    """Logs get a stable identifier without accumulating personal data."""
    monkeypatch.setattr(config, "FIREBASE_PROJECT_ID", "test-project")
    claims = {"sub": "uid-123", "email": "someone@example.com"}
    with patch("app.api.auth.id_token.verify_firebase_token", return_value=claims):
        ident = auth.identity_from_header("Bearer good-token")
    assert ident.log_value == "uid-123"
    assert auth.ANONYMOUS.log_value == "anonymous"


def test_upload_limits_raise_when_authenticated(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_FILES", 3)
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB", 2)
    monkeypatch.setattr(config, "MAX_UPLOAD_FILES_AUTHED", 10)
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB_AUTHED", 10)

    assert auth.upload_limits(auth.ANONYMOUS) == (3, 2)
    assert auth.upload_limits(auth.Identity(uid="u1")) == (10, 10)
