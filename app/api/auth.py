"""
Optional Firebase identity.

Deliberately *additive*: this module never rejects a request. A valid
Firebase ID token identifies the caller and unlocks raised limits; an
absent, expired, or malformed token simply leaves the caller anonymous
with the public defaults. That inversion is the whole point -- this
deployment is a public demo, and an auth layer that gates the front door
would make it unusable for the people it exists to show.

Verification uses google-auth's `verify_firebase_token`, which checks
Google's published signing keys, the issuer, the audience, and expiry.
google-auth is already an indirect dependency via google-cloud-firestore,
so this adds no new package -- `firebase-admin` would pull in a great deal
more for the one call we need.
"""
import logging
from dataclasses import dataclass

from google.auth.transport import requests as ga_requests
from google.oauth2 import id_token

from app import config

logger = logging.getLogger(__name__)

# Reused across requests: it holds a connection pool for fetching Google's
# signing certificates, which are cached and rotated infrequently. Building
# one per request would add a TLS handshake to every authenticated call.
_transport = None


def _get_transport():
    global _transport
    if _transport is None:
        _transport = ga_requests.Request()
    return _transport


@dataclass(frozen=True)
class Identity:
    """A resolved caller. `uid` is None for anonymous visitors."""

    uid: str | None = None
    email: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.uid is not None

    @property
    def log_value(self) -> str:
        """What goes in structured logs -- the uid, not the email, so logs
        carry a stable identifier without also accumulating personal data."""
        return self.uid or "anonymous"


ANONYMOUS = Identity()


def identity_from_header(authorization: str | None) -> Identity:
    """Resolve `Authorization: Bearer <firebase-id-token>` to an Identity.

    Returns ANONYMOUS for anything that isn't a currently-valid token --
    missing header, wrong scheme, expired, wrong audience, bad signature.
    A rejected token is logged at debug and treated as "not signed in"
    rather than raising: a stale token in someone's browser tab should
    degrade them to the public experience, not lock them out of a demo.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return ANONYMOUS

    audience = config.FIREBASE_PROJECT_ID
    if not audience:
        # Firebase isn't configured for this deployment; nothing to verify
        # against. Anonymous is the correct answer, not an error.
        return ANONYMOUS

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return ANONYMOUS

    try:
        claims = id_token.verify_firebase_token(
            token,
            _get_transport(),
            audience=audience,
            # Small tolerance for client/server clock drift; without it a
            # freshly-minted token can be rejected as "not yet valid".
            clock_skew_in_seconds=10,
        )
    except Exception as e:
        logger.debug(f"Firebase token rejected, treating as anonymous: {e}")
        return ANONYMOUS

    if not claims:
        return ANONYMOUS

    return Identity(uid=claims.get("sub"), email=claims.get("email"))


def upload_limits(identity: Identity) -> tuple[int, int]:
    """(max_files, max_size_mb) for this caller.

    Signing in raises the ceiling rather than being what grants access at
    all -- anonymous visitors can still upload, just less.

    The authed values are floored at the anonymous ones instead of being
    returned as configured. This used to be a plain swap, and the shipped
    defaults were 50MB anonymous against 10MB authed -- so signing in cut
    the per-file ceiling to a fifth, under a docstring promising the
    opposite. Since all four values are independently settable by env, the
    same inversion is one careless MAX_UPLOAD_SIZE_MB= away from returning;
    max() makes "authenticated is never worse off" a property of the code
    rather than of whoever last edited the config.
    """
    if identity.is_authenticated:
        return (
            max(config.MAX_UPLOAD_FILES_AUTHED, config.MAX_UPLOAD_FILES),
            max(config.MAX_UPLOAD_SIZE_MB_AUTHED, config.MAX_UPLOAD_SIZE_MB),
        )
    return config.MAX_UPLOAD_FILES, config.MAX_UPLOAD_SIZE_MB
