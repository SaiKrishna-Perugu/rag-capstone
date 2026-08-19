"""
Production middleware: tiered access control and optional Firebase identity.

Two deliberately different postures live here:

* ``AccessControlMiddleware`` **gates** -- it decides whether a request is
  allowed to reach a route at all.
* ``IdentityMiddleware`` **enriches** -- it resolves who the caller is when
  they present a Firebase token and moves on when they don't. It never
  rejects anything.

Keeping both is intentional. The public demo runs with no API key, so
everyone reaches the app, and Firebase identity only decides how much they
may upload. Collapsing the two into one "auth" layer would lose that
distinction and make it easy to accidentally wall off the demo.

Why access control is tiered rather than a single switch
--------------------------------------------------------
It used to be one boolean: with ``API_KEY`` set every route needed the key,
with it unset every route was open. Both positions are wrong for a demo that
is *supposed* to be publicly usable. Production runs with no key, which meant
``GET /metrics`` (token counts, spend, latency, error rates) and
``POST /internal/process-ingest-job`` (triggers real ingestion work) were
both callable by anyone with the URL. Verified against the live service, not
inferred: ``/metrics`` returned 200 to an unauthenticated request.

So the surface is split by *who the route is for*, not by one global flag:

===========  ==========================================  =====================
Tier         Routes                                      Control
===========  ==========================================  =====================
Probe        /health, /ready                             always open
Public       /, /config, /ask*, /upload, /jobs/*, /docs  open, unless API_KEY
Admin        /metrics                                    X-Admin-Key, else 404
Internal     /internal/*                                 Cloud Tasks OIDC only
===========  ==========================================  =====================

Three details worth keeping:

* **Admin returns 404, not 401.** A 401 confirms the route exists to anyone
  probing; a 404 says nothing.
* **``API_KEY`` still makes a whole deployment private**, which is what
  staging uses it for. The admin and internal tiers sit *on top* of that
  rather than replacing it, so a private deployment keeps working unchanged
  while the public one stops leaking.
* **Unknown paths 404 at the middleware.** The default is closed: a route
  added to ``main.py`` is unreachable until it is listed here. That is a
  small maintenance cost bought deliberately, because the failure it
  prevents -- a new endpoint silently inheriting public access -- is exactly
  how ``/internal/process-ingest-job`` ended up exposed.

Other middleware (CORS, rate limiting) is configured directly on the
FastAPI app in main.py because FastAPI/Starlette provide first-class
support for those -- no need to reinvent them as custom middleware.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import auth, config

logger = logging.getLogger(__name__)

# Liveness/readiness probes. Never gated by anything -- Cloud Run calls these
# itself and cannot present a key, so gating them takes the service down.
_PROBE_PATHS = frozenset({"/health", "/ready"})

# The demo surface. Open when API_KEY is unset (the public deployment);
# requires the key when it is set (staging).
#
# /docs, /openapi.json and /redoc are deliberately here rather than in the
# admin tier: they expose no secrets, and an interactive API explorer is a
# portfolio asset someone may genuinely want to open. /metrics is the route
# that actually leaks, and it is the one gated below.
_PUBLIC_PATHS = frozenset({
    "/", "/config",
    "/ask", "/ask-stream", "/ask-agentic", "/upload",
    "/documents",
    "/docs", "/openapi.json", "/redoc",
})

# Prefixes whose sub-paths follow the public tier -- /jobs/{job_id} and
# /documents/{filename}. Both are session-scoped by their handlers; being in
# the public tier only means "reachable without an API key", not "unowned".
_PUBLIC_PREFIXES = ("/jobs/", "/documents/")

# Operator surface. Leaks spend, token counts and error rates.
_ADMIN_PATHS = frozenset({"/metrics"})

_INTERNAL_PREFIX = "/internal/"

def _not_found() -> JSONResponse:
    # A fresh response per call: Starlette responses are single-use.
    return JSONResponse(status_code=404, content={"detail": "Not Found"})


def _verify_oidc_token(request: Request) -> bool:
    """Verify a Cloud Tasks OIDC token on an /internal/* request.

    Cloud Tasks signs each task with a service account identity and the
    target URL as audience, so a token is useless to anyone who cannot make
    the queue mint one. That is what turns /internal/* from
    "undocumented" into "unreachable".

    Falls back to the API key when no Tasks service account is configured,
    so a deployment mid-migration keeps working -- but if neither is
    configured this returns False rather than standing open. Denying an
    internal endpoint is a broken upload; leaving it open is a stranger
    running ingestion on your database.
    """
    if not config.TASKS_SERVICE_ACCOUNT_EMAIL:
        # Not yet migrated to OIDC: fall back to the shared key, and deny
        # outright if that is unset too.
        provided = request.headers.get("X-API-Key", "")
        return bool(config.API_KEY) and provided == config.API_KEY

    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        return False

    token = header.split(" ", 1)[1].strip()
    if not token:
        return False

    try:
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(
            token,
            # Reuses app/auth.py's pooled transport rather than building a
            # second one -- it exists to avoid a TLS handshake per call.
            auth._get_transport(),
            audience=config.INGEST_TARGET_URL or None,
        )
    except Exception as e:
        logger.warning(f"Rejected /internal request: OIDC verification failed ({e})")
        return False

    email = claims.get("email")
    if email != config.TASKS_SERVICE_ACCOUNT_EMAIL:
        logger.warning(f"Rejected /internal request: unexpected OIDC identity {email!r}")
        return False
    if not claims.get("email_verified", False):
        logger.warning("Rejected /internal request: OIDC email not verified")
        return False
    return True


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Route requests to the probe / public / admin / internal tiers above."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 1. Probes -- always open, no exceptions.
        if path in _PROBE_PATHS:
            return await call_next(request)

        # 2. Internal -- Cloud Tasks only. Checked before everything else so
        #    no other tier's rules can ever grant access here.
        if path.startswith(_INTERNAL_PREFIX):
            if not _verify_oidc_token(request):
                return JSONResponse(status_code=403, content={"detail": "Forbidden"})
            return await call_next(request)

        # 3. Admin -- separate secret from anything the browser UI holds.
        #    404 rather than 401 so probing does not confirm the route.
        if path in _ADMIN_PATHS:
            if not config.ADMIN_KEY:
                return _not_found()
            if request.headers.get("X-Admin-Key", "") != config.ADMIN_KEY:
                return _not_found()
            return await call_next(request)

        # 4. Public -- open by design on the demo, key-gated when API_KEY is
        #    set (staging, or any deployment meant to be wholly private).
        if path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES):
            if config.API_KEY and request.headers.get("X-API-Key", "") != config.API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )
            return await call_next(request)

        # 5. Anything unlisted is closed by default.
        return _not_found()


class IdentityMiddleware(BaseHTTPMiddleware):
    """Attach ``request.state.identity`` from an optional Firebase token.

    Always sets the attribute -- ``auth.ANONYMOUS`` when there's no token,
    the token is invalid, or Firebase isn't configured -- so handlers can
    read it unconditionally without a ``getattr`` dance.

    This middleware has no rejection path by design. An expired token left
    in a browser tab degrades that visitor to the public experience rather
    than locking them out, which is the correct behaviour for a demo whose
    entire purpose is being immediately usable by a stranger.
    """

    async def dispatch(self, request: Request, call_next):
        request.state.identity = auth.identity_from_header(
            request.headers.get("Authorization")
        )
        return await call_next(request)
