"""
Production middleware: API key authentication and optional Firebase identity.

Two deliberately different postures live here:

* ``APIKeyMiddleware`` **gates** -- a wrong key is a 401. Used by the staging
  deployment and anywhere the whole service should be private.
* ``IdentityMiddleware`` **enriches** -- it resolves who the caller is when
  they present a Firebase token and moves on when they don't. It never
  rejects anything.

Keeping both is intentional. The public demo runs with no API key, so
everyone reaches the app, and Firebase identity only decides how much they
may upload. Collapsing the two into one "auth" layer would lose that
distinction and make it easy to accidentally wall off the demo.

Other middleware (CORS, rate limiting) is configured directly on the
FastAPI app in main.py because FastAPI/Starlette provide first-class
support for those -- no need to reinvent them as custom middleware.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import auth, config

# Paths that bypass API key authentication (public endpoints).
_PUBLIC_PATHS = frozenset({"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid ``X-API-Key`` header.

    Disabled entirely when ``config.API_KEY`` is empty (the default) so
    that local development and demo deployments work without any key.
    Enable by setting ``API_KEY`` in ``.env`` or as a Cloud Run
    environment variable / Secret Manager reference.
    """

    async def dispatch(self, request: Request, call_next):
        # Auth disabled when no key is configured.
        if not config.API_KEY:
            return await call_next(request)

        # Public endpoints never require auth.
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != config.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )

        return await call_next(request)


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
