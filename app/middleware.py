"""
Production middleware: API key authentication.

Other middleware (CORS, rate limiting) is configured directly on the
FastAPI app in main.py because FastAPI/Starlette provide first-class
support for those -- no need to reinvent them as custom middleware.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import config

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
