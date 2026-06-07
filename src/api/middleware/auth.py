"""API-key authentication middleware.

@requirement REQ-API-001
@requirement NFR-002
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D5

Three roles:

* **public** — health, docs, OpenAPI, root (no key required)
* **admin**  — paths under ``/api/v1/admin/`` require ``X-Admin-API-Key``
* **inference** — every other path requires ``X-API-Key``

Both keys are configured via ``configs/config.yaml::middleware.api_key``
(and ``admin_api_key``) or env vars ``CAUSALCREDIT_API_KEY`` /
``CAUSALCREDIT_ADMIN_API_KEY``. Empty / unset key → that role is
unauthenticated (dev / CI default).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.utils.config import load_config

logger = logging.getLogger("causalcredit.auth")

# Trimmed paths that bypass all auth. The exact-match set is small; we
# also wildcard ``/docs*`` and ``/openapi*`` for Swagger UI assets.
PUBLIC_PATHS = {
    "/",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
}
ADMIN_PREFIX = "/api/v1/admin/"


def _load_api_keys() -> tuple[Optional[str], Optional[str]]:
    """Resolve (api_key, admin_api_key) from env-override + YAML.

    Env wins over YAML to keep prod-secrets out of the repo.
    """
    api_key = os.environ.get("CAUSALCREDIT_API_KEY")
    admin_key = os.environ.get("CAUSALCREDIT_ADMIN_API_KEY")
    if api_key and admin_key:
        return api_key, admin_key
    try:
        cfg = load_config()
    except Exception:
        return api_key or None, admin_key or None
    mw = cfg.get("middleware", {}) if isinstance(cfg, dict) else {}
    return (
        api_key or mw.get("api_key") or None,
        admin_key or mw.get("admin_api_key") or None,
    )


class AuthMiddleware(BaseHTTPMiddleware):
    """Per-request auth gate. Public / admin / inference as above."""

    def __init__(self, app, api_key: Optional[str] = None, admin_api_key: Optional[str] = None) -> None:
        super().__init__(app)
        # Resolve from ctor → env → YAML → None. Ctor wins (test override).
        env_api, env_admin = _load_api_keys()
        self._api_key = api_key if api_key is not None else env_api
        self._admin_key = admin_api_key if admin_api_key is not None else env_admin

    @property
    def api_key_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def admin_key_configured(self) -> bool:
        return bool(self._admin_key)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/")
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/openapi"):
            return await call_next(request)

        if path.startswith(ADMIN_PREFIX):
            return await self._check_admin(request, call_next)

        return await self._check_inference(request, call_next)

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------
    async def _check_admin(self, request: Request, call_next):
        supplied = request.headers.get("X-Admin-API-Key", "")
        if not self._admin_key:
            # No admin key configured → admin endpoints are disabled.
            logger.warning("admin endpoint hit but ADMIN_API_KEY not configured: %s", request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Admin API key not configured on server", "error_code": "UNAUTHORIZED"},
            )
        if supplied != self._admin_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid admin API key", "error_code": "UNAUTHORIZED"},
            )
        return await call_next(request)

    async def _check_inference(self, request: Request, call_next):
        if not self._api_key:
            # No inference key configured → open access (dev default).
            return await call_next(request)
        supplied = request.headers.get("X-API-Key", "")
        if supplied != self._api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key", "error_code": "UNAUTHORIZED"},
            )
        return await call_next(request)
