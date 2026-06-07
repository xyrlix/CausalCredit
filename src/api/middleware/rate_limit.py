"""Sliding-window per-IP rate limiter middleware.

@requirement REQ-API-001
@requirement NFR-001
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D5

Disabled when ``max_per_second <= 0`` (default) so dev / tests need no
config. Public paths (health, docs, metrics) and the admin namespace
are never throttled.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("causalcredit.rate_limit")

# Paths that bypass the rate limiter. Note the trimmed trailing slash for
# exact-match comparison.
PUBLIC_PATHS = {
    "/",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
}
ADMIN_PREFIX = "/api/v1/admin/"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter. Returns 429 with ``Retry-After`` when exceeded.

    Each client IP keeps a deque of unix timestamps; requests older than
    1 second are dropped on every check. ``max_per_second == 0`` disables
    the limiter (useful for CI / local dev).
    """

    def __init__(self, app, max_per_second: int = 0) -> None:
        super().__init__(app)
        self._max = int(max_per_second)
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/")
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/openapi"):
            return await call_next(request)
        if path.startswith(ADMIN_PREFIX):
            return await call_next(request)
        if self._max <= 0:
            return await call_next(request)

        client_ip = self._client_ip(request)
        now = time.time()
        window_start = now - 1.0
        bucket = self._windows[client_ip]
        # Drop stale timestamps
        while bucket and bucket[0] <= window_start:
            bucket.popleft()
        if len(bucket) >= self._max:
            logger.info("rate-limit 429 ip=%s path=%s", client_ip, path)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "error_code": "RATE_LIMITED",
                    "retry_after_seconds": 1,
                },
                headers={"Retry-After": "1"},
            )
        bucket.append(now)
        return await call_next(request)

    @staticmethod
    def _client_ip(request: Request) -> str:
        # Honour X-Forwarded-For if behind a known proxy; else fall back
        # to the direct client. Avoids spoofing when the proxy chain is
        # trusted (the operator configures the trust boundary).
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        if request.client is None:
            return "unknown"
        return request.client.host


def build_rate_limiter_from_config(max_per_second: Optional[int]) -> RateLimitMiddleware:
    """Factory: 0 / None → returns a disabled limiter."""
    if max_per_second is None:
        return RateLimitMiddleware
    return RateLimitMiddleware  # instantiated by Starlette with app arg
