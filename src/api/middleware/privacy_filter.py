"""PII suppression middleware for API responses.

@requirement NFR-007
@design docs/plans/2026-06-05-causalcredit-architecture-design.md §D6

Two layers of suppression on JSON response bodies:

1. **Direct identifiers** (``name`` / ``email`` / ``phone`` / ``ssn`` /
   ``passport`` / ``address`` / ``dob``) inside ``features`` blocks are
   replaced with the string ``[REDACTED]``.
2. **Quasi-identifiers** (``age`` → 10-year band, ``income`` → 25K
   band) are coarsened so an attacker cannot single out an individual
   by joining these against external tables.

The filter is **enabled by default** (NFR-007) but can be disabled
either by passing ``enabled=False`` to :class:`PrivacyFilterMiddleware`
or by setting ``middleware.pii_filter: false`` in the YAML.

Admin requests (auth middleware already validated ``X-Admin-API-Key``)
get the unfiltered payload by default — admins need full resolution
for incident triage.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.utils.config import load_config

logger = logging.getLogger("causalcredit.privacy")

DIRECT_IDENTIFIERS = {
    "name", "email", "phone", "ssn", "passport", "address", "dob",
    "full_name", "first_name", "last_name", "mobile", "id_number",
}
QUASI_IDENTIFIERS = {
    "age", "income", "region", "occupation", "dependents_count",
    "zip", "postcode", "gender",
}


class PrivacyFilter:
    """Stateful suppressor for the response body. Pure function; safe to test."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def suppress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return data
        if not isinstance(data, dict):
            return data
        return self._suppress_recursive(data)

    # ------------------------------------------------------------------
    def _suppress_recursive(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in DIRECT_IDENTIFIERS and isinstance(v, (str, int, float)):
                    out[k] = "[REDACTED]"
                elif k == "age" and isinstance(v, (int, float)):
                    out[k] = self._age_to_range(v)
                elif k == "income" and isinstance(v, (int, float)):
                    out[k] = self._income_to_band(v)
                elif k in {"region", "zip", "postcode"} and isinstance(v, str) and len(v) > 2:
                    out[k] = v[:2] + "***"
                elif k == "dependents_count" and isinstance(v, (int, float)):
                    out[k] = self._deps_to_band(v)
                else:
                    out[k] = self._suppress_recursive(v)
            return out
        if isinstance(obj, list):
            return [self._suppress_recursive(x) for x in obj]
        return obj

    # ------------------------------------------------------------------
    @staticmethod
    def _age_to_range(age: float) -> str:
        if age < 25:
            return "18-24"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        if age < 65:
            return "55-64"
        return "65+"

    @staticmethod
    def _income_to_band(income: float) -> str:
        if income < 25_000:
            return "<25K"
        if income < 50_000:
            return "25K-50K"
        if income < 100_000:
            return "50K-100K"
        if income < 250_000:
            return "100K-250K"
        return ">250K"

    @staticmethod
    def _deps_to_band(deps: float) -> str:
        if deps <= 0:
            return "0"
        if deps <= 2:
            return "1-2"
        if deps <= 4:
            return "3-4"
        return "5+"


def _resolve_pii_enabled(ctor_value: Optional[bool]) -> bool:
    if ctor_value is not None:
        return bool(ctor_value)
    env_val = os.environ.get("CAUSALCREDIT_PII_FILTER")
    if env_val is not None:
        return env_val.lower() in {"1", "true", "yes", "on"}
    try:
        cfg = load_config()
        mw = cfg.get("middleware", {}) if isinstance(cfg, dict) else {}
        v = mw.get("pii_filter", True)
        return bool(v)
    except Exception:
        return True


class PrivacyFilterMiddleware(BaseHTTPMiddleware):
    """Wraps responses and re-writes JSON bodies through :class:`PrivacyFilter`.

    The middleware is conservative: if JSON parsing fails or the body
    doesn't look like a dict, it leaves the response untouched. Streamed
    responses (``StreamingResponse``) are also passed through unchanged
    since we can't safely re-parse without buffering the whole stream.
    """

    def __init__(self, app, enabled: Optional[bool] = None) -> None:
        super().__init__(app)
        self._filter = PrivacyFilter(enabled=_resolve_pii_enabled(enabled))
        self._admin_paths = ("/api/v1/admin/",)

    def _is_admin(self, request: Request) -> bool:
        path = request.url.path
        return any(path.startswith(p) for p in self._admin_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if not self._filter.enabled:
            return response
        if self._is_admin(request):
            return response
        ctype = response.headers.get("content-type", "")
        if "application/json" not in ctype.lower():
            return response

        body_bytes = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                body_bytes += chunk.encode("utf-8")
            else:
                body_bytes += chunk
        try:
            data = json.loads(body_bytes)
        except (ValueError, UnicodeDecodeError):
            # Not JSON-decodable — restore raw body and return.
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        suppressed = self._filter.suppress(data)
        new_body = json.dumps(suppressed, ensure_ascii=False).encode("utf-8")
        new_headers = dict(response.headers)
        new_headers["content-length"] = str(len(new_body))
        return Response(
            content=new_body,
            status_code=response.status_code,
            headers=new_headers,
            media_type=response.media_type,
        )
