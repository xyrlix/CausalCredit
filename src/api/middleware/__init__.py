"""Middleware package for the CausalCredit API.

Three middlewares are wired into ``src.api.app`` (in this order, last-added
is outermost in the request path):

1. :class:`PrivacyFilterMiddleware`  — masks PII in JSON responses
2. :class:`AuthMiddleware`           — X-API-Key / X-Admin-API-Key enforcement
3. :class:`RateLimitMiddleware`      — sliding-window per-IP throttle

All three are off by default (zero/empty config values) so local dev and
unit tests need no setup. Production deployments set the values in
``configs/config.yaml::middleware`` or via environment variables
``CAUSALCREDIT_API_KEY`` / ``CAUSALCREDIT_ADMIN_API_KEY`` /
``CAUSALCREDIT_RATE_LIMIT_PER_SEC`` / ``CAUSALCREDIT_PII_FILTER`` (env
wins over YAML).
"""

from .auth import AuthMiddleware
from .privacy_filter import PrivacyFilter, PrivacyFilterMiddleware
from .rate_limit import RateLimitMiddleware

__all__ = [
    "AuthMiddleware",
    "PrivacyFilter",
    "PrivacyFilterMiddleware",
    "RateLimitMiddleware",
]
