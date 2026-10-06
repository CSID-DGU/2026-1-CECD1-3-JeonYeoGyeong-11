"""Signed session cookies -- stdlib only, no new dependency.

A random secret is generated once per process at import time. Sessions do not
survive a server restart; that is an accepted gap for this stage, same spirit
as the other "의도적으로 아직 없는 것" callouts in this package's README.
Cookies are set SameSite=Lax (blocks cross-site POST, the common CSRF vector)
but there is no per-form CSRF token yet -- see README.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Optional

_SECRET = secrets.token_bytes(32)
_MAX_AGE_SECONDS = 7 * 24 * 3600

CUSTOMER_COOKIE = "customer_session"
SELLER_COOKIE = "seller_session"


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def sign(payload: dict[str, Any]) -> str:
    body = {**payload, "exp": time.time() + _MAX_AGE_SECONDS}
    body_b64 = _b64encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_SECRET, body_b64.encode("ascii"), hashlib.sha256).digest()
    return "%s.%s" % (body_b64, _b64encode(signature))


def unsign(token: Optional[str]) -> Optional[dict[str, Any]]:
    if not token or "." not in token:
        return None
    body_b64, _, signature_b64 = token.partition(".")
    expected = hmac.new(_SECRET, body_b64.encode("ascii"), hashlib.sha256).digest()
    try:
        given = _b64decode(signature_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected, given):
        return None
    try:
        body = json.loads(_b64decode(body_b64))
    except Exception:
        return None
    if body.get("exp", 0) < time.time():
        return None
    return body
