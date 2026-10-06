"""Signed session cookies and per-session CSRF tokens -- stdlib only, no new dependency.

The signing key comes from MERCHANT_SECRET when it is set, so sessions survive
a server restart. Without it a random key is generated once per process (fine
for tests and a quick local run; every restart logs everyone out).

Cookies are host-only, HttpOnly and SameSite=Lax. Every state-changing form of
a logged-in user also carries a CSRF token bound to that session cookie
(csrf_token/check_csrf), which is the "호스트별 쿠키와 CSRF" interfaces.md §6
asks for. Login and signup forms have no session yet and rely on SameSite=Lax.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from typing import Any, Optional


def _load_secret() -> bytes:
    configured = os.environ.get("MERCHANT_SECRET")
    if configured:
        # Derived, not used raw: MERCHANT_SECRET may later sign other things too.
        return hashlib.sha256(b"merchant-session-v1:" + configured.encode("utf-8")).digest()
    return secrets.token_bytes(32)


_SECRET = _load_secret()
_MAX_AGE_SECONDS = 7 * 24 * 3600



def _cookie_name(role: str, seller_id: str) -> str:
    """One cookie per seller: run_local.py starts every seller on 127.0.0.1 with
    only the port differing, and browsers do not separate cookies by port
    (working-agreement.md §3), so a shared name would make logging in to one
    seller log you out of the next. Non-token characters are dropped."""
    return "%s_session_%s" % (role, re.sub(r"[^A-Za-z0-9_-]", "", seller_id))


def customer_cookie(seller_id: str) -> str:
    return _cookie_name("customer", seller_id)


def seller_cookie(seller_id: str) -> str:
    return _cookie_name("seller", seller_id)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def sign(payload: dict[str, Any]) -> str:
    # nonce: two logins of the same account get different cookies, hence different CSRF tokens.
    body = {**payload, "exp": time.time() + _MAX_AGE_SECONDS, "nonce": secrets.token_hex(8)}
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


def csrf_token(session_token: Optional[str]) -> Optional[str]:
    """The CSRF token for one session cookie value, or None without a session."""
    if not session_token:
        return None
    return _b64encode(hmac.new(_SECRET, b"csrf:" + session_token.encode("ascii"), hashlib.sha256).digest())


def check_csrf(session_token: Optional[str], given: Optional[str]) -> bool:
    expected = csrf_token(session_token)
    return expected is not None and given is not None and hmac.compare_digest(expected, given)
