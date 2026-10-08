"""Signup/login business logic for customers and this store's own seller staff.

Not a wire contract (no JSON schema under commerce/packages/contracts/):
purely local to one seller's orders.sqlite, like social_service.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import sqlite3
from typing import Any, Optional

from commerce.services.merchant_api import accounts_db as db
from commerce.services.merchant_api import nts_client

_PBKDF2_ITERATIONS = 200_000


class AccountError(Exception):
    """Signup/login failure meant to be shown back on the form, not a wire error."""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _hash_password(password: str) -> tuple[str, str]:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(digest.hex(), hash_hex)


# --- customers ---------------------------------------------------------------

def signup_customer(conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str,
                     display_name: str, password: str) -> dict[str, Any]:
    if not customer_id_local.strip() or not password:
        raise AccountError("아이디와 비밀번호를 입력하세요.")
    if len(password) < 4:
        raise AccountError("비밀번호는 4자 이상이어야 합니다.")
    if db.fetch_customer(conn, seller_id, customer_id_local) is not None:
        raise AccountError("이미 사용 중인 아이디입니다.")
    salt_hex, hash_hex = _hash_password(password)
    with conn:
        db.insert_customer(conn, seller_id, customer_id_local, display_name or customer_id_local,
                            hash_hex, salt_hex, _now_iso())
    return {"seller_id": seller_id, "customer_id_local": customer_id_local, "display_name": display_name}


def authenticate_customer(conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str, password: str) -> dict[str, Any]:
    row = db.fetch_customer(conn, seller_id, customer_id_local)
    if row is None or not _verify_password(password, row["password_salt"], row["password_hash"]):
        raise AccountError("아이디 또는 비밀번호가 올바르지 않습니다.")
    return row


# --- seller staff (business-registration gated) -----------------------------

def signup_seller(
    conn: sqlite3.Connection, *, seller_id: str, username: str, display_name: str, password: str,
    business_reg_no: str, business_open_date: str, business_rep_name: str,
) -> dict[str, Any]:
    if not username.strip() or not password:
        raise AccountError("아이디와 비밀번호를 입력하세요.")
    if len(password) < 4:
        raise AccountError("비밀번호는 4자 이상이어야 합니다.")
    if db.fetch_seller_account(conn, seller_id, username) is not None:
        raise AccountError("이미 사용 중인 아이디입니다.")

    result = nts_client.verify_business_registration(business_reg_no, business_open_date, business_rep_name)
    if not result.verified:
        raise AccountError("사업자등록 인증에 실패했습니다: %s" % result.detail)

    salt_hex, hash_hex = _hash_password(password)
    with conn:
        db.insert_seller_account(
            conn, seller_id, username, display_name or username, hash_hex, salt_hex,
            business_reg_no, result.verified, result.mode, _now_iso(),
        )
    return {
        "seller_id": seller_id, "username": username, "display_name": display_name,
        "business_verification_mode": result.mode, "business_verification_detail": result.detail,
    }


def authenticate_seller(conn: sqlite3.Connection, *, seller_id: str, username: str, password: str) -> dict[str, Any]:
    row = db.fetch_seller_account(conn, seller_id, username)
    if row is None or not _verify_password(password, row["password_salt"], row["password_hash"]):
        raise AccountError("아이디 또는 비밀번호가 올바르지 않습니다.")
    return row
