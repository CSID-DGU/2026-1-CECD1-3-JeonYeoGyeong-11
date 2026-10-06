"""Customer and seller-staff accounts -- local to this seller's orders.sqlite.

architecture.md §6: "고객 계정은 판매자 로컬이다" -- no central identity, no
cross-role contract. seller_accounts is the login for THIS store's own
dashboard (/seller/...), gated on NTS business-registration verification
(accounts_service.signup_seller); it is not multi-tenant platform signup --
provisioning a brand new seller's server is explicitly out of scope
(architecture.md §1: "서버의 조달·설치·운영 서비스는 이번 팀프로젝트의 확정 범위 밖이다").
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS customers (
            seller_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, customer_id_local)
        );

        CREATE TABLE IF NOT EXISTS seller_accounts (
            seller_id TEXT NOT NULL,
            username TEXT NOT NULL,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            business_reg_no TEXT NOT NULL,
            business_verified INTEGER NOT NULL DEFAULT 0,
            business_verification_mode TEXT NOT NULL DEFAULT 'mock',
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, username)
        );
        """
    )


def insert_customer(conn: sqlite3.Connection, seller_id: str, customer_id_local: str,
                     display_name: str, password_hash: str, password_salt: str, created_at: str) -> None:
    conn.execute(
        "INSERT INTO customers (seller_id, customer_id_local, display_name, password_hash, password_salt, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (seller_id, customer_id_local, display_name, password_hash, password_salt, created_at),
    )


def fetch_customer(conn: sqlite3.Connection, seller_id: str, customer_id_local: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM customers WHERE seller_id = ? AND customer_id_local = ?", (seller_id, customer_id_local)
    ).fetchone()
    return dict(row) if row else None


def list_customers(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    """IDs and display names only -- never the password columns."""
    rows = conn.execute(
        "SELECT customer_id_local, display_name FROM customers WHERE seller_id = ? ORDER BY customer_id_local",
        (seller_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def insert_seller_account(conn: sqlite3.Connection, seller_id: str, username: str, display_name: str,
                           password_hash: str, password_salt: str, business_reg_no: str,
                           business_verified: bool, business_verification_mode: str, created_at: str) -> None:
    conn.execute(
        "INSERT INTO seller_accounts (seller_id, username, display_name, password_hash, password_salt, "
        "business_reg_no, business_verified, business_verification_mode, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (seller_id, username, display_name, password_hash, password_salt, business_reg_no,
         1 if business_verified else 0, business_verification_mode, created_at),
    )


def fetch_seller_account(conn: sqlite3.Connection, seller_id: str, username: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM seller_accounts WHERE seller_id = ? AND username = ?", (seller_id, username)
    ).fetchone()
    return dict(row) if row else None
