"""Restaurant POS tables in orders.sqlite (A-only, seller-local, no cross-role contract).

A table's open check is a ticket; its lines are what the guests ordered (sent to the kitchen
or not, served or not, voided after sending). Payments settle a ticket, possibly in several
parts (split bill). Members are the store's point-card customers, known only by a salted
hash of their phone number plus its last four digits. A day closing keeps the totals the
owner saw when closing.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pos_settings (
            seller_id TEXT PRIMARY KEY,
            tables INTEGER NOT NULL DEFAULT 12,
            point_rate INTEGER NOT NULL DEFAULT 5,
            open_hours TEXT,
            member_salt TEXT
        );
        CREATE TABLE IF NOT EXISTS pos_tickets (
            seller_id TEXT NOT NULL,
            ticket_id TEXT NOT NULL,
            table_no INTEGER,
            kind TEXT NOT NULL CHECK (kind IN ('dine_in', 'takeout')),
            guests INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL CHECK (status IN ('open', 'paid', 'void', 'refunded')),
            member_id TEXT,
            opened_at TEXT NOT NULL,
            paid_at TEXT,
            receipt_no TEXT,
            discount_minor INTEGER NOT NULL DEFAULT 0,
            points_used INTEGER NOT NULL DEFAULT 0,
            points_earned INTEGER NOT NULL DEFAULT 0,
            order_id TEXT,
            note TEXT,
            PRIMARY KEY (seller_id, ticket_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pos_tickets_open ON pos_tickets(seller_id, status, table_no);
        CREATE INDEX IF NOT EXISTS idx_pos_tickets_paid ON pos_tickets(seller_id, paid_at);
        CREATE TABLE IF NOT EXISTS pos_lines (
            seller_id TEXT NOT NULL,
            ticket_id TEXT NOT NULL,
            line_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            title TEXT NOT NULL,
            unit_price INTEGER NOT NULL,
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            memo TEXT,
            created_at TEXT NOT NULL,
            sent_at TEXT,
            served_at TEXT,
            voided_at TEXT,
            PRIMARY KEY (seller_id, ticket_id, line_id)
        );
        CREATE TABLE IF NOT EXISTS pos_payments (
            seller_id TEXT NOT NULL,
            payment_id TEXT NOT NULL,
            ticket_id TEXT NOT NULL,
            method TEXT NOT NULL CHECK (method IN ('card', 'cash', 'easy', 'transfer', 'points')),
            amount INTEGER NOT NULL CHECK (amount > 0),
            received INTEGER,
            change_due INTEGER,
            approval_no TEXT,
            paid_at TEXT NOT NULL,
            refunded_at TEXT,
            PRIMARY KEY (seller_id, payment_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pos_payments_ticket ON pos_payments(seller_id, ticket_id);
        CREATE TABLE IF NOT EXISTS pos_members (
            seller_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            phone_hash TEXT NOT NULL,
            last4 TEXT NOT NULL,
            joined_at TEXT NOT NULL,
            last_visit_at TEXT,
            visits INTEGER NOT NULL DEFAULT 0,
            points INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (seller_id, member_id),
            UNIQUE (seller_id, phone_hash)
        );
        CREATE TABLE IF NOT EXISTS pos_closings (
            seller_id TEXT NOT NULL,
            business_date TEXT NOT NULL,
            closed_at TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            PRIMARY KEY (seller_id, business_date)
        );
        """
    )


def row(conn, sql: str, args: tuple) -> Optional[dict[str, Any]]:
    r = conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def rows(conn, sql: str, args: tuple) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, args)]
