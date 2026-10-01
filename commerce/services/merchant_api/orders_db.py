"""orders.sqlite schema and row-level access. A owns this file exclusively.

Only this module touches orders.sqlite directly. orders_service.py composes
these calls into transactions; it never runs raw SQL itself, so the durability
rules in docs/design/interfaces.md §2 have one place to change.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Default isolation_level ("") keeps sqlite3's implicit transaction handling,
    # so `with conn:` in orders_service.py commits/rolls back atomically (§2 durable write).
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS orders (
            seller_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            status_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            cancelled_at TEXT,
            currency TEXT NOT NULL,
            items_json TEXT NOT NULL,
            PRIMARY KEY (seller_id, order_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_idempotency
            ON orders(seller_id, idempotency_key);

        CREATE TABLE IF NOT EXISTS purchase_events (
            seller_id TEXT NOT NULL,
            purchase_event_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, purchase_event_id)
        );

        CREATE TABLE IF NOT EXISTS catalog_items (
            seller_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            source_seq INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            display_price_minor INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (seller_id, item_id_local)
        );

        CREATE TABLE IF NOT EXISTS outbox (
            seller_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            source_seq INTEGER,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, kind, ref_id)
        );
        """
    )


def _row_to_order(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "seller_id": row["seller_id"],
        "order_id": row["order_id"],
        "customer_id_local": row["customer_id_local"],
        "idempotency_key": row["idempotency_key"],
        "request_hash": row["request_hash"],
        "status": row["status"],
        "status_version": row["status_version"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "cancelled_at": row["cancelled_at"],
        "currency": row["currency"],
        "items": json.loads(row["items_json"]),
    }


def fetch_order(conn: sqlite3.Connection, seller_id: str, order_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM orders WHERE seller_id = ? AND order_id = ?", (seller_id, order_id)
    ).fetchone()
    return _row_to_order(row) if row else None


def list_orders(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM orders WHERE seller_id = ? ORDER BY created_at DESC", (seller_id,)
    ).fetchall()
    return [_row_to_order(r) for r in rows]


def list_orders_by_customer(conn: sqlite3.Connection, seller_id: str, customer_id_local: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM orders WHERE seller_id = ? AND customer_id_local = ? ORDER BY created_at DESC",
        (seller_id, customer_id_local),
    ).fetchall()
    return [_row_to_order(r) for r in rows]


def fetch_order_by_idempotency_key(
    conn: sqlite3.Connection, seller_id: str, idempotency_key: str
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM orders WHERE seller_id = ? AND idempotency_key = ?",
        (seller_id, idempotency_key),
    ).fetchone()
    return _row_to_order(row) if row else None


def insert_order(conn: sqlite3.Connection, order: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO orders (
            seller_id, order_id, customer_id_local, idempotency_key, request_hash,
            status, status_version, created_at, completed_at, cancelled_at, currency, items_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order["seller_id"], order["order_id"], order["customer_id_local"],
            order["idempotency_key"], order["request_hash"], order["status"],
            order["status_version"], order["created_at"], order["completed_at"],
            order["cancelled_at"], order["currency"], json.dumps(order["items"]),
        ),
    )


def conditional_update_status(
    conn: sqlite3.Connection, seller_id: str, order_id: str, expected_status_version: int,
    new_status: str, new_status_version: int, *, completed_at: Optional[str], cancelled_at: Optional[str],
) -> bool:
    """Atomic guard for concurrent transitions (interfaces.md §2: only one wins)."""
    cur = conn.execute(
        """
        UPDATE orders SET status = ?, status_version = ?, completed_at = ?, cancelled_at = ?
        WHERE seller_id = ? AND order_id = ? AND status_version = ?
        """,
        (new_status, new_status_version, completed_at, cancelled_at, seller_id, order_id, expected_status_version),
    )
    return cur.rowcount == 1


def insert_purchase_event(conn: sqlite3.Connection, seller_id: str, order_id: str, event: dict[str, Any], created_at: str) -> None:
    conn.execute(
        "INSERT INTO purchase_events (seller_id, purchase_event_id, order_id, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (seller_id, event["purchase_event_id"], order_id, json.dumps(event), created_at),
    )


def fetch_catalog_item(conn: sqlite3.Connection, seller_id: str, item_id_local: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM catalog_items WHERE seller_id = ? AND item_id_local = ?",
        (seller_id, item_id_local),
    ).fetchone()
    if row is None:
        return None
    return {
        "source_seq": row["source_seq"], "payload": json.loads(row["payload_json"]),
        "display_price_minor": row["display_price_minor"],
    }


def upsert_catalog_item(conn: sqlite3.Connection, seller_id: str, item_id_local: str, source_seq: int,
                         payload: dict[str, Any], display_price_minor: int) -> None:
    """display_price_minor is A's own local display convenience, never part of the
    catalog_item.v1 payload sent to B (that contract has no price field)."""
    conn.execute(
        """
        INSERT INTO catalog_items (seller_id, item_id_local, source_seq, payload_json, display_price_minor)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(seller_id, item_id_local) DO UPDATE SET
            source_seq = excluded.source_seq, payload_json = excluded.payload_json,
            display_price_minor = excluded.display_price_minor
        """,
        (seller_id, item_id_local, source_seq, json.dumps(payload), display_price_minor),
    )


def list_catalog_items(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM catalog_items WHERE seller_id = ? ORDER BY item_id_local", (seller_id,)
    ).fetchall()
    return [
        {**json.loads(r["payload_json"]), "display_price_minor": r["display_price_minor"]}
        for r in rows
    ]


def insert_outbox(conn: sqlite3.Connection, seller_id: str, kind: str, ref_id: str,
                   payload: dict[str, Any], created_at: str, source_seq: Optional[int] = None) -> None:
    conn.execute(
        """
        INSERT INTO outbox (seller_id, kind, ref_id, source_seq, payload_json, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?)
        ON CONFLICT(seller_id, kind, ref_id) DO UPDATE SET
            source_seq = excluded.source_seq, payload_json = excluded.payload_json,
            status = 'pending'
        """,
        (seller_id, kind, ref_id, source_seq, json.dumps(payload), created_at),
    )


def fetch_pending_outbox(conn: sqlite3.Connection, seller_id: str, kind: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM outbox WHERE seller_id = ? AND kind = ? AND status = 'pending' ORDER BY created_at",
        (seller_id, kind),
    ).fetchall()
    return [
        {"ref_id": r["ref_id"], "source_seq": r["source_seq"], "payload": json.loads(r["payload_json"]),
         "attempts": r["attempts"]}
        for r in rows
    ]


def mark_outbox_delivered(conn: sqlite3.Connection, seller_id: str, kind: str, ref_id: str) -> None:
    conn.execute(
        "UPDATE outbox SET status = 'delivered', last_error = NULL WHERE seller_id = ? AND kind = ? AND ref_id = ?",
        (seller_id, kind, ref_id),
    )


def mark_outbox_attempt(conn: sqlite3.Connection, seller_id: str, kind: str, ref_id: str, error: str) -> None:
    conn.execute(
        "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE seller_id = ? AND kind = ? AND ref_id = ?",
        (error, seller_id, kind, ref_id),
    )


def mark_outbox_quarantined(conn: sqlite3.Connection, seller_id: str, kind: str, ref_id: str, error: str) -> None:
    conn.execute(
        "UPDATE outbox SET status = 'quarantined', last_error = ? WHERE seller_id = ? AND kind = ? AND ref_id = ?",
        (error, seller_id, kind, ref_id),
    )
