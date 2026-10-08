"""Layer 6 social/add-on tables (M25 DM, M26/M27 feed+shortform, M29 group-buy,
M30 price history) -- orders.sqlite, same connection as orders_db.py.

These are local-only screen features with no cross-role contract: no schema
under commerce/packages/contracts/ references them, and B/C never read this
file. Kept separate from orders_db.py so the order/catalog domain module stays
focused; both share the same sqlite connection orders_db.connect() opens.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS messages (
            seller_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            sender TEXT NOT NULL CHECK (sender IN ('customer', 'seller')),
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, message_id)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread
            ON messages(seller_id, customer_id_local, created_at);

        CREATE TABLE IF NOT EXISTS posts (
            seller_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('article', 'short_video')),
            title TEXT NOT NULL,
            body TEXT,
            media_path TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, post_id)
        );

        CREATE TABLE IF NOT EXISTS group_buys (
            seller_id TEXT NOT NULL,
            group_buy_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            target_quantity INTEGER NOT NULL,
            unit_price_minor INTEGER NOT NULL,
            deadline_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'succeeded', 'failed')),
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, group_buy_id)
        );

        CREATE TABLE IF NOT EXISTS group_buy_participants (
            seller_id TEXT NOT NULL,
            group_buy_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            joined_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, group_buy_id, customer_id_local)
        );

        CREATE TABLE IF NOT EXISTS price_history (
            seller_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            price_date TEXT NOT NULL,
            price_minor INTEGER NOT NULL,
            PRIMARY KEY (seller_id, item_id_local, price_date)
        );
        """
    )


# --- messages (M25) ---------------------------------------------------------

def insert_message(conn: sqlite3.Connection, seller_id: str, message_id: str,
                    customer_id_local: str, sender: str, body: str, created_at: str) -> None:
    conn.execute(
        "INSERT INTO messages (seller_id, message_id, customer_id_local, sender, body, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (seller_id, message_id, customer_id_local, sender, body, created_at),
    )


def list_thread_messages(conn: sqlite3.Connection, seller_id: str, customer_id_local: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE seller_id = ? AND customer_id_local = ? ORDER BY created_at",
        (seller_id, customer_id_local),
    ).fetchall()
    return [dict(r) for r in rows]


def list_threads(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    """One row per customer thread: last message + when, newest first.

    Picks by (created_at, rowid) rather than created_at alone: two messages to
    the same customer can land on the same clock tick, and rowid (insertion
    order) is the only thing that disambiguates which one is actually last."""
    rows = conn.execute(
        """
        SELECT customer_id_local, body AS last_body, sender AS last_sender, created_at AS last_at
        FROM messages m
        WHERE seller_id = ? AND rowid = (
            SELECT rowid FROM messages
            WHERE seller_id = m.seller_id AND customer_id_local = m.customer_id_local
            ORDER BY created_at DESC, rowid DESC LIMIT 1
        )
        ORDER BY last_at DESC
        """,
        (seller_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# --- posts / feed (M26, M27) -------------------------------------------------

def insert_post(conn: sqlite3.Connection, seller_id: str, post_id: str, kind: str,
                 title: str, body: Optional[str], media_path: Optional[str], created_at: str) -> None:
    conn.execute(
        "INSERT INTO posts (seller_id, post_id, kind, title, body, media_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (seller_id, post_id, kind, title, body, media_path, created_at),
    )


def list_posts(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    # rowid tiebreak: two posts can land on the same created_at tick (clock
    # resolution), and insertion order should still decide which is "newest".
    rows = conn.execute(
        "SELECT * FROM posts WHERE seller_id = ? ORDER BY created_at DESC, rowid DESC", (seller_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- group buys (M29) --------------------------------------------------------

def insert_group_buy(conn: sqlite3.Connection, seller_id: str, group_buy_id: str, item_id_local: str,
                      target_quantity: int, unit_price_minor: int, deadline_at: str, created_at: str) -> None:
    conn.execute(
        "INSERT INTO group_buys (seller_id, group_buy_id, item_id_local, target_quantity, "
        "unit_price_minor, deadline_at, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
        (seller_id, group_buy_id, item_id_local, target_quantity, unit_price_minor, deadline_at, created_at),
    )


def fetch_group_buy(conn: sqlite3.Connection, seller_id: str, group_buy_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM group_buys WHERE seller_id = ? AND group_buy_id = ?", (seller_id, group_buy_id)
    ).fetchone()
    return dict(row) if row else None


def list_group_buys(conn: sqlite3.Connection, seller_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM group_buys WHERE seller_id = ? ORDER BY deadline_at", (seller_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def update_group_buy_status(conn: sqlite3.Connection, seller_id: str, group_buy_id: str, status: str) -> None:
    conn.execute(
        "UPDATE group_buys SET status = ? WHERE seller_id = ? AND group_buy_id = ?",
        (status, seller_id, group_buy_id),
    )


def insert_group_buy_participant(conn: sqlite3.Connection, seller_id: str, group_buy_id: str,
                                  customer_id_local: str, quantity: int, joined_at: str) -> bool:
    """False if this customer already joined this group buy (primary key conflict)."""
    try:
        conn.execute(
            "INSERT INTO group_buy_participants (seller_id, group_buy_id, customer_id_local, quantity, joined_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (seller_id, group_buy_id, customer_id_local, quantity, joined_at),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def list_group_buy_participants(conn: sqlite3.Connection, seller_id: str, group_buy_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM group_buy_participants WHERE seller_id = ? AND group_buy_id = ?",
        (seller_id, group_buy_id),
    ).fetchall()
    return [dict(r) for r in rows]


def sum_group_buy_quantity(conn: sqlite3.Connection, seller_id: str, group_buy_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(quantity), 0) AS total FROM group_buy_participants "
        "WHERE seller_id = ? AND group_buy_id = ?",
        (seller_id, group_buy_id),
    ).fetchone()
    return row["total"]


# --- price history (M30) -----------------------------------------------------

def upsert_price(conn: sqlite3.Connection, seller_id: str, item_id_local: str, price_date: str, price_minor: int) -> None:
    conn.execute(
        """
        INSERT INTO price_history (seller_id, item_id_local, price_date, price_minor)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(seller_id, item_id_local, price_date) DO UPDATE SET price_minor = excluded.price_minor
        """,
        (seller_id, item_id_local, price_date, price_minor),
    )


def list_price_history(conn: sqlite3.Connection, seller_id: str, item_id_local: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM price_history WHERE seller_id = ? AND item_id_local = ? ORDER BY price_date",
        (seller_id, item_id_local),
    ).fetchall()
    return [dict(r) for r in rows]
