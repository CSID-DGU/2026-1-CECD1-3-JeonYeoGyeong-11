"""Reviews and wishlists in orders.sqlite (A-only, seller-local, no cross-role contract)."""
from __future__ import annotations

import sqlite3
from typing import Any


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            seller_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            body TEXT,
            seller_reply TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, item_id_local, customer_id_local)
        );
        CREATE TABLE IF NOT EXISTS wishlist (
            seller_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, customer_id_local, item_id_local)
        );
        """
    )


def upsert_review(conn, seller_id: str, item_id: str, customer_id: str, rating: int, body, now: str) -> None:
    conn.execute(
        """
        INSERT INTO reviews (seller_id, item_id_local, customer_id_local, rating, body, created_at) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(seller_id, item_id_local, customer_id_local)
        DO UPDATE SET rating = excluded.rating, body = excluded.body, created_at = excluded.created_at
        """,
        (seller_id, item_id, customer_id, rating, body, now),
    )


def reply_review(conn, seller_id: str, item_id: str, customer_id: str, reply) -> bool:
    cur = conn.execute("UPDATE reviews SET seller_reply = ? WHERE seller_id = ? AND item_id_local = ? AND customer_id_local = ?",
                       (reply, seller_id, item_id, customer_id))
    return cur.rowcount == 1


def reviews_for_item(conn, seller_id: str, item_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM reviews WHERE seller_id = ? AND item_id_local = ? ORDER BY created_at DESC", (seller_id, item_id))]


def all_reviews(conn, seller_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM reviews WHERE seller_id = ? ORDER BY created_at DESC", (seller_id,))]


def rating_summary(conn, seller_id: str) -> dict[str, tuple[float, int]]:
    """item -> (average rating, count)."""
    return {r["item_id_local"]: (round(r["avg"], 1), r["n"]) for r in conn.execute(
        "SELECT item_id_local, AVG(rating) AS avg, COUNT(*) AS n FROM reviews WHERE seller_id = ? GROUP BY item_id_local",
        (seller_id,))}


def toggle_wish(conn, seller_id: str, customer_id: str, item_id: str, now: str) -> bool:
    cur = conn.execute("DELETE FROM wishlist WHERE seller_id = ? AND customer_id_local = ? AND item_id_local = ?",
                       (seller_id, customer_id, item_id))
    if cur.rowcount:
        return False
    conn.execute("INSERT INTO wishlist (seller_id, customer_id_local, item_id_local, created_at) VALUES (?, ?, ?, ?)",
                 (seller_id, customer_id, item_id, now))
    return True


def wishlist(conn, seller_id: str, customer_id: str) -> list[str]:
    return [r["item_id_local"] for r in conn.execute(
        "SELECT item_id_local FROM wishlist WHERE seller_id = ? AND customer_id_local = ? ORDER BY created_at DESC",
        (seller_id, customer_id))]


def wish_counts(conn, seller_id: str) -> dict[str, int]:
    return {r["item_id_local"]: r["n"] for r in conn.execute(
        "SELECT item_id_local, COUNT(*) AS n FROM wishlist WHERE seller_id = ? GROUP BY item_id_local", (seller_id,))}
