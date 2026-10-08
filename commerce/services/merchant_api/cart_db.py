"""Buyer carts: rows in this seller's orders.sqlite (A-only, working-agreement.md §3).

A cart is screen state before an order exists, so it has no wire contract;
checkout turns it into one ordinary commerce_order (orders_service.checkout_cart).
"""
from __future__ import annotations

import sqlite3
from typing import Any


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cart_items (
            seller_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            added_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, customer_id_local, item_id_local)
        );
        """
    )


def add(conn: sqlite3.Connection, seller_id: str, customer_id_local: str, item_id_local: str,
        quantity: int, added_at: str) -> None:
    """Adding an item already in the cart adds to its quantity."""
    conn.execute(
        """
        INSERT INTO cart_items (seller_id, customer_id_local, item_id_local, quantity, added_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (seller_id, customer_id_local, item_id_local) DO UPDATE SET quantity = quantity + excluded.quantity
        """,
        (seller_id, customer_id_local, item_id_local, quantity, added_at),
    )


def set_quantity(conn: sqlite3.Connection, seller_id: str, customer_id_local: str, item_id_local: str,
                 quantity: int) -> None:
    conn.execute(
        "UPDATE cart_items SET quantity = ? WHERE seller_id = ? AND customer_id_local = ? AND item_id_local = ?",
        (quantity, seller_id, customer_id_local, item_id_local),
    )


def remove(conn: sqlite3.Connection, seller_id: str, customer_id_local: str, item_id_local: str) -> None:
    conn.execute(
        "DELETE FROM cart_items WHERE seller_id = ? AND customer_id_local = ? AND item_id_local = ?",
        (seller_id, customer_id_local, item_id_local),
    )


def clear(conn: sqlite3.Connection, seller_id: str, customer_id_local: str) -> None:
    conn.execute("DELETE FROM cart_items WHERE seller_id = ? AND customer_id_local = ?", (seller_id, customer_id_local))


def list_items(conn: sqlite3.Connection, seller_id: str, customer_id_local: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT item_id_local, quantity, added_at FROM cart_items WHERE seller_id = ? AND customer_id_local = ? "
        "ORDER BY added_at, rowid",
        (seller_id, customer_id_local),
    ).fetchall()
    return [dict(r) for r in rows]
