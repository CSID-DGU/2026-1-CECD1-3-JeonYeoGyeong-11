"""Buyer notifications and store follows (단골), A-local in orders.sqlite.

A notification is one line with a link: an order the seller accepted, completed or
cancelled, a group buy that reached its target, the seller's reply to a review or
a comment, a DM, or a new post from a store the buyer follows. The bell in the
buyer header shows the unread count; opening the list marks them read.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from typing import Any, Iterable, Optional

KIND_ICONS = {"order": "🧾", "group_buy": "👥", "reply": "💬", "message": "✉️", "post": "📣"}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            seller_id TEXT NOT NULL,
            notification_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT,
            dedupe_key TEXT,
            created_at TEXT NOT NULL,
            read_at TEXT,
            PRIMARY KEY (seller_id, notification_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_notifications_dedupe
            ON notifications(seller_id, customer_id_local, dedupe_key) WHERE dedupe_key IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_notifications_customer
            ON notifications(seller_id, customer_id_local, created_at);
        CREATE TABLE IF NOT EXISTS store_follows (
            seller_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, customer_id_local)
        );
        """
    )


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def notify(conn, seller_id: str, customer_ids: Iterable[str], kind: str, title: str, link: Optional[str] = None,
           dedupe_key: Optional[str] = None, at: Optional[str] = None) -> int:
    """One notification per customer; the same dedupe_key for a customer is stored once."""
    n = 0
    with conn:
        for customer_id in dict.fromkeys(customer_ids):
            if not customer_id:
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO notifications (seller_id, notification_id, customer_id_local, kind, title, link, "
                "dedupe_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seller_id, uuid.uuid4().hex, customer_id, kind, title[:200], link, dedupe_key, at or _now()))
            n += cur.rowcount
    return n


def unread_count(conn, seller_id: str, customer_id: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM notifications WHERE seller_id = ? AND customer_id_local = ? AND read_at IS NULL",
                        (seller_id, customer_id)).fetchone()[0]


def recent(conn, seller_id: str, customer_id: str, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM notifications WHERE seller_id = ? AND customer_id_local = ? "
                        "ORDER BY created_at DESC, rowid DESC LIMIT ?", (seller_id, customer_id, limit)).fetchall()
    return [dict(r, icon=KIND_ICONS.get(r["kind"], "🔔")) for r in rows]


def mark_all_read(conn, seller_id: str, customer_id: str) -> None:
    with conn:
        conn.execute("UPDATE notifications SET read_at = ? WHERE seller_id = ? AND customer_id_local = ? AND read_at IS NULL",
                     (_now(), seller_id, customer_id))


# --- follows ------------------------------------------------------------------------------

def toggle_follow(conn, seller_id: str, customer_id: str) -> bool:
    """True if the customer follows the store after the call."""
    with conn:
        cur = conn.execute("DELETE FROM store_follows WHERE seller_id = ? AND customer_id_local = ?", (seller_id, customer_id))
        if cur.rowcount:
            return False
        conn.execute("INSERT INTO store_follows (seller_id, customer_id_local, created_at) VALUES (?, ?, ?)",
                     (seller_id, customer_id, _now()))
        return True


def is_following(conn, seller_id: str, customer_id: Optional[str]) -> bool:
    if not customer_id:
        return False
    return conn.execute("SELECT 1 FROM store_follows WHERE seller_id = ? AND customer_id_local = ?",
                        (seller_id, customer_id)).fetchone() is not None


def followers(conn, seller_id: str) -> list[str]:
    return [r[0] for r in conn.execute("SELECT customer_id_local FROM store_follows WHERE seller_id = ?", (seller_id,))]


# --- the events that notify ---------------------------------------------------------------

ORDER_TITLES = {"accept": "판매자가 주문을 확인했어요. 준비가 시작됐어요.",
                "complete": "주문이 전달 완료됐어요. 리뷰를 남겨 주세요!",
                "cancel": "판매자가 주문을 취소했어요."}


def on_order(conn, seller_id: str, order: dict[str, Any], action: str) -> None:
    if action in ORDER_TITLES:
        notify(conn, seller_id, [order["customer_id_local"]], "order", ORDER_TITLES[action],
               "/buyer/%s/orders/%s" % (seller_id, order["order_id"]), dedupe_key="order:%s:%s" % (order["order_id"], action))


def on_group_buy_succeeded(conn, seller_id: str, group_buy_id: str, title: str, customer_ids: Iterable[str]) -> None:
    notify(conn, seller_id, customer_ids, "group_buy", "공동구매 '%s' 목표 달성! 주문이 만들어졌어요." % title,
           "/buyer/%s/orders" % seller_id, dedupe_key="gb:%s" % group_buy_id)
