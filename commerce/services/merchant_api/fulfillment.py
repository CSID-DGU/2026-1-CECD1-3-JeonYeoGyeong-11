"""How an order reaches the buyer, the stock that limits it, and when it moved.

Direct trade means the buyer either picks the order up at the store or has it
sent. Each screen order gets one fulfillment row (method, recipient, contact,
address or pickup slot, memo, shipping fee, tracking number); a buyer's last
details are kept as their profile so the next checkout is pre-filled. Stock is
optional per item (NULL = not tracked): orders reserve it, cancels give it back.
The status log keeps when an order was accepted, which the orders table does
not hold, for the order timeline.

All of this is A-local in orders.sqlite. None of it goes to B: purchase_event.v1
carries items and time only, so contact details and addresses never leave the
seller's store (the FL premise).
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
from typing import Any, Iterable, Optional

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api.migrate import add_columns

METHODS = {"pickup": "매장 픽업", "delivery": "택배"}
PICKUP_SLOTS = ("오늘 17:00~19:00", "내일 10:00~12:00", "내일 14:00~16:00", "내일 17:00~19:00", "모레 10:00~12:00")
DEFAULT_DELIVERY_FEE = 3000
DEFAULT_FREE_OVER = 30000
_PHONE = re.compile(r"^0\d{1,2}-?\d{3,4}-?\d{4}$")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS order_fulfillment (
            seller_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            method TEXT NOT NULL CHECK (method IN ('pickup', 'delivery')),
            recipient TEXT,
            phone TEXT,
            address TEXT,
            pickup_slot TEXT,
            memo TEXT,
            shipping_fee INTEGER NOT NULL DEFAULT 0,
            tracking_no TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, order_id)
        );
        CREATE TABLE IF NOT EXISTS customer_profiles (
            seller_id TEXT NOT NULL,
            customer_id_local TEXT NOT NULL,
            method TEXT,
            recipient TEXT,
            phone TEXT,
            address TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (seller_id, customer_id_local)
        );
        CREATE TABLE IF NOT EXISTS item_stock (
            seller_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            stock INTEGER CHECK (stock IS NULL OR stock >= 0),
            PRIMARY KEY (seller_id, item_id_local)
        );
        CREATE TABLE IF NOT EXISTS order_status_log (
            seller_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            status TEXT NOT NULL,
            at TEXT NOT NULL,
            PRIMARY KEY (seller_id, order_id, status)
        );
        CREATE TABLE IF NOT EXISTS order_stock_holds (
            seller_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            item_id_local TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            PRIMARY KEY (seller_id, order_id, item_id_local)
        );
        """
    )
    # Pickup place, hours and delivery pricing are the seller's store settings.
    conn.execute("CREATE TABLE IF NOT EXISTS store_settings (seller_id TEXT PRIMARY KEY, "
                 "group_discount_pct INTEGER NOT NULL DEFAULT 10, group_min_target INTEGER NOT NULL DEFAULT 3, store_intro TEXT)")
    add_columns(conn, "store_settings", {"pickup_address": "TEXT", "pickup_hours": "TEXT",
                                         "delivery_fee": "INTEGER", "free_delivery_over": "INTEGER"})


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- store delivery settings ------------------------------------------------------

def store_terms(conn, seller_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT pickup_address, pickup_hours, delivery_fee, free_delivery_over FROM store_settings "
                       "WHERE seller_id = ?", (seller_id,)).fetchone()
    row = dict(row) if row else {}
    return {"pickup_address": row.get("pickup_address") or "매장 앞 (판매자에게 쪽지로 위치 확인)",
            "pickup_hours": row.get("pickup_hours") or "매일 10:00~19:00",
            "delivery_fee": DEFAULT_DELIVERY_FEE if row.get("delivery_fee") is None else row["delivery_fee"],
            "free_delivery_over": DEFAULT_FREE_OVER if row.get("free_delivery_over") is None else row["free_delivery_over"]}


def save_store_terms(conn, seller_id: str, *, pickup_address: str, pickup_hours: str, delivery_fee: int,
                     free_delivery_over: int) -> None:
    if not 0 <= delivery_fee <= 50000 or not 0 <= free_delivery_over <= 1_000_000:
        raise ContractError("INVALID_TYPE", "/delivery_fee")
    with conn:
        conn.execute("INSERT OR IGNORE INTO store_settings (seller_id) VALUES (?)", (seller_id,))
        conn.execute("UPDATE store_settings SET pickup_address = ?, pickup_hours = ?, delivery_fee = ?, free_delivery_over = ? "
                     "WHERE seller_id = ?", ((pickup_address or "").strip()[:200] or None, (pickup_hours or "").strip()[:100] or None,
                                             delivery_fee, free_delivery_over, seller_id))


def shipping_fee(terms: dict[str, Any], method: str, subtotal: int) -> int:
    if method != "delivery":
        return 0
    return 0 if terms["free_delivery_over"] and subtotal >= terms["free_delivery_over"] else terms["delivery_fee"]


# --- checkout details ---------------------------------------------------------------

def profile(conn, seller_id: str, customer_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM customer_profiles WHERE seller_id = ? AND customer_id_local = ?",
                       (seller_id, customer_id)).fetchone()
    return dict(row) if row else {"method": "pickup", "recipient": None, "phone": None, "address": None}


def validate(method: str, recipient: str, phone: str, address: str, pickup_slot: str, memo: str) -> dict[str, Any]:
    """Clean checkout input or a ContractError naming the field (shown next to the form)."""
    method = method or "pickup"
    if method not in METHODS:
        raise ContractError("INVALID_ENUM_VALUE", "/method")
    recipient, phone, address = (recipient or "").strip(), (phone or "").strip(), (address or "").strip()
    out = {"method": method, "recipient": recipient[:40] or None, "phone": phone[:20] or None,
           "address": None, "pickup_slot": None, "memo": (memo or "").strip()[:200] or None}
    if phone and not _PHONE.match(phone):
        raise ContractError("SCHEMA_INVALID", "/phone")
    if method == "delivery":
        if not recipient:
            raise ContractError("MISSING_REQUIRED_FIELD", "/recipient")
        if not phone:
            raise ContractError("MISSING_REQUIRED_FIELD", "/phone")
        if len(address) < 5:
            raise ContractError("MISSING_REQUIRED_FIELD", "/address")
        out["address"] = address[:200]
    else:
        out["pickup_slot"] = pickup_slot if pickup_slot in PICKUP_SLOTS else PICKUP_SLOTS[0]
    return out


def record(conn, seller_id: str, order: dict[str, Any], customer_id: str, details: dict[str, Any], fee: int) -> None:
    """The order's fulfillment and the buyer's profile (a retried checkout keeps the first row)."""
    now = _now()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO order_fulfillment (seller_id, order_id, method, recipient, phone, address, pickup_slot, "
            "memo, shipping_fee, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (seller_id, order["order_id"], details["method"], details["recipient"], details["phone"], details["address"],
             details["pickup_slot"], details["memo"], fee, now))
        conn.execute(
            """INSERT INTO customer_profiles (seller_id, customer_id_local, method, recipient, phone, address, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(seller_id, customer_id_local) DO UPDATE SET method = excluded.method,
                   recipient = COALESCE(excluded.recipient, recipient), phone = COALESCE(excluded.phone, phone),
                   address = COALESCE(excluded.address, address), updated_at = excluded.updated_at""",
            (seller_id, customer_id, details["method"], details["recipient"], details["phone"], details["address"], now))
        log_status(conn, seller_id, order["order_id"], "requested", order.get("created_at") or now)


def of_order(conn, seller_id: str, order_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM order_fulfillment WHERE seller_id = ? AND order_id = ?", (seller_id, order_id)).fetchone()
    return dict(row) if row else None


def of_orders(conn, seller_id: str) -> dict[str, dict[str, Any]]:
    return {r["order_id"]: dict(r) for r in conn.execute("SELECT * FROM order_fulfillment WHERE seller_id = ?", (seller_id,))}


def set_tracking(conn, seller_id: str, order_id: str, tracking_no: str) -> None:
    tracking_no = re.sub(r"[^0-9A-Za-z-]", "", tracking_no or "")[:30]
    with conn:
        cur = conn.execute("UPDATE order_fulfillment SET tracking_no = ? WHERE seller_id = ? AND order_id = ? AND method = 'delivery'",
                           (tracking_no or None, seller_id, order_id))
    if cur.rowcount != 1:
        raise ContractError("NOT_FOUND", "/order_id")


def masked_phone(phone: Optional[str]) -> str:
    digits = re.sub(r"\D", "", phone or "")
    return "%s-****-%s" % (digits[:3], digits[-4:]) if len(digits) >= 9 else (phone or "")


# --- status log -------------------------------------------------------------------------

def log_status(conn, seller_id: str, order_id: str, status: str, at: Optional[str] = None) -> None:
    conn.execute("INSERT OR IGNORE INTO order_status_log (seller_id, order_id, status, at) VALUES (?, ?, ?, ?)",
                 (seller_id, order_id, status, at or _now()))


def timeline(conn, seller_id: str, order: dict[str, Any]) -> list[dict[str, Any]]:
    """requested -> accepted -> completed (or cancelled) with the times known."""
    logged = {r["status"]: r["at"] for r in conn.execute(
        "SELECT status, at FROM order_status_log WHERE seller_id = ? AND order_id = ?", (seller_id, order["order_id"]))}
    known = {"requested": order.get("created_at"), "accepted": logged.get("accepted"),
             "completed": order.get("completed_at"), "cancelled": order.get("cancelled_at")}
    labels = {"requested": "주문 접수", "accepted": "판매자 확인·준비 중", "completed": "전달 완료", "cancelled": "주문 취소"}
    if order["status"] == "cancelled":
        steps = ["requested", "accepted", "cancelled"] if known["accepted"] else ["requested", "cancelled"]
    else:
        steps = ["requested", "accepted", "completed"]
    current = steps.index(order["status"])
    return [{"status": step, "label": labels[step], "at": known.get(step), "done": n <= current, "current": n == current}
            for n, step in enumerate(steps)]


# --- stock ----------------------------------------------------------------------------

def stock_levels(conn, seller_id: str) -> dict[str, Optional[int]]:
    return {r["item_id_local"]: r["stock"] for r in conn.execute(
        "SELECT item_id_local, stock FROM item_stock WHERE seller_id = ?", (seller_id,))}


def set_stock(conn, seller_id: str, item_id: str, stock: Optional[int]) -> None:
    if stock is not None and not 0 <= stock <= 100000:
        raise ContractError("INVALID_TYPE", "/stock")
    with conn:
        conn.execute("INSERT INTO item_stock (seller_id, item_id_local, stock) VALUES (?, ?, ?) "
                     "ON CONFLICT(seller_id, item_id_local) DO UPDATE SET stock = excluded.stock", (seller_id, item_id, stock))


def check_stock(conn, seller_id: str, lines: Iterable[tuple[str, int]]) -> None:
    """INSUFFICIENT: ContractError('INVALID_TYPE', '/stock/<item>') before any order is made."""
    levels = stock_levels(conn, seller_id)
    for item, quantity in lines:
        left = levels.get(item)
        if left is not None and quantity > left:
            raise ContractError("INVALID_TYPE", "/stock/%s" % item)


def hold_stock(conn, seller_id: str, order: dict[str, Any]) -> None:
    """Take the order's quantities off tracked stock once (a retried checkout holds nothing more)."""
    levels = stock_levels(conn, seller_id)
    with conn:
        for item in order["items"]:
            if levels.get(item["item_id_local"]) is None:
                continue
            cur = conn.execute("INSERT OR IGNORE INTO order_stock_holds (seller_id, order_id, item_id_local, quantity) "
                               "VALUES (?, ?, ?, ?)", (seller_id, order["order_id"], item["item_id_local"], item["quantity"]))
            if cur.rowcount:
                conn.execute("UPDATE item_stock SET stock = MAX(0, stock - ?) WHERE seller_id = ? AND item_id_local = ?",
                             (item["quantity"], seller_id, item["item_id_local"]))


def release_stock(conn, seller_id: str, order_id: str) -> None:
    """A cancelled order gives its held stock back, once."""
    with conn:
        for hold in conn.execute("SELECT item_id_local, quantity FROM order_stock_holds WHERE seller_id = ? AND order_id = ?",
                                 (seller_id, order_id)).fetchall():
            conn.execute("UPDATE item_stock SET stock = stock + ? WHERE seller_id = ? AND item_id_local = ? AND stock IS NOT NULL",
                         (hold["quantity"], seller_id, hold["item_id_local"]))
        conn.execute("DELETE FROM order_stock_holds WHERE seller_id = ? AND order_id = ?", (seller_id, order_id))
