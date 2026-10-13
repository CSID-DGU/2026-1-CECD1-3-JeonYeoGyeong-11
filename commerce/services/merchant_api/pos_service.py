"""A restaurant POS on top of A's order path.

The counter works like a real POS: pick a table (or takeout), take the order, send it to the
kitchen, add more later, look the guest up by phone number for points, settle the bill in one
or several payments (card, cash with change, easy pay, transfer, points), print a receipt.
Settling a ticket is the moment the purchase happens: the ticket's lines become one ordinary
A order that is accepted and completed at once, so the purchase_event reaches B through the
same outbox as every other order (interfaces.md §2). Nothing else about the ticket goes to B.

Members: a phone number is never stored. It is hashed with a per-store salt; the member id
(also the customer_id_local B sees) is derived from that hash, and only the last four digits
are kept to show "****-1234" on screen. A guest who does not give a number is a walk-in:
each walk-in ticket gets its own customer id, so B learns from the basket but never links
two walk-in visits.

Refunds: a refunded ticket keeps its completed order, because purchase_event.v1 has no
retraction (a team decision if one is wanted); the screen says so.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import secrets
import uuid
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import fulfillment, orders_db, orders_service, pos_layout
from commerce.services.merchant_api.pos_db import row, rows

KST = dt.timezone(dt.timedelta(hours=9))
METHODS = {"card": "카드", "cash": "현금", "easy": "간편결제", "transfer": "계좌이체", "points": "포인트"}
KINDS = {"dine_in": "매장", "takeout": "포장"}
SERVICE_MODES = {"postpay": "후불 (식사 후 결제)", "prepay": "선불 (주문할 때 결제)"}
_NS = uuid.UUID("2b1f7c61-8a1e-5d4b-9c3e-6f7a8b9c0d1e")
_PHONE = re.compile(r"^01[016789]\d{7,8}$")
MEMOS = ("덜 맵게", "맵게", "곱빼기", "양 적게", "얼음 적게", "따로 포장", "빨리 부탁")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def business_date(ts: str) -> str:
    return _parse(ts).astimezone(KST).date().isoformat()


def _id(*parts: str) -> str:
    return uuid.uuid5(_NS, "/".join(parts)).hex


# --- settings ----------------------------------------------------------------------

def settings(conn, seller_id: str) -> dict[str, Any]:
    s = row(conn, "SELECT * FROM pos_settings WHERE seller_id = ?", (seller_id,))
    if s is None or not s["member_salt"]:
        with conn:
            conn.execute("INSERT OR IGNORE INTO pos_settings (seller_id) VALUES (?)", (seller_id,))
            conn.execute("UPDATE pos_settings SET member_salt = ? WHERE seller_id = ? AND member_salt IS NULL",
                         (secrets.token_hex(16), seller_id))
        s = row(conn, "SELECT * FROM pos_settings WHERE seller_id = ?", (seller_id,))
    s["service_mode_label"] = SERVICE_MODES.get(s["service_mode"], s["service_mode"])
    return s


def save_settings(conn, seller_id: str, *, point_rate: int, open_hours: str, service_mode: str = "postpay",
                  takeout_enabled: bool = True, time_limit: int = 0) -> None:
    """How this store runs its counter (the floor plan itself is pos_layout)."""
    if not 0 <= point_rate <= 20:
        raise ContractError("INVALID_TYPE", "/point_rate")
    if service_mode not in SERVICE_MODES:
        raise ContractError("INVALID_ENUM_VALUE", "/service_mode")
    if not 0 <= time_limit <= 600:
        raise ContractError("INVALID_TYPE", "/time_limit")
    settings(conn, seller_id)
    with conn:
        conn.execute("UPDATE pos_settings SET point_rate = ?, open_hours = ?, service_mode = ?, takeout_enabled = ?, time_limit = ? "
                     "WHERE seller_id = ?", (point_rate, (open_hours or "").strip()[:60] or None, service_mode,
                                             1 if takeout_enabled else 0, int(time_limit), seller_id))


# --- menu ----------------------------------------------------------------------------

def menu(conn, seller_id: str) -> list[dict[str, Any]]:
    """Active menu items with their section (category_path's last part), price and sold-out flag."""
    stock = fulfillment.stock_levels(conn, seller_id)
    out = []
    for item in orders_service.list_catalog_for_display(conn, seller_id=seller_id):
        if item["listing_status"] != "active":
            continue
        path = item.get("category_path") or ["기타"]
        out.append({"item_id": item["item_id_local"], "title": item["title_text"], "price": item["display_price_minor"],
                    "section": path[-1], "description": item.get("description_text"),
                    "category_path": path, "sold_out": stock.get(item["item_id_local"]) == 0,
                    "stock": stock.get(item["item_id_local"])})
    return out


def sections(items: Iterable[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        seen.setdefault(i["section"], None)
    return list(seen)


# --- tickets -------------------------------------------------------------------------

def _ticket(conn, seller_id: str, ticket_id: str) -> dict[str, Any]:
    t = row(conn, "SELECT * FROM pos_tickets WHERE seller_id = ? AND ticket_id = ?", (seller_id, ticket_id))
    if t is None:
        raise ContractError("NOT_FOUND", "/ticket_id")
    return t


def _open(conn, seller_id: str, ticket_id: str) -> dict[str, Any]:
    t = _ticket(conn, seller_id, ticket_id)
    if t["status"] != "open":
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")
    return t


def open_ticket(conn, seller_id: str, *, table_no: Optional[int], guests: int = 1, at: Optional[str] = None,
                ticket_id: Optional[str] = None, kind: Optional[str] = None) -> dict[str, Any]:
    """The table's open check, or a new one. Without a table it is a counter order: takeout, or
    (kind dine_in) eaten in the store at no particular table; it gets the day's call number."""
    settings(conn, seller_id)
    table_name = call_no = None
    at = at or _now()
    if table_no is not None:
        t = pos_layout.table(conn, seller_id, int(table_no))
        if t is None:
            raise ContractError("NOT_FOUND", "/table")
        table_name = t["name"]
        existing = row(conn, "SELECT * FROM pos_tickets WHERE seller_id = ? AND table_no = ? AND status = 'open'",
                       (seller_id, table_no))
        if existing:
            return existing
        kind = "dine_in"
    else:
        kind = kind if kind in KINDS else "takeout"
        call_no = _next_call_no(conn, seller_id, at)
    ticket_id = ticket_id or uuid.uuid4().hex
    with conn:
        conn.execute("INSERT INTO pos_tickets (seller_id, ticket_id, table_no, table_name, call_no, kind, guests, status, opened_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                     (seller_id, ticket_id, table_no, table_name, call_no, kind, max(1, min(int(guests or 1), 30)), at))
    return _ticket(conn, seller_id, ticket_id)


def _next_call_no(conn, seller_id: str, at: str) -> int:
    """The number called out at the counter ("12번 손님"), counted per business day."""
    day = dt.datetime.fromisoformat(business_date(at)).replace(tzinfo=KST).astimezone(dt.timezone.utc)
    lo, hi = day.strftime("%Y-%m-%dT%H:%M:%S"), (day + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    n = conn.execute("SELECT COALESCE(MAX(call_no), 0) FROM pos_tickets WHERE seller_id = ? AND opened_at >= ? AND opened_at < ?",
                     (seller_id, lo, hi)).fetchone()[0]
    return n + 1


def set_guests(conn, seller_id: str, ticket_id: str, guests: int) -> None:
    _open(conn, seller_id, ticket_id)
    with conn:
        conn.execute("UPDATE pos_tickets SET guests = ? WHERE seller_id = ? AND ticket_id = ?",
                     (max(1, min(int(guests), 30)), seller_id, ticket_id))


def move_table(conn, seller_id: str, ticket_id: str, table_no: int) -> None:
    _open(conn, seller_id, ticket_id)
    t = pos_layout.table(conn, seller_id, int(table_no))
    if t is None:
        raise ContractError("NOT_FOUND", "/table")
    if row(conn, "SELECT 1 FROM pos_tickets WHERE seller_id = ? AND table_no = ? AND status = 'open' AND ticket_id != ?",
           (seller_id, table_no, ticket_id)):
        raise ContractError("DUPLICATE_EVENT", "/table_no")  # that table is busy
    with conn:
        conn.execute("UPDATE pos_tickets SET table_no = ?, table_name = ?, kind = 'dine_in' WHERE seller_id = ? AND ticket_id = ?",
                     (table_no, t["name"], seller_id, ticket_id))


def add_line(conn, seller_id: str, ticket_id: str, item_id: str, quantity: int = 1, memo: Optional[str] = None,
             at: Optional[str] = None, line_id: Optional[str] = None) -> None:
    """Add an item; the same item with the same memo that is not yet in the kitchen just grows."""
    _open(conn, seller_id, ticket_id)
    item = next((m for m in menu(conn, seller_id) if m["item_id"] == item_id), None)
    if item is None:
        raise ContractError("NOT_FOUND", "/item_id_local")
    if item["sold_out"]:
        raise ContractError("INVALID_TYPE", "/stock")
    quantity = max(1, min(int(quantity), 99))
    memo = (memo or "").strip()[:40] or None
    same = row(conn, "SELECT line_id, quantity FROM pos_lines WHERE seller_id = ? AND ticket_id = ? AND item_id_local = ? "
                     "AND sent_at IS NULL AND voided_at IS NULL AND COALESCE(memo, '') = ?",
               (seller_id, ticket_id, item_id, memo or ""))
    with conn:
        if same:
            conn.execute("UPDATE pos_lines SET quantity = ? WHERE seller_id = ? AND ticket_id = ? AND line_id = ?",
                         (min(99, same["quantity"] + quantity), seller_id, ticket_id, same["line_id"]))
        else:
            conn.execute("INSERT INTO pos_lines (seller_id, ticket_id, line_id, item_id_local, title, unit_price, quantity, memo, created_at) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (seller_id, ticket_id, line_id or uuid.uuid4().hex, item_id, item["title"], item["price"],
                          quantity, memo, at or _now()))


def set_line_quantity(conn, seller_id: str, ticket_id: str, line_id: str, quantity: int, at: Optional[str] = None) -> None:
    """0 removes a line not yet sent, or voids one the kitchen already has (shown struck through)."""
    _open(conn, seller_id, ticket_id)
    line = row(conn, "SELECT * FROM pos_lines WHERE seller_id = ? AND ticket_id = ? AND line_id = ?", (seller_id, ticket_id, line_id))
    if line is None or line["voided_at"]:
        raise ContractError("NOT_FOUND", "/line_id")
    with conn:
        if quantity <= 0:
            if line["sent_at"]:
                conn.execute("UPDATE pos_lines SET voided_at = ? WHERE seller_id = ? AND ticket_id = ? AND line_id = ?",
                             (at or _now(), seller_id, ticket_id, line_id))
            else:
                conn.execute("DELETE FROM pos_lines WHERE seller_id = ? AND ticket_id = ? AND line_id = ?", (seller_id, ticket_id, line_id))
        elif line["sent_at"] and quantity > line["quantity"]:
            # More of something already cooking is a new kitchen order, not an edit of the old one.
            add_line(conn, seller_id, ticket_id, line["item_id_local"], quantity - line["quantity"], line["memo"], at)
        else:
            conn.execute("UPDATE pos_lines SET quantity = ? WHERE seller_id = ? AND ticket_id = ? AND line_id = ?",
                         (min(99, quantity), seller_id, ticket_id, line_id))


def set_line_memo(conn, seller_id: str, ticket_id: str, line_id: str, memo: Optional[str]) -> None:
    _open(conn, seller_id, ticket_id)
    with conn:
        cur = conn.execute("UPDATE pos_lines SET memo = ? WHERE seller_id = ? AND ticket_id = ? AND line_id = ? AND sent_at IS NULL",
                           ((memo or "").strip()[:40] or None, seller_id, ticket_id, line_id))
    if cur.rowcount != 1:
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/line_id")  # already in the kitchen


def send_to_kitchen(conn, seller_id: str, ticket_id: str, at: Optional[str] = None) -> int:
    _open(conn, seller_id, ticket_id)
    with conn:
        cur = conn.execute("UPDATE pos_lines SET sent_at = ? WHERE seller_id = ? AND ticket_id = ? AND sent_at IS NULL AND voided_at IS NULL",
                           (at or _now(), seller_id, ticket_id))
    return cur.rowcount


def serve_line(conn, seller_id: str, line_id: str, at: Optional[str] = None) -> None:
    with conn:
        cur = conn.execute("UPDATE pos_lines SET served_at = ? WHERE seller_id = ? AND line_id = ? AND sent_at IS NOT NULL "
                           "AND served_at IS NULL AND voided_at IS NULL", (at or _now(), seller_id, line_id))
    if cur.rowcount != 1:
        raise ContractError("NOT_FOUND", "/line_id")


def set_discount(conn, seller_id: str, ticket_id: str, *, amount: int = 0, percent: int = 0) -> None:
    t = _open(conn, seller_id, ticket_id)
    subtotal = _subtotal(conn, seller_id, ticket_id)
    discount = amount if amount else round(subtotal * max(0, min(percent, 100)) / 100, -1)
    discount = max(0, min(int(discount), subtotal))
    if _paid(conn, seller_id, ticket_id) > subtotal - discount:
        raise ContractError("INVALID_TYPE", "/discount")
    with conn:
        conn.execute("UPDATE pos_tickets SET discount_minor = ? WHERE seller_id = ? AND ticket_id = ?", (discount, seller_id, t["ticket_id"]))


def void_ticket(conn, seller_id: str, ticket_id: str, reason: str = "") -> None:
    """The guests left without paying (or it was opened by mistake). Nothing reaches B."""
    _open(conn, seller_id, ticket_id)
    if _paid(conn, seller_id, ticket_id):
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")  # refund the partial payment first
    with conn:
        conn.execute("UPDATE pos_tickets SET status = 'void', note = ? WHERE seller_id = ? AND ticket_id = ?",
                     ((reason or "").strip()[:80] or "주문 취소", seller_id, ticket_id))


# --- members ------------------------------------------------------------------------

def phone_digits(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if not _PHONE.match(digits):
        raise ContractError("SCHEMA_INVALID", "/phone")
    return digits


def member_for_phone(conn, seller_id: str, phone: str, at: Optional[str] = None, create: bool = True) -> Optional[dict[str, Any]]:
    digits = phone_digits(phone)
    salt = settings(conn, seller_id)["member_salt"]
    phone_hash = hashlib.sha256(("%s:%s" % (salt, digits)).encode()).hexdigest()
    m = row(conn, "SELECT * FROM pos_members WHERE seller_id = ? AND phone_hash = ?", (seller_id, phone_hash))
    if m or not create:
        return m
    member_id = "m-" + phone_hash[:16]
    with conn:
        conn.execute("INSERT OR IGNORE INTO pos_members (seller_id, member_id, phone_hash, last4, joined_at) VALUES (?, ?, ?, ?, ?)",
                     (seller_id, member_id, phone_hash, digits[-4:], at or _now()))
    return row(conn, "SELECT * FROM pos_members WHERE seller_id = ? AND member_id = ?", (seller_id, member_id))


def attach_member(conn, seller_id: str, ticket_id: str, phone: str, at: Optional[str] = None) -> dict[str, Any]:
    _open(conn, seller_id, ticket_id)
    m = member_for_phone(conn, seller_id, phone, at)
    with conn:
        conn.execute("UPDATE pos_tickets SET member_id = ? WHERE seller_id = ? AND ticket_id = ?", (m["member_id"], seller_id, ticket_id))
    return m


def detach_member(conn, seller_id: str, ticket_id: str) -> None:
    _open(conn, seller_id, ticket_id)
    if _ticket(conn, seller_id, ticket_id)["points_used"]:
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/member")
    with conn:
        conn.execute("UPDATE pos_tickets SET member_id = NULL WHERE seller_id = ? AND ticket_id = ?", (seller_id, ticket_id))


def member(conn, seller_id: str, member_id: str) -> Optional[dict[str, Any]]:
    m = row(conn, "SELECT * FROM pos_members WHERE seller_id = ? AND member_id = ?", (seller_id, member_id))
    if m:
        m["label"] = "****-" + m["last4"]
    return m


# --- money ------------------------------------------------------------------------------

def _lines(conn, seller_id: str, ticket_id: str) -> list[dict[str, Any]]:
    return rows(conn, "SELECT * FROM pos_lines WHERE seller_id = ? AND ticket_id = ? ORDER BY created_at, rowid", (seller_id, ticket_id))


def _subtotal(conn, seller_id: str, ticket_id: str) -> int:
    return sum(l["unit_price"] * l["quantity"] for l in _lines(conn, seller_id, ticket_id) if not l["voided_at"])


def _paid(conn, seller_id: str, ticket_id: str) -> int:
    return conn.execute("SELECT COALESCE(SUM(amount), 0) FROM pos_payments WHERE seller_id = ? AND ticket_id = ? AND refunded_at IS NULL",
                        (seller_id, ticket_id)).fetchone()[0]


def view(conn, seller_id: str, ticket_id: str) -> dict[str, Any]:
    t = _ticket(conn, seller_id, ticket_id)
    lines = _lines(conn, seller_id, ticket_id)
    for l in lines:
        l["total"] = l["unit_price"] * l["quantity"]
        l["state"] = "void" if l["voided_at"] else ("served" if l["served_at"] else ("sent" if l["sent_at"] else "new"))
    subtotal = sum(l["total"] for l in lines if not l["voided_at"])
    total = subtotal - t["discount_minor"]
    payments = rows(conn, "SELECT * FROM pos_payments WHERE seller_id = ? AND ticket_id = ? ORDER BY paid_at, rowid", (seller_id, ticket_id))
    paid = sum(p["amount"] for p in payments if not p["refunded_at"])
    m = member(conn, seller_id, t["member_id"]) if t["member_id"] else None
    rate = settings(conn, seller_id)["point_rate"]
    return {**t, "lines": lines, "subtotal": subtotal, "total": total, "paid": paid, "due": max(0, total - paid),
            "payments": [dict(p, method_label=METHODS[p["method"]]) for p in payments], "member": m,
            "unsent": sum(1 for l in lines if l["state"] == "new"), "kind_label": KINDS[t["kind"]],
            "label": ticket_label(conn, seller_id, t, long=True), "short_label": ticket_label(conn, seller_id, t),
            "points_preview": int(max(0, total - t["points_used"]) * rate / 100) if m else 0, "point_rate": rate}


def ticket_label(conn, seller_id: str, t: dict[str, Any], long: bool = False) -> str:
    """How the screens name a check: the table's name when it was opened ("3번 테이블", "룸1"), or
    a counter order's call number ("포장 12", "매장 7")."""
    if t["table_no"]:
        name = t.get("table_name") or pos_layout.name_of(conn, seller_id, t["table_no"]) or str(t["table_no"])
        return pos_layout.label(name, long)
    base = KINDS[t["kind"]]
    return "%s %d" % (base, t["call_no"]) if t.get("call_no") else base


def _approval(payment_id: str) -> str:
    return "%08d" % (int(payment_id[:8], 16) % 100000000)


def pay(conn, seller_id: str, ticket_id: str, *, method: str, amount: Optional[int] = None, received: Optional[int] = None,
        at: Optional[str] = None, runtime=None, payment_id: Optional[str] = None) -> dict[str, Any]:
    """Take one payment; the last one that covers the bill completes the purchase (see module doc)."""
    if method not in METHODS:
        raise ContractError("INVALID_ENUM_VALUE", "/method")
    v = view(conn, seller_id, ticket_id)
    if v["status"] != "open":
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")
    if v["unsent"] and not v["paid"]:
        send_to_kitchen(conn, seller_id, ticket_id, at)  # counter-service shops pay first: the order goes in with the payment
        v = view(conn, seller_id, ticket_id)
    if v["total"] <= 0:
        raise ContractError("MISSING_REQUIRED_FIELD", "/lines")
    amount = v["due"] if amount is None else int(amount)
    if not 0 < amount <= v["due"]:
        raise ContractError("INVALID_TYPE", "/amount")
    change = None
    if method == "cash":
        received = amount if received is None else int(received)
        if received < amount:
            raise ContractError("INVALID_TYPE", "/received")
        change = received - amount
    if method == "points":
        if not v["member"] or v["member"]["points"] < amount:
            raise ContractError("INVALID_TYPE", "/points")
    payment_id = payment_id or uuid.uuid4().hex
    with conn:
        conn.execute("INSERT INTO pos_payments (seller_id, payment_id, ticket_id, method, amount, received, change_due, approval_no, paid_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (seller_id, payment_id, ticket_id, method, amount, received if method == "cash" else None, change,
                      _approval(payment_id) if method in ("card", "easy") else None, at or _now()))
        if method == "points":
            conn.execute("UPDATE pos_members SET points = points - ? WHERE seller_id = ? AND member_id = ?",
                         (amount, seller_id, v["member"]["member_id"]))
            conn.execute("UPDATE pos_tickets SET points_used = points_used + ? WHERE seller_id = ? AND ticket_id = ?",
                         (amount, seller_id, ticket_id))
    if amount == v["due"]:
        _settle(conn, seller_id, ticket_id, at or _now(), runtime, backdated=at is not None)
    out = view(conn, seller_id, ticket_id)
    out["change"] = change
    return out


def _settle(conn, seller_id: str, ticket_id: str, paid_at: str, runtime, backdated: bool = False) -> None:
    v = view(conn, seller_id, ticket_id)
    day = business_date(paid_at)
    seq = conn.execute("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = ? AND receipt_no LIKE ?",
                       (seller_id, day.replace("-", "") + "-%")).fetchone()[0] + 1
    receipt_no = "%s-%04d" % (day.replace("-", ""), seq)
    earned = 0
    if v["member"]:
        earned = int((v["total"] - v["points_used"]) * v["point_rate"] / 100)
    # The purchase: one A order per ticket (items merged per menu at their ticket price).
    items: dict[str, dict[str, Any]] = {}
    for l in v["lines"]:
        if l["voided_at"]:
            continue
        key = "%s@%d" % (l["item_id_local"], l["unit_price"])
        items.setdefault(key, {"item_id_local": l["item_id_local"], "quantity": 0, "unit_price_minor": l["unit_price"]})
        items[key]["quantity"] += l["quantity"]
    merged: dict[str, dict[str, Any]] = {}
    for it in items.values():  # an order holds each item once
        if it["item_id_local"] in merged:
            merged[it["item_id_local"]]["quantity"] += it["quantity"]
        else:
            merged[it["item_id_local"]] = dict(it)
    customer = v["member_id"] or ("guest-" + ticket_id[:16])
    key = "pos-" + ticket_id
    order = orders_service.place_order(conn, seller_id=seller_id, customer_id_local=customer, idempotency_key=key,
                                       items=list(merged.values()), currency="KRW", order_id=_id(seller_id, key))
    if order["status"] == "requested":
        orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="accept", expected_status_version=1)
        orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="complete",
                                        expected_status_version=2, runtime=runtime)
    with conn:
        conn.execute("UPDATE pos_tickets SET status = 'paid', paid_at = ?, receipt_no = ?, points_earned = ?, order_id = ? "
                     "WHERE seller_id = ? AND ticket_id = ?", (paid_at, receipt_no, earned, order["order_id"], seller_id, ticket_id))
        if v["member"]:
            conn.execute("UPDATE pos_members SET points = points + ?, visits = visits + 1, last_visit_at = ? WHERE seller_id = ? AND member_id = ?",
                         (earned, paid_at, seller_id, v["member_id"]))
        for it in merged.values():  # tracked stock goes down with the sale
            conn.execute("UPDATE item_stock SET stock = MAX(0, stock - ?) WHERE seller_id = ? AND item_id_local = ? AND stock IS NOT NULL",
                         (it["quantity"], seller_id, it["item_id_local"]))
        if backdated:  # a sale with given times (demo data): the order and its event carry them
            backdate(conn, seller_id, order["order_id"], v["opened_at"], paid_at)


def backdate(conn, seller_id: str, order_id: str, created_at: str, completed_at: str) -> None:
    """Give a just-made order and its not-yet-delivered purchase_event the times of the sale."""
    conn.execute("UPDATE orders SET created_at = ?, completed_at = ? WHERE seller_id = ? AND order_id = ?",
                 (created_at, completed_at, seller_id, order_id))
    ev = conn.execute("SELECT purchase_event_id, payload_json FROM purchase_events WHERE seller_id = ? AND order_id = ?",
                      (seller_id, order_id)).fetchone()
    if ev is None:
        return
    payload = json.loads(ev["payload_json"])
    payload["time"] = {"kind": "absolute", "value": completed_at}
    body = json.dumps(payload, ensure_ascii=False)
    conn.execute("UPDATE purchase_events SET payload_json = ? WHERE seller_id = ? AND purchase_event_id = ?",
                 (body, seller_id, ev["purchase_event_id"]))
    conn.execute("UPDATE outbox SET payload_json = ? WHERE seller_id = ? AND kind = 'purchase_event' AND ref_id = ? AND status = 'pending'",
                 (body, seller_id, ev["purchase_event_id"]))


def refund(conn, seller_id: str, ticket_id: str, at: Optional[str] = None) -> None:
    """Cancel every payment of a paid ticket and take its points back (the order stays; module doc)."""
    t = _ticket(conn, seller_id, ticket_id)
    if t["status"] != "paid":
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")
    now = at or _now()
    with conn:
        conn.execute("UPDATE pos_payments SET refunded_at = ? WHERE seller_id = ? AND ticket_id = ? AND refunded_at IS NULL",
                     (now, seller_id, ticket_id))
        conn.execute("UPDATE pos_tickets SET status = 'refunded', note = '결제 취소' WHERE seller_id = ? AND ticket_id = ?", (seller_id, ticket_id))
        if t["member_id"]:
            conn.execute("UPDATE pos_members SET points = MAX(0, points - ? + ?) WHERE seller_id = ? AND member_id = ?",
                         (t["points_earned"], t["points_used"], seller_id, t["member_id"]))


# --- screens' read side --------------------------------------------------------------------

def floor(conn, seller_id: str, now: Optional[dt.datetime] = None) -> dict[str, Any]:
    """The store's floor plan with each table's open check, plus open counter orders."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cfg = settings(conn, seller_id)
    plan = pos_layout.plan(conn, seller_id)
    open_tickets = rows(conn, "SELECT ticket_id FROM pos_tickets WHERE seller_id = ? AND status = 'open' ORDER BY opened_at", (seller_id,))
    views = [view(conn, seller_id, t["ticket_id"]) for t in open_tickets]
    limit = cfg["time_limit"] or 0
    for v in views:
        v["minutes"] = max(0, int((now - _parse(v["opened_at"])).total_seconds() // 60))
        v["waiting"] = sum(1 for l in v["lines"] if l["state"] == "sent")
        v["overtime"] = bool(limit) and v["minutes"] >= limit
    by_table = {v["table_no"]: v for v in views if v["table_no"]}
    zones = []
    for z in plan["zones"]:
        items = [dict(i, ticket=by_table.get(i["no"]) if i["kind"] == "table" else None,
                      label=pos_layout.label(i["name"]) if i["kind"] == "table" else i["name"])
                 for i in plan["items"] if i["zone"] == z]
        tables_in = [i for i in items if i["kind"] == "table"]
        zones.append({"name": z, "spots": items, "tables": len(tables_in), "busy": sum(1 for i in tables_in if i["ticket"]),
                      "seats": sum(i["seats"] for i in tables_in)})
    table_count = sum(z["tables"] for z in zones)
    return {"zones": zones, "cols": plan["cols"], "rows": plan["rows"], "counter": [v for v in views if not v["table_no"]],
            "settings": cfg, "busy": len(by_table), "table_count": table_count,
            "seats": sum(z["seats"] for z in zones), "open_total": sum(v["total"] for v in views)}


def table_choices(conn, seller_id: str) -> list[dict[str, Any]]:
    """Tables to move a check to, by zone, with whether each is taken."""
    busy = {r["table_no"] for r in rows(conn, "SELECT table_no FROM pos_tickets WHERE seller_id = ? AND status = 'open' "
                                             "AND table_no IS NOT NULL", (seller_id,))}
    out: dict[str, list[dict[str, Any]]] = {}
    for t in pos_layout.tables(conn, seller_id):
        out.setdefault(t["zone"], []).append({"no": t["no"], "label": pos_layout.label(t["name"], long=True), "seats": t["seats"],
                                              "busy": t["no"] in busy})
    return [{"zone": z, "tables": ts} for z, ts in out.items()]


def kitchen_queue(conn, seller_id: str) -> list[dict[str, Any]]:
    tickets: dict[str, dict[str, Any]] = {}
    for l in rows(conn, "SELECT l.*, t.table_no, t.table_name, t.call_no, t.kind FROM pos_lines l JOIN pos_tickets t USING (seller_id, ticket_id) "
                        "WHERE l.seller_id = ? AND l.sent_at IS NOT NULL AND l.served_at IS NULL AND l.voided_at IS NULL "
                        "AND t.status IN ('open', 'paid') ORDER BY l.sent_at, l.rowid", (seller_id,)):
        t = tickets.setdefault(l["ticket_id"], {"ticket_id": l["ticket_id"], "label": ticket_label(conn, seller_id, l),
                                                 "sent_at": l["sent_at"], "lines": []})
        t["lines"].append(l)
    now = dt.datetime.now(dt.timezone.utc)
    for t in tickets.values():
        t["minutes"] = max(0, int((now - _parse(t["sent_at"])).total_seconds() // 60))
    return list(tickets.values())


def paid_tickets(conn, seller_id: str, day: str) -> list[dict[str, Any]]:
    """Paid and refunded tickets of one business day (KST), newest first."""
    start = dt.datetime.fromisoformat(day).replace(tzinfo=KST).astimezone(dt.timezone.utc)
    end = start + dt.timedelta(days=1)
    out = []
    for t in rows(conn, "SELECT ticket_id FROM pos_tickets WHERE seller_id = ? AND status IN ('paid', 'refunded') "
                        "AND paid_at >= ? AND paid_at < ? ORDER BY paid_at DESC",
                  (seller_id, start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))):
        out.append(view(conn, seller_id, t["ticket_id"]))
    return out


def day_summary(conn, seller_id: str, day: str) -> dict[str, Any]:
    tickets = paid_tickets(conn, seller_id, day)
    paid = [t for t in tickets if t["status"] == "paid"]
    by_method: Counter = Counter()
    for t in paid:
        for p in t["payments"]:
            if not p["refunded_at"]:
                by_method[p["method"]] += p["amount"]
    items: Counter = Counter()
    hours: dict[int, int] = defaultdict(int)
    for t in paid:
        hours[_parse(t["paid_at"]).astimezone(KST).hour] += t["total"]
        for l in t["lines"]:
            if not l["voided_at"]:
                items[l["title"]] += l["quantity"]
    revenue = sum(t["total"] for t in paid)
    guests = sum(t["guests"] for t in paid)
    members = sum(1 for t in paid if t["member_id"])
    closing = row(conn, "SELECT * FROM pos_closings WHERE seller_id = ? AND business_date = ?", (seller_id, day))
    return {"day": day, "revenue": revenue, "count": len(paid), "guests": guests,
            "per_guest": round(revenue / guests) if guests else 0, "per_ticket": round(revenue / len(paid)) if paid else 0,
            "refunds": sum(1 for t in tickets if t["status"] == "refunded"),
            "refunded_amount": sum(t["total"] for t in tickets if t["status"] == "refunded"),
            "member_share": round(100 * members / len(paid)) if paid else 0,
            "takeout": sum(1 for t in paid if t["kind"] == "takeout"),
            "by_method": [(METHODS[m], a) for m, a in by_method.most_common()],
            "top_items": items.most_common(5), "hours": dict(hours),
            "discount": sum(t["discount_minor"] for t in paid), "points_used": sum(t["points_used"] for t in paid),
            "closed": closing}


def close_day(conn, seller_id: str, day: str, at: Optional[str] = None) -> dict[str, Any]:
    """Day closing (마감): record the totals once; open checks must be settled or voided first."""
    if row(conn, "SELECT 1 FROM pos_tickets WHERE seller_id = ? AND status = 'open'", (seller_id,)):
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/open_tickets")
    s = day_summary(conn, seller_id, day)
    s.pop("closed", None)
    with conn:
        conn.execute("INSERT OR REPLACE INTO pos_closings (seller_id, business_date, closed_at, summary_json) VALUES (?, ?, ?, ?)",
                     (seller_id, day, at or _now(), json.dumps(s, ensure_ascii=False)))
    return s


def members_overview(conn, seller_id: str) -> list[dict[str, Any]]:
    spend = {r["member_id"]: r["s"] for r in conn.execute(
        "SELECT t.member_id, SUM(o.amount) AS s FROM pos_tickets t JOIN (SELECT ticket_id, SUM(amount) AS amount FROM pos_payments "
        "WHERE seller_id = ? AND refunded_at IS NULL GROUP BY ticket_id) o USING (ticket_id) "
        "WHERE t.seller_id = ? AND t.status = 'paid' AND t.member_id IS NOT NULL GROUP BY t.member_id", (seller_id, seller_id))}
    out = rows(conn, "SELECT * FROM pos_members WHERE seller_id = ? ORDER BY visits DESC, last_visit_at DESC", (seller_id,))
    for m in out:
        m["label"] = "****-" + m["last4"]
        m["spend"] = spend.get(m["member_id"], 0)
    return out


def member_history(conn, seller_id: str, member_id: str, limit: int = 20) -> list[dict[str, Any]]:
    ids = rows(conn, "SELECT ticket_id FROM pos_tickets WHERE seller_id = ? AND member_id = ? AND status IN ('paid', 'refunded') "
                     "ORDER BY paid_at DESC LIMIT ?", (seller_id, member_id, limit))
    return [view(conn, seller_id, t["ticket_id"]) for t in ids]


def favourite_items(conn, seller_id: str, member_id: str) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for r in conn.execute("SELECT l.title, l.quantity FROM pos_lines l JOIN pos_tickets t USING (seller_id, ticket_id) "
                          "WHERE l.seller_id = ? AND t.member_id = ? AND t.status = 'paid' AND l.voided_at IS NULL", (seller_id, member_id)):
        c[r["title"]] += r["quantity"]
    return c.most_common(5)


def usual_menu(conn, seller_id: str, member_id: str, limit: int = 3) -> list[dict[str, Any]]:
    """This member's most ordered menus that are on sale now (counted, not predicted)."""
    on_sale = {m["item_id"]: m for m in menu(conn, seller_id) if not m["sold_out"]}
    out = []
    for r in conn.execute("SELECT l.item_id_local, SUM(l.quantity) AS n FROM pos_lines l JOIN pos_tickets t USING (seller_id, ticket_id) "
                          "WHERE l.seller_id = ? AND t.member_id = ? AND t.status = 'paid' AND l.voided_at IS NULL "
                          "GROUP BY l.item_id_local ORDER BY n DESC", (seller_id, member_id)):
        if r["item_id_local"] in on_sale:
            out.append(dict(on_sale[r["item_id_local"]], times=r["n"]))
        if len(out) >= limit:
            break
    return out


def menu_pairs(conn, seller_id: str, limit: int = 8, min_count: int = 3) -> list[dict[str, Any]]:
    """Menus ordered together, counted from this store's completed orders (lift = how much more than chance)."""
    baskets = []
    for o in orders_db.list_orders(conn, seller_id):
        if o["status"] == "completed":
            baskets.append({i["item_id_local"] for i in o["items"]})
    n = len(baskets)
    if not n:
        return []
    single: Counter = Counter()
    pair: Counter = Counter()
    for b in baskets:
        single.update(b)
        items = sorted(b)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair[(items[i], items[j])] += 1
    titles = {i["item_id_local"]: i["title_text"] for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
    out = []
    for (a, b), c in pair.items():
        if c < min_count:
            continue
        lift = c * n / (single[a] * single[b])
        out.append({"a": titles.get(a, a), "b": titles.get(b, b), "count": c, "lift": round(lift, 1),
                    "share": round(100 * c / min(single[a], single[b]))})
    out.sort(key=lambda p: (-p["lift"] * min(p["count"], 30), -p["count"]))
    return out[:limit]
