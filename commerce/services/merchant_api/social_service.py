"""Layer 6 social/add-on business logic (M25 DM, M26/M27 feed, M29 group-buy,
M30 price history). All local to one seller's orders.sqlite; no cross-role
contract. Group-buy settlement creates ordinary orders through orders_service
(M29 -> M6 per the module doc); it does not create any payout/settlement
record (M10 is out of scope for now).
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from typing import Any, Optional

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_service
from commerce.services.merchant_api import social_db as db


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_id() -> str:
    return uuid.uuid4().hex


# --- messages (M25) ----------------------------------------------------------

def send_message(conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str,
                  sender: str, body: str) -> dict[str, Any]:
    if sender not in ("customer", "seller"):
        raise ValueError("sender must be 'customer' or 'seller'")
    if not body.strip():
        raise ContractError("MISSING_REQUIRED_FIELD", "/body")
    now = _now_iso()
    message = {
        "message_id": _new_id(), "seller_id": seller_id, "customer_id_local": customer_id_local,
        "sender": sender, "body": body.strip(), "created_at": now,
    }
    with conn:
        db.insert_message(conn, seller_id, message["message_id"], customer_id_local, sender, message["body"], now)
    return message


def list_threads(conn: sqlite3.Connection, *, seller_id: str) -> list[dict[str, Any]]:
    return db.list_threads(conn, seller_id)


def list_thread_messages(conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str) -> list[dict[str, Any]]:
    return db.list_thread_messages(conn, seller_id, customer_id_local)


# --- posts / feed (M26, M27) --------------------------------------------------

def create_post(conn: sqlite3.Connection, *, seller_id: str, kind: str, title: str,
                 body: Optional[str] = None, media_path: Optional[str] = None) -> dict[str, Any]:
    if kind not in ("article", "short_video"):
        raise ValueError("kind must be 'article' or 'short_video'")
    now = _now_iso()
    post = {
        "post_id": _new_id(), "seller_id": seller_id, "kind": kind, "title": title,
        "body": body, "media_path": media_path, "created_at": now,
    }
    with conn:
        db.insert_post(conn, seller_id, post["post_id"], kind, title, body, media_path, now)
    return post


def list_feed(conn: sqlite3.Connection, *, seller_id: str) -> list[dict[str, Any]]:
    """M28 feed ranking slot: a local recency heuristic today (list_posts already
    orders by created_at desc). No RecommenderRuntime hook exists for feed/short-form
    ranking yet -- ports.py's predict_local is item-ranking shaped, not post-ranking
    shaped, so this is not wired as a B fallback the way product recommendations are.
    If/when the team adds a feed-ranking contract, swap the body of this function,
    not its callers.
    """
    return db.list_posts(conn, seller_id)


# --- group buys (M29) ---------------------------------------------------------

def create_group_buy(conn: sqlite3.Connection, *, seller_id: str, item_id_local: str,
                      target_quantity: int, unit_price_minor: int, deadline_at: str) -> dict[str, Any]:
    if target_quantity <= 0:
        raise ContractError("INVALID_TYPE", "/target_quantity")
    now = _now_iso()
    group_buy = {
        "group_buy_id": _new_id(), "seller_id": seller_id, "item_id_local": item_id_local,
        "target_quantity": target_quantity, "unit_price_minor": unit_price_minor,
        "deadline_at": deadline_at, "status": "open", "created_at": now,
    }
    with conn:
        db.insert_group_buy(conn, seller_id, group_buy["group_buy_id"], item_id_local,
                             target_quantity, unit_price_minor, deadline_at, now)
    return group_buy


def _with_progress(conn: sqlite3.Connection, seller_id: str, group_buy: dict[str, Any]) -> dict[str, Any]:
    joined = db.sum_group_buy_quantity(conn, seller_id, group_buy["group_buy_id"])
    return {**group_buy, "joined_quantity": joined}


def list_group_buys(conn: sqlite3.Connection, *, seller_id: str) -> list[dict[str, Any]]:
    settle_due_group_buys(conn, seller_id=seller_id)
    return [_with_progress(conn, seller_id, gb) for gb in db.list_group_buys(conn, seller_id)]


def join_group_buy(conn: sqlite3.Connection, *, seller_id: str, group_buy_id: str,
                    customer_id_local: str, quantity: int) -> dict[str, Any]:
    if quantity <= 0:
        raise ContractError("INVALID_TYPE", "/quantity")
    group_buy = db.fetch_group_buy(conn, seller_id, group_buy_id)
    if group_buy is None:
        raise ContractError("NOT_FOUND", "/group_buy_id")
    if group_buy["status"] != "open":
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")
    with conn:
        joined_ok = db.insert_group_buy_participant(
            conn, seller_id, group_buy_id, customer_id_local, quantity, _now_iso(),
        )
    if not joined_ok:
        raise ContractError("DUPLICATE_EVENT", "/customer_id_local")

    # Module doc: "목표 수량 도달 시 할인 적용" -- success fires the moment the
    # target is met, not only at the deadline. The expiry sweep below only
    # needs to cover the "ran out of time, still short" case.
    joined_quantity = db.sum_group_buy_quantity(conn, seller_id, group_buy_id)
    if joined_quantity >= group_buy["target_quantity"]:
        _settle_succeeded(conn, seller_id=seller_id, group_buy=group_buy)
        group_buy = db.fetch_group_buy(conn, seller_id, group_buy_id)
    return _with_progress(conn, seller_id, group_buy)


def _settle_succeeded(conn: sqlite3.Connection, *, seller_id: str, group_buy: dict[str, Any]) -> None:
    """One ordinary order per participant (M29 -> M6 per the module doc); no
    settlement/payout record is created (M10 deferred). idempotency_key is
    deterministic so a retry (e.g. the expiry sweep re-checking) never double-orders."""
    for participant in db.list_group_buy_participants(conn, seller_id, group_buy["group_buy_id"]):
        orders_service.place_order(
            conn, seller_id=seller_id, customer_id_local=participant["customer_id_local"],
            idempotency_key="group-buy-%s-%s" % (group_buy["group_buy_id"], participant["customer_id_local"]),
            items=[{
                "item_id_local": group_buy["item_id_local"], "quantity": participant["quantity"],
                "unit_price_minor": group_buy["unit_price_minor"],
            }],
            currency="KRW",
        )
    with conn:
        db.update_group_buy_status(conn, seller_id, group_buy["group_buy_id"], "succeeded")


def settle_due_group_buys(conn: sqlite3.Connection, *, seller_id: str) -> None:
    """Past-deadline open group buys that never reached target are marked
    failed, checked lazily on each list call -- same pattern as A's outbox
    delivery, no cron needed for the demo. Reaching target is already settled
    immediately in join_group_buy, so this only has the under-target case left,
    but still re-checks >= target defensively before failing a group buy."""
    now = _now_iso()
    for group_buy in db.list_group_buys(conn, seller_id):
        if group_buy["status"] != "open" or group_buy["deadline_at"] > now:
            continue
        joined = db.sum_group_buy_quantity(conn, seller_id, group_buy["group_buy_id"])
        if joined >= group_buy["target_quantity"]:
            _settle_succeeded(conn, seller_id=seller_id, group_buy=group_buy)
        else:
            with conn:
                db.update_group_buy_status(conn, seller_id, group_buy["group_buy_id"], "failed")


# --- price history (M30) ------------------------------------------------------

def record_price(conn: sqlite3.Connection, *, seller_id: str, item_id_local: str,
                  price_minor: int, price_date: Optional[str] = None) -> dict[str, Any]:
    date = price_date or _now_iso()[:10]
    with conn:
        db.upsert_price(conn, seller_id, item_id_local, date, price_minor)
    return {"seller_id": seller_id, "item_id_local": item_id_local, "price_date": date, "price_minor": price_minor}


def get_price_history(conn: sqlite3.Connection, *, seller_id: str, item_id_local: str) -> list[dict[str, Any]]:
    return db.list_price_history(conn, seller_id, item_id_local)
