"""Shop extras: search, reviews (verified buyers only), wishlist, product editing, buyer cancel."""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_db, orders_service, shop_db


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def category_of(item: dict[str, Any]) -> str:
    path = item.get("category_path") or []
    return path[-1] if path else "기타"


def search(catalog: list[dict[str, Any]], q: Optional[str], category: Optional[str]) -> list[dict[str, Any]]:
    """Active items whose title, description or categories contain every word of q, in the given category."""
    words = [w.lower() for w in (q or "").split() if w]
    out = []
    for item in catalog:
        if item["listing_status"] != "active":
            continue
        if category and category_of(item) != category:
            continue
        haystack = " ".join([item["title_text"], item.get("description_text") or "",
                             " ".join(item.get("category_path") or [])]).lower()
        if all(w in haystack for w in words):
            out.append(item)
    return out


def categories(catalog: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for item in catalog:
        if item["listing_status"] == "active":
            counts[category_of(item)] = counts.get(category_of(item), 0) + 1
    return sorted(counts, key=lambda c: (-counts[c], c))


def bought(conn, seller_id: str, customer_id: str, item_id: str) -> bool:
    return any(o["status"] == "completed" and any(i["item_id_local"] == item_id for i in o["items"])
               for o in orders_db.list_orders_by_customer(conn, seller_id, customer_id))


def write_review(conn, *, seller_id: str, customer_id_local: str, item_id_local: str, rating: int,
                 body: Optional[str], at: Optional[str] = None) -> None:
    """One review per customer and item, only after a completed purchase of it (edits replace it)."""
    if not 1 <= rating <= 5:
        raise ContractError("INVALID_TYPE", "/rating")
    if not bought(conn, seller_id, customer_id_local, item_id_local):
        raise ContractError("FORBIDDEN", "/item_id_local")
    with conn:
        shop_db.upsert_review(conn, seller_id, item_id_local, customer_id_local, rating,
                              (body or "").strip()[:1000] or None, at or _now())


def reply_review(conn, *, seller_id: str, item_id_local: str, customer_id_local: str, reply: str) -> None:
    with conn:
        if not shop_db.reply_review(conn, seller_id, item_id_local, customer_id_local, (reply or "").strip()[:500] or None):
            raise ContractError("NOT_FOUND", "/review")


def toggle_wish(conn, *, seller_id: str, customer_id_local: str, item_id_local: str) -> bool:
    if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local) is None:
        raise ContractError("NOT_FOUND", "/item_id_local")
    with conn:
        return shop_db.toggle_wish(conn, seller_id, customer_id_local, item_id_local, _now())


def edit_product(conn, *, seller_id: str, item_id_local: str, title_text: str, display_price_minor: int,
                 category_path: list[str], description_text: Optional[str], listing_status: str, runtime=None) -> None:
    """Re-registers the item: B receives it as a catalog update with the next source_seq."""
    if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local) is None:
        raise ContractError("NOT_FOUND", "/item_id_local")
    if not title_text.strip() or display_price_minor < 0:
        raise ContractError("SCHEMA_INVALID", "/title_text")
    if listing_status not in ("active", "inactive"):
        raise ContractError("INVALID_ENUM_VALUE", "/listing_status")
    orders_service.register_catalog_item(
        conn, seller_id=seller_id, item_id_local=item_id_local, title_text=title_text.strip(),
        description_text=(description_text or "").strip() or None, category_path=[c for c in category_path if c] or None,
        listing_status=listing_status, display_price_minor=display_price_minor, runtime=runtime)


def cancel_by_buyer(conn, *, seller_id: str, customer_id_local: str, order_id: str, runtime=None) -> None:
    """A buyer may cancel their own order while the seller has not accepted it."""
    order = orders_service.get_order(conn, seller_id=seller_id, order_id=order_id)
    if order["customer_id_local"] != customer_id_local:
        raise ContractError("NOT_FOUND", "/order_id")
    if order["status"] != "requested":
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")
    orders_service.transition_order(conn, seller_id=seller_id, order_id=order_id, action="cancel",
                                    expected_status_version=order["status_version"], runtime=runtime)
