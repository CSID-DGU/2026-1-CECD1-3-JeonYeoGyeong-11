"""Order state machine, durable purchase-event handoff, catalog upsert.

Boundary (docs/design/interfaces.md §2-3, A card "경계와 통합"):
- A owns orders.sqlite exclusively; this module is the only caller of orders_db.
- completed order + purchase_event + outbox row are written in one DB
  transaction, then B's runtime.ingest_purchase_event is called after commit.
  A never marks an event delivered before B returns normally.
- Screens/HTTP auth are intentionally not built here: OQ13 (auth error code),
  OQ14/OQ19 (central catalog auth and seller link discovery) and OQ15 (order_id
  in seller-origin URLs) are open human decisions (docs/design/open-questions.md).
  This module only exposes plain Python calls a route layer can wrap once those
  land.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import sqlite3
import uuid
from typing import Any, Optional

from commerce.evaluation.metrics import ranking
from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented
from commerce.packages.contracts.ids import canonical_json, purchase_event_id
from commerce.packages.contracts.ports import RecommenderRuntime

from commerce.services.merchant_api import orders_db as db

# Screen-facing recommendation slot while B's runtime is unimplemented. Not a
# wire enum value; only a local tag so callers/templates can tell a mock
# ranking apart from a real B model_version string.
MOCK_MODEL_VERSION = "mock-p-topfreq-v1"

_log = logging.getLogger(__name__)

# interfaces.md §2 table, "live" row.
_LIVE_SOURCE = "live"
_LIVE_PARTITION = "platform_seller"

_ALLOWED_TRANSITIONS = {
    "accept": ("requested",),
    "complete": ("accepted",),
    "cancel": ("requested", "accepted"),
}
_TARGET_STATUS = {"accept": "accepted", "complete": "completed", "cancel": "cancelled"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_order_id() -> str:
    return uuid.uuid4().hex


def _request_hash(customer_id_local: str, items: list[dict[str, Any]], currency: str) -> str:
    body = {"customer_id_local": customer_id_local, "items": items, "currency": currency}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate item_id_local by summing quantity (interfaces.md §2)."""
    merged: dict[str, dict[str, Any]] = {}
    for raw in items:
        item_id = raw["item_id_local"]
        if item_id in merged:
            merged[item_id]["quantity"] += raw["quantity"]
        else:
            merged[item_id] = {
                "item_id_local": item_id,
                "quantity": raw["quantity"],
                "unit_price_minor": raw["unit_price_minor"],
            }
    return [merged[key] for key in sorted(merged)]


def place_order(
    conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str, idempotency_key: str,
    items: list[dict[str, Any]], currency: str,
) -> dict[str, Any]:
    """Create a requested order, or return the original on an identical retry.

    interfaces.md §2: same key + same normalized input -> return original order.
    Same key + different input -> DUPLICATE_IDEMPOTENCY_KEY.
    """
    normalized_items = _normalize_items(items)
    request_hash = _request_hash(customer_id_local, normalized_items, currency)

    existing = db.fetch_order_by_idempotency_key(conn, seller_id, idempotency_key)
    if existing is not None:
        if existing["request_hash"] == request_hash:
            return _to_commerce_order(existing)
        raise ContractError("DUPLICATE_IDEMPOTENCY_KEY", "/idempotency_key")

    now = _now_iso()
    order = {
        "seller_id": seller_id,
        "order_id": _new_order_id(),
        "customer_id_local": customer_id_local,
        "idempotency_key": idempotency_key,
        "request_hash": request_hash,
        "status": "requested",
        "status_version": 1,
        "created_at": now,
        "completed_at": None,
        "cancelled_at": None,
        "currency": currency,
        "items": normalized_items,
    }
    with conn:
        db.insert_order(conn, order)
    return _to_commerce_order(order)


def get_order(conn: sqlite3.Connection, *, seller_id: str, order_id: str) -> dict[str, Any]:
    """Cross-seller lookups return NOT_FOUND rather than FORBIDDEN so a probing
    request cannot tell "wrong seller" apart from "no such order"."""
    row = db.fetch_order(conn, seller_id, order_id)
    if row is None:
        raise ContractError("NOT_FOUND", "/order_id")
    return _to_commerce_order(row)


def list_orders(conn: sqlite3.Connection, *, seller_id: str) -> list[dict[str, Any]]:
    return [_to_commerce_order(row) for row in db.list_orders(conn, seller_id)]


def list_orders_by_customer(conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str) -> list[dict[str, Any]]:
    return [_to_commerce_order(row) for row in db.list_orders_by_customer(conn, seller_id, customer_id_local)]


def transition_order(
    conn: sqlite3.Connection, *, seller_id: str, order_id: str, action: str,
    expected_status_version: int, runtime: Optional[RecommenderRuntime] = None,
) -> dict[str, Any]:
    """Apply accept/complete/cancel. `action` picks the target status.

    On `complete`, the order update, the derived purchase_event, and its outbox
    row commit together in one transaction (interfaces.md §2 step 1); B is only
    called after that commit (step 2), outside this function's transaction.
    """
    if action not in _ALLOWED_TRANSITIONS:
        raise ValueError("action must be accept, complete, or cancel")

    row = db.fetch_order(conn, seller_id, order_id)
    if row is None:
        raise ContractError("NOT_FOUND", "/order_id")
    if row["status_version"] != expected_status_version:
        raise ContractError("STALE_STATUS_VERSION", "/status_version")
    if row["status"] not in _ALLOWED_TRANSITIONS[action]:
        raise ContractError("ILLEGAL_STATE_TRANSITION", "/status")

    new_status = _TARGET_STATUS[action]
    new_version = expected_status_version + 1
    now = _now_iso()
    completed_at = now if action == "complete" else row["completed_at"]
    cancelled_at = now if action == "cancel" else row["cancelled_at"]

    event = _build_purchase_event(seller_id, order_id, row, now) if action == "complete" else None

    with conn:
        changed = db.conditional_update_status(
            conn, seller_id, order_id, expected_status_version, new_status, new_version,
            completed_at=completed_at, cancelled_at=cancelled_at,
        )
        if not changed:
            # Someone else's concurrent transition won the race first.
            raise ContractError("STALE_STATUS_VERSION", "/status_version")
        if event is not None:
            db.insert_purchase_event(conn, seller_id, order_id, event, now)
            db.insert_outbox(conn, seller_id, "purchase_event", event["purchase_event_id"], event, now)

    updated = db.fetch_order(conn, seller_id, order_id)
    assert updated is not None
    if event is not None and runtime is not None:
        deliver_pending_purchase_events(conn, seller_id=seller_id, runtime=runtime)
    return _to_commerce_order(updated)


def _build_purchase_event(seller_id: str, order_id: str, order_row: dict[str, Any], now_iso: str) -> dict[str, Any]:
    pe_id = purchase_event_id(seller_id, _LIVE_SOURCE, order_id)
    return {
        "schema_version": "purchase_event.v1",
        "seller_id": seller_id,
        "source": _LIVE_SOURCE,
        "seller_partition": _LIVE_PARTITION,
        "customer_id_local": order_row["customer_id_local"],
        "basket_id_local": order_id,
        "purchase_event_id": pe_id,
        "time": {"kind": "absolute", "value": now_iso},
        "order_rank": None,
        "items": [
            {"item_id_local": item["item_id_local"], "quantity_observed": item["quantity"]}
            for item in order_row["items"]
        ],
    }


def deliver_pending_purchase_events(conn: sqlite3.Connection, *, seller_id: str, runtime: RecommenderRuntime) -> None:
    """Retry every pending purchase_event once. Safe to call repeatedly (interfaces.md §2 steps 4-5).

    - B returns normally, or raises on an identical redelivery that B already
      has: both count as delivered.
    - DUPLICATE_EVENT (same id, different body) is quarantined: never retried,
      never silently treated as delivered.
    - Any other failure (including B's current FeatureNotImplemented stub)
      stays pending for the next call.
    """
    # Each outcome is committed on its own (`with conn:`): an uncommitted mark is
    # rolled back when the request's connection closes, and every later request
    # would redeliver the whole outbox.
    for pending in db.fetch_pending_outbox(conn, seller_id, kind="purchase_event"):
        try:
            runtime.ingest_purchase_event(pending["payload"])
        except ContractError as exc:
            with conn:
                if exc.code == "DUPLICATE_EVENT":
                    db.mark_outbox_quarantined(conn, seller_id, "purchase_event", pending["ref_id"], exc.code)
                else:
                    db.mark_outbox_attempt(conn, seller_id, "purchase_event", pending["ref_id"], exc.code)
            continue
        except Exception as exc:  # B stub / transient failure: retry later.
            with conn:
                db.mark_outbox_attempt(conn, seller_id, "purchase_event", pending["ref_id"], str(exc))
            continue
        with conn:
            db.mark_outbox_delivered(conn, seller_id, "purchase_event", pending["ref_id"])


def register_catalog_item(
    conn: sqlite3.Connection, *, seller_id: str, item_id_local: str, title_text: str,
    description_text: Optional[str] = None, category_path: Optional[list[str]] = None,
    listing_status: str = "active", source: str = "live", display_price_minor: int = 0,
    runtime: Optional[RecommenderRuntime] = None,
) -> dict[str, Any]:
    """Upsert a catalog item locally, then forward it with an increasing source_seq
    (interfaces.md §3). B ignores an out-of-order (stale) source_seq on its side;
    A still keeps a durable outbox row so a delivery failure is retried, not lost.

    `display_price_minor` is a screen-only convenience (catalog_item.v1 has no
    price field; interfaces.md §2 puts price on the order, not the catalog). It
    is stored locally and never included in the payload forwarded to B/central.
    """
    prior = db.fetch_catalog_item(conn, seller_id, item_id_local)
    next_seq = (prior["source_seq"] + 1) if prior else 1
    first_listed_at = prior["payload"]["first_listed_at"] if prior else _now_iso()

    item = {
        "schema_version": "catalog_item.v1",
        "seller_id": seller_id,
        "item_id_local": item_id_local,
        "source": source,
        "title_text": title_text,
        "description_text": description_text,
        "category_path": category_path,
        "listing_status": listing_status,
        "first_listed_at": first_listed_at,
    }
    now = _now_iso()
    with conn:
        db.upsert_catalog_item(conn, seller_id, item_id_local, next_seq, item, display_price_minor)
        db.insert_outbox(conn, seller_id, "catalog_item", item_id_local, item, now, source_seq=next_seq)

    if runtime is not None:
        deliver_pending_catalog_items(conn, seller_id=seller_id, runtime=runtime)
    return item


def list_catalog_for_display(conn: sqlite3.Connection, *, seller_id: str) -> list[dict[str, Any]]:
    """Catalog rows plus the screen-only display_price_minor, for buyer/seller templates."""
    return db.list_catalog_items(conn, seller_id)


def get_catalog_item_for_display(conn: sqlite3.Connection, *, seller_id: str, item_id_local: str) -> Optional[dict[str, Any]]:
    row = db.fetch_catalog_item(conn, seller_id, item_id_local)
    if row is None:
        return None
    return {**row["payload"], "display_price_minor": row["display_price_minor"]}


def build_recommendation_request(
    seller_id: str, customer_id_local: str, *,
    candidate_item_ids: Optional[list[str]] = None, top_n: int = 4,
) -> dict[str, Any]:
    return {
        "schema_version": "recommendation_request.v1",
        "seller_id": seller_id,
        "customer_id_local": customer_id_local,
        "as_of": _now_iso(),
        "candidate_item_ids": candidate_item_ids,
        "top_n": top_n,
    }


def get_recommendations_for_display(
    conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str,
    top_n: int = 4, runtime: Optional[RecommenderRuntime] = None,
) -> dict[str, Any]:
    """Buyer-screen recommendation slot (A card "B의 실제 추천을 화면에 연결한다").

    Tries B's real `predict_local` first. B's own fallback (no installed base,
    no seller/customer history) comes back as a normal recommendation with
    `is_cold_start`/`fallback_reason` set; `recommendation_label` turns that
    into the screen badge. Only when B cannot answer at all -- the
    `UnimplementedRuntime` stub's `FeatureNotImplemented`, or any other failure
    -- does this fall back to A's P-TopFreq non-model baseline
    (`commerce.evaluation.metrics.ranking.p_topfreq_ranking`), tagged
    MOCK_MODEL_VERSION. A buyer page never fails because the recommender did.

    Pending catalog deliveries are retried first, and candidates are left to B
    (`candidate_item_ids=None`, B's active catalog mirror): an explicit
    candidate B has not received yet is NOT_FOUND on B's side, which would
    otherwise take down the whole slot. B's answer is returned unchanged; the
    screen drops any item missing from A's own catalog when it renders.
    """
    active_items = [
        item["item_id_local"] for item in db.list_catalog_items(conn, seller_id)
        if item["listing_status"] == "active"
    ]
    request = build_recommendation_request(seller_id, customer_id_local, candidate_item_ids=None, top_n=top_n)
    if runtime is not None:
        deliver_pending_catalog_items(conn, seller_id=seller_id, runtime=runtime)
        try:
            return runtime.predict_local(request)
        except FeatureNotImplemented:
            pass
        except Exception as exc:  # recommender failure must not break the buyer page
            _log.warning("predict_local failed, using A baseline: %s", exc)

    seller_counts: dict[str, int] = {}
    for order in db.list_orders(conn, seller_id):
        if order["status"] != "completed":
            continue
        for item in order["items"]:
            seller_counts[item["item_id_local"]] = seller_counts.get(item["item_id_local"], 0) + item["quantity"]
    ranked = ranking.p_topfreq_ranking(seller_counts, active_items)[:top_n]
    return {
        "schema_version": "recommendation.v1",
        "seller_id": seller_id,
        "customer_id_local": customer_id_local,
        "as_of": request["as_of"],
        "model_version": MOCK_MODEL_VERSION,
        "score_semantics": "next_purchase",
        "horizon_days": None,
        "is_cold_start": True,
        "fallback_reason": "no_shared_model",
        "items": [{"item_id_local": item_id, "score": score} for item_id, score in ranked],
    }


def deliver_all_pending(conn: sqlite3.Connection, *, seller_id: str, runtime: RecommenderRuntime) -> None:
    """Retry the whole outbox once: catalog first, so B knows an item before an
    event that names it. Called at app startup (interfaces.md §2: a restart
    redelivers what was committed but not acknowledged) -- which also hands B
    every order the demo seed wrote while no runtime was attached."""
    deliver_pending_catalog_items(conn, seller_id=seller_id, runtime=runtime)
    deliver_pending_purchase_events(conn, seller_id=seller_id, runtime=runtime)


def delivery_summary(conn: sqlite3.Connection, *, seller_id: str) -> dict[str, dict[str, int]]:
    """kind -> {"pending", "delivered", "quarantined"} counts, for the seller dashboard."""
    counts = db.count_outbox(conn, seller_id)
    return {
        kind: {status: counts.get((kind, status), 0) for status in ("pending", "delivered", "quarantined")}
        for kind in ("catalog_item", "purchase_event")
    }


def purchase_event_status_by_order(conn: sqlite3.Connection, *, seller_id: str) -> dict[str, str]:
    return db.purchase_event_status_by_order(conn, seller_id)


_FALLBACK_LABELS = {
    "no_shared_model": "공유 모델 준비 전 · 매장 인기순",
    "no_seller_history": "판매 이력 부족 · 매장 인기순",
    "no_customer_history": "첫 방문 고객 · 매장 인기순",
}


def recommendation_label(recommendation: dict[str, Any]) -> Optional[str]:
    """Badge text for a recommendation that is not a model ranking, or None for a real one.

    Covers both A's own baseline (MOCK_MODEL_VERSION) and B's fallback, which
    reports itself through is_cold_start/fallback_reason (interfaces.md §4)
    rather than through a special model_version.
    """
    if recommendation["model_version"] == MOCK_MODEL_VERSION:
        return "임시 · 추천 모델 미연결, 인기순"
    if recommendation.get("fallback_reason"):
        return _FALLBACK_LABELS.get(recommendation["fallback_reason"], "대체 추천 · 매장 인기순")
    return None


def get_comparison_for_display(
    conn: sqlite3.Connection, *, seller_id: str, customer_id_local: str,
    top_n: int = 5, runtime: Optional[RecommenderRuntime] = None,
) -> Optional[Any]:
    """One B compare_local call for the seller's comparison screen (comparison.md §5).

    A sends exactly one request and computes nothing model-side; B pins the
    snapshot, candidates and handles for all four arms. Returns None while B
    cannot compare at all (stub), so the screen can say so instead of
    inventing arms. Permission/request errors propagate (interfaces.md §4.1:
    they reject the comparison, they are not "unavailable").
    """
    if runtime is None:
        return None
    deliver_pending_catalog_items(conn, seller_id=seller_id, runtime=runtime)
    request = build_recommendation_request(seller_id, customer_id_local, candidate_item_ids=None, top_n=top_n)
    try:
        return runtime.compare_local(request)
    except FeatureNotImplemented:
        return None


def deliver_pending_catalog_items(conn: sqlite3.Connection, *, seller_id: str, runtime: RecommenderRuntime) -> None:
    for pending in db.fetch_pending_outbox(conn, seller_id, kind="catalog_item"):
        try:
            runtime.upsert_catalog_item(pending["payload"], pending["source_seq"])
        except Exception as exc:  # B stub / transient failure: retry later.
            with conn:
                db.mark_outbox_attempt(conn, seller_id, "catalog_item", pending["ref_id"], str(exc))
            continue
        with conn:  # committed per item, as in deliver_pending_purchase_events
            db.mark_outbox_delivered(conn, seller_id, "catalog_item", pending["ref_id"])


def _to_commerce_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "commerce_order.v1",
        "seller_id": order["seller_id"],
        "order_id": order["order_id"],
        "customer_id_local": order["customer_id_local"],
        "idempotency_key": order["idempotency_key"],
        "status": order["status"],
        "status_version": order["status_version"],
        "created_at": order["created_at"],
        "completed_at": order["completed_at"],
        "cancelled_at": order["cancelled_at"],
        "currency": order["currency"],
        "items": order["items"],
    }
