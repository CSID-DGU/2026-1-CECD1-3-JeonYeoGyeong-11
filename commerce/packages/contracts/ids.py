"""Identifiers and hashes that two roles compute independently and must agree on.

The rules are in docs/design/interfaces.md (§1 canonical JSON, §2 purchase_event_id).
Call these instead of re-implementing them: A computes purchase_event_id for live
orders, B for historical baskets, and the validator checks both against this code.
"""
import hashlib
import json
from typing import Any, Iterable, Mapping


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def purchase_event_id(seller_id: str, source: str, basket_id_local: str) -> str:
    return hashlib.sha256(canonical_json([seller_id, source, basket_id_local])).hexdigest()[:32]


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def snapshot_digest(purchase_event_ids: Iterable[str], catalog: Mapping[str, Mapping[str, Any]]) -> str:
    """Content hash of a seller's training snapshot (D0025, OQ17).

    B computes it over the events and the latest catalog_item.v1 per item in a
    local_data_ref snapshot; the synthetic demo runner computes it over what its
    generator wrote. Equal digests mean the FL client trains on exactly the
    generated input. Order does not matter; a repeated event ID is an error.
    """
    events = sorted(purchase_event_ids)
    if len(set(events)) != len(events):
        raise ValueError("a purchase_event_id appears twice")
    items = sorted([item_id, hashlib.sha256(canonical_json(body)).hexdigest()] for item_id, body in catalog.items())
    return hashlib.sha256(canonical_json({"events": events, "catalog": items})).hexdigest()
