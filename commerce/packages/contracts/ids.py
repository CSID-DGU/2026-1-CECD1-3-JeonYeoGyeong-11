"""Identifiers and hashes that two roles compute independently and must agree on.

The rules are in docs/design/interfaces.md (§1 canonical JSON, §2 purchase_event_id).
Call these instead of re-implementing them: A computes purchase_event_id for live
orders, B for historical baskets, and the validator checks both against this code.
"""
import hashlib
import json
from typing import Any, Mapping


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def purchase_event_id(seller_id: str, source: str, basket_id_local: str) -> str:
    return hashlib.sha256(canonical_json([seller_id, source, basket_id_local])).hexdigest()[:32]


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return hashlib.sha256(canonical_json(body)).hexdigest()
