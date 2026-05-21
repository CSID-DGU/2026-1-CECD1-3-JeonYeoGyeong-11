"""Module trace context helpers for GraphFL diagnostic artifacts."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def with_run_context(
    payload: Mapping[str, Any],
    *,
    round_number: int,
    run_id: str,
    variant: str,
    seed: int,
) -> Dict[str, Any]:
    enriched = dict(payload)
    if enriched.get("round") is None:
        enriched["round"] = int(round_number)
    values = dict(enriched.get("values") or {})
    values.setdefault("run_id", run_id)
    values.setdefault("variant", variant)
    values.setdefault("seed", int(seed))
    enriched["values"] = values
    return enriched


__all__ = ["with_run_context"]
