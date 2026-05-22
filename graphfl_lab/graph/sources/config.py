"""Configuration helpers for graph source selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphSourceConfig:
    source: str = "update"
    layer_start: int = 0
    layer_end: int = 0


def normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")
