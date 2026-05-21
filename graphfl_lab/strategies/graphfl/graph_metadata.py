"""Graph metadata normalization helpers for GraphFL diagnostics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def client_cluster_ids_from_meta(
    graph_meta: Mapping[str, Any],
    cids: Sequence[str],
) -> list[int]:
    cluster_ids = graph_meta.get("cluster_ids")
    if isinstance(cluster_ids, list) and len(cluster_ids) == len(cids):
        return [int(value) for value in cluster_ids]
    return [-1 for _ in cids]


__all__ = ["client_cluster_ids_from_meta"]
