"""Aggregation-target variant parsing for vision suites."""

from __future__ import annotations

import re
from typing import List, Tuple

ParsedVariant = Tuple[str, str, List[str]]


def _target_mode(target: str, mode: str) -> List[str]:
    return ["--aggregation-target", target, "--graph-mode", mode]


def _target_knn(target: str, mode: str, k: str) -> List[str]:
    return ["--aggregation-target", target, "--graph-mode", mode, "--knn-k", k]


def _parse_filtered_target(prefix: str, target: str, v: str) -> ParsedVariant | None:
    if v == f"ours_{prefix}_dense":
        return "ours", v, _target_mode(target, "dense")

    if v == f"ours_{prefix}_uniform":
        return "ours", v, _target_mode(target, "uniform")

    m = re.match(rf"^ours_{prefix}_knn_k(\d+)$", v)
    if m:
        return "ours", v, _target_knn(target, "knn", m.group(1))

    if v == f"ours_{prefix}_magnitude":
        return "ours", v, _target_mode(target, "magnitude")

    m = re.match(rf"^ours_{prefix}_magnitude_knn_k(\d+)$", v)
    if m:
        return "ours", v, _target_knn(target, "magnitude_knn", m.group(1))

    if v == f"ours_{prefix}_rbf":
        return "ours", v, _target_mode(target, "rbf")

    m = re.match(rf"^ours_{prefix}_rbf_knn_k(\d+)$", v)
    if m:
        return "ours", v, _target_knn(target, "rbf_knn", m.group(1))

    m = re.match(rf"^ours_{prefix}_random_matched_k(\d+)$", v)
    if m:
        return "ours", v, _target_knn(target, "random", m.group(1))

    return None


def parse_target_variant(v: str) -> ParsedVariant | None:
    if v == "ours_weight_agg":
        return "ours", "ours_weight_agg", ["--aggregation-target", "weight"]

    graph_filtered = _parse_filtered_target(
        "graph_filtered",
        "graph_filtered_update",
        v,
    )
    if graph_filtered is not None:
        return graph_filtered

    spectral_filtered = _parse_filtered_target(
        "spectral_filtered",
        "spectral_filtered_update",
        v,
    )
    if spectral_filtered is not None:
        return spectral_filtered

    return None
