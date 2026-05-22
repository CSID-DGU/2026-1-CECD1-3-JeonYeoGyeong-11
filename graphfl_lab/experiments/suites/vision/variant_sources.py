"""Source-specific Ours graph variant parsing."""

from __future__ import annotations

import re
from typing import List, Tuple

ParsedVariant = Tuple[str, str, List[str]]


def _source_mode(source: str, mode: str = "dense") -> List[str]:
    return ["--graph-source", source, "--graph-mode", mode]


def _source_target_mode(source: str, target: str, mode: str = "dense") -> List[str]:
    return ["--graph-source", source, "--aggregation-target", target, "--graph-mode", mode]


def _source_knn(source: str, k: str) -> List[str]:
    return ["--graph-source", source, "--graph-mode", "knn", "--knn-k", k]


def _source_target_knn(source: str, target: str, k: str) -> List[str]:
    return [
        "--graph-source",
        source,
        "--aggregation-target",
        target,
        "--graph-mode",
        "knn",
        "--knn-k",
        k,
    ]


def _tail_source(source: str, start: str, k: str) -> List[str]:
    return [
        "--graph-source",
        source,
        "--graph-layer-start",
        start,
        "--graph-mode",
        "knn",
        "--knn-k",
        k,
    ]


def parse_source_variant(v: str) -> ParsedVariant | None:
    if v == "ours_weight_graph":
        return "ours", "ours_weight_graph", _source_mode("weight")

    if v == "ours_weight_graph_weight_agg":
        return "ours", "ours_weight_graph_weight_agg", _source_target_mode("weight", "weight")

    if v == "ours_weight_graph_filtered_weight_agg":
        return (
            "ours",
            "ours_weight_graph_filtered_weight_agg",
            _source_target_mode("weight", "graph_filtered_weight"),
        )

    if v == "ours_weight_graph_spectral_weight_agg":
        return (
            "ours",
            "ours_weight_graph_spectral_weight_agg",
            _source_target_mode("weight", "graph_filtered_weight"),
        )

    m = re.match(r"^ours_weight_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("weight", m.group(1))

    m = re.match(r"^ours_weight_graph_weight_agg_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_target_knn("weight", "weight", m.group(1))

    m = re.match(r"^ours_weight_graph_filtered_weight_agg_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_target_knn("weight", "graph_filtered_weight", m.group(1))

    m = re.match(r"^ours_weight_graph_spectral_weight_agg_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_target_knn("weight", "graph_filtered_weight", m.group(1))

    if v == "ours_layerwise_graph":
        return "ours", "ours_layerwise_graph", _source_mode("layerwise_update")

    m = re.match(r"^ours_layerwise_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("layerwise_update", m.group(1))

    m = re.match(r"^ours_layerwise_weight_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("layerwise_weight", m.group(1))

    m = re.match(r"^ours_head_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("classifier_head_update", m.group(1))

    m = re.match(r"^ours_head_ema_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("classifier_head_ema_update", m.group(1))

    m = re.match(r"^ours_head_weight_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("classifier_head_weight", m.group(1))

    m = re.match(r"^ours_head_weight_graph_filtered_weight_agg_knn_k(\d+)$", v)
    if m:
        return (
            "ours",
            v,
            _source_target_knn("classifier_head_weight", "graph_filtered_weight", m.group(1)),
        )

    m = re.match(r"^ours_head_weight_graph_spectral_weight_agg_knn_k(\d+)$", v)
    if m:
        return (
            "ours",
            v,
            _source_target_knn("classifier_head_weight", "graph_filtered_weight", m.group(1)),
        )

    m = re.match(r"^ours_layerwise_head_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("layerwise_classifier_head_update", m.group(1))

    m = re.match(r"^ours_layerwise_head_ema_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("layerwise_classifier_head_ema_update", m.group(1))

    m = re.match(r"^ours_layerwise_head_weight_graph_knn_k(\d+)$", v)
    if m:
        return "ours", v, _source_knn("layerwise_classifier_head_weight", m.group(1))

    m = re.match(r"^ours_ema_graph_knn_k(\d+)$", v)
    if m:
        return (
            "ours",
            v,
            _source_target_knn("ema_update", "graph_filtered_update", m.group(1)),
        )

    m = re.match(r"^ours_ema_signal_knn_k(\d+)$", v)
    if m:
        return (
            "ours",
            v,
            _source_target_knn("ema_update", "graph_filtered_ema_update", m.group(1)),
        )

    m = re.match(r"^ours_tail_m(\d+)_knn_k(\d+)$", v)
    if m:
        return "ours", v, _tail_source("layer_slice_update", f"-{m.group(1)}", m.group(2))

    m = re.match(r"^ours_layerwise_tail_m(\d+)_knn_k(\d+)$", v)
    if m:
        return "ours", v, _tail_source("layerwise_slice_update", f"-{m.group(1)}", m.group(2))

    m = re.match(r"^ours_weight_tail_m(\d+)_knn_k(\d+)$", v)
    if m:
        return "ours", v, _tail_source("layer_slice_weight", f"-{m.group(1)}", m.group(2))

    return None
