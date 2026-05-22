"""Core Ours graph-mode variant parsing."""

from __future__ import annotations

import re
from typing import List, Tuple

ParsedVariant = Tuple[str, str, List[str]]


def parse_core_graph_variant(v: str, default_knn_k: int) -> ParsedVariant | None:
    if v == "ours_default_graph":
        return "ours", v, ["--graph-method", "default_similarity_knn"]

    m = re.match(r"^ours_default_graph_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-method",
                "default_similarity_knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "knn", "--knn-k", k]

    if v == "ours_knn":
        return "ours", "ours_knn", ["--graph-mode", "knn", "--knn-k", str(default_knn_k)]

    if v in {"ours_update_graph_update_agg", "ours_grad_graph_grad_agg"}:
        return (
            "ours",
            v,
            ["--graph-source", "update", "--aggregation-target", "update"],
        )

    m = re.match(r"^ours_(?:update|grad)_graph_(?:update|grad)_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "update",
                "--aggregation-target",
                "update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_mutual_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "mutual_knn", "--knn-k", k]

    if v == "ours_mutual_knn":
        return (
            "ours",
            "ours_mutual_knn",
            ["--graph-mode", "mutual_knn", "--knn-k", str(default_knn_k)],
        )

    if v == "ours_dense":
        return "ours", "ours_dense", ["--graph-mode", "dense"]

    if v == "ours_magnitude":
        return "ours", "ours_magnitude", ["--graph-mode", "magnitude"]

    m = re.match(r"^ours_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "magnitude_knn", "--knn-k", k]

    if v == "ours_global_alignment":
        return "ours", "ours_global_alignment", ["--graph-mode", "global_alignment"]

    if v == "ours_rbf":
        return "ours", "ours_rbf", ["--graph-mode", "rbf"]

    m = re.match(r"^ours_rbf_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "rbf_knn", "--knn-k", k]

    if v in {"ours_learned_graph", "ours_learned_smooth"}:
        return "ours", v, ["--graph-mode", "learned_smooth"]

    m = re.match(r"^ours_learned_smooth_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "learned_smooth_knn", "--knn-k", k]

    if v == "ours_signed_abs":
        return "ours", "ours_signed_abs", ["--graph-mode", "signed_abs"]

    m = re.match(r"^ours_signed_abs_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "signed_abs_knn", "--knn-k", k]

    if v == "ours_negative":
        return "ours", "ours_negative", ["--graph-mode", "negative"]

    m = re.match(r"^ours_negative_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "negative_knn", "--knn-k", k]

    if v == "ours_uniform":
        return "ours", "ours_uniform", ["--graph-mode", "uniform"]

    m = re.match(r"^ours_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "random", "--knn-k", k]

    if v in ("ours_random", "ours_random_matched"):
        return "ours", v, ["--graph-mode", "random", "--knn-k", str(default_knn_k)]

    if v == "ours_threshold":
        return "ours", "ours_threshold", ["--graph-mode", "threshold"]

    if v == "ours_no_ema":
        return "ours", "ours_no_ema", ["--graph-mode", "dense", "--use-ema-graph", "false"]

    if v == "ours_fixed_tau":
        return ("ours", "ours_fixed_tau", ["--graph-mode", "dense", "--disable-adaptive-tau", "true"])

    custom_prefix = "ours_graph_mode_"
    if v.startswith(custom_prefix) and len(v) > len(custom_prefix):
        mode = v[len(custom_prefix):]
        return "ours", v, ["--graph-mode", mode]

    return None
