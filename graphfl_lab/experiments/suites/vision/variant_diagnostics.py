"""Current diagnostic protocol variant parsing."""

from __future__ import annotations

import re
from typing import List, Tuple

from graphfl_lab.experiments.suites.vision.variant_helpers import (
    diagnostic_graph_args,
    diagnostic_graph_free_args,
)

ParsedVariant = Tuple[str, str, List[str]]


def parse_diagnostic_variant(v: str, default_knn_k: int) -> ParsedVariant | None:
    if v == "ours_real_graph":
        return (
            "ours",
            v,
            diagnostic_graph_args(
                correction_family="real_graph",
                knn_k=str(default_knn_k),
            ),
        )

    m = re.match(r"^ours_real_graph_k(\d+)$", v)
    if m:
        return (
            "ours",
            v,
            diagnostic_graph_args(
                correction_family="real_graph",
                knn_k=m.group(1),
            ),
        )

    m = re.match(r"^ours_(random|shuffled|uniform|identity)_control(?:_k(\d+))?$", v)
    if m:
        mode = m.group(1)
        k = m.group(2) or str(default_knn_k)
        return (
            "ours",
            v,
            diagnostic_graph_args(
                correction_family="control_graph",
                control_graph_mode=mode,
                knn_k=k,
            ),
        )

    m = re.match(r"^ours_cluster_only(?:_k(\d+))?$", v)
    if m:
        k = m.group(1) or str(default_knn_k)
        return (
            "ours",
            v,
            diagnostic_graph_args(
                correction_family="clustering_only",
                knn_k=k,
            ),
        )

    graph_free_modes = {
        "normclip": "norm_clip",
        "norm_clip": "norm_clip",
        "cap": "contribution_cap",
        "contribution_cap": "contribution_cap",
        "reweight": "dominance_reweight",
        "dominance_reweight": "dominance_reweight",
    }
    m = re.match(
        r"^ours_graphfree_(normclip|norm_clip|cap|contribution_cap|reweight|dominance_reweight)$",
        v,
    )
    if m:
        mode = graph_free_modes[m.group(1)]
        return "ours", v, diagnostic_graph_free_args(mode)

    return None
