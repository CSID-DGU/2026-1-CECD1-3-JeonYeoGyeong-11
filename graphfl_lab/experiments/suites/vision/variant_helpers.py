"""Small helpers shared by vision suite variant parsing."""

from __future__ import annotations

from pathlib import Path
from typing import List


def token_float(value: str) -> str:
    """Convert compact variant floats such as ``0p01`` to CLI text ``0.01``."""
    return str(value).replace("p", ".")


def legacy_residual_reweight_args(graph_mode: str, knn_k: str = "") -> List[str]:
    """Return CLI args for the pre-low-pass residual reweighting path.

    This preserves the earlier behavior where the graph low-pass filter was
    used to score projected-update residuals, then raw update deltas were
    aggregated with shaped client weights.
    """
    out = [
        "--graph-source",
        "update",
        "--aggregation-target",
        "update",
        "--graph-mode",
        graph_mode,
        "--conflict-mix",
        "0.2",
        "--min-client-weight",
        "0.0",
    ]
    if knn_k:
        out += ["--knn-k", knn_k]
    return out


def diagnostic_graph_args(
    *,
    correction_family: str,
    knn_k: str,
    control_graph_mode: str = "random",
    cluster_method: str = "kmeans",
) -> List[str]:
    """Return the current diagnostic protocol graph-correction args."""
    out = [
        "--graph-source",
        "classifier_head_update",
        "--aggregation-target",
        "graph_filtered_update",
        "--graph-mode",
        "knn",
        "--knn-k",
        str(knn_k),
        "--correction-family",
        str(correction_family),
        "--control-graph-mode",
        str(control_graph_mode),
    ]
    if correction_family == "clustering_only":
        out += [
            "--cluster-method",
            str(cluster_method),
            "--cluster-auto-k",
            "true",
        ]
    return out


def diagnostic_graph_free_args(mode: str) -> List[str]:
    """Return graph-free correction args for attribution controls."""
    out = [
        "--aggregation-target",
        "update",
        "--correction-family",
        "graph_free",
        "--graph-free-mode",
        str(mode),
        "--conflict-mix",
        "0.0",
        "--min-client-weight",
        "0.0",
    ]
    if mode == "contribution_cap":
        out += ["--contribution-cap", "0.35"]
    if mode == "norm_clip":
        out += ["--clip-quantile", "0.9"]
    if mode == "dominance_reweight":
        out += ["--graph-free-gamma", "1.0"]
    return out


def result_path_for_variant(out_dir: Path, method: str, seed: int, run_tag: str) -> Path:
    return out_dir / f"result_general_{method}_seed{seed}_{run_tag}.json"
