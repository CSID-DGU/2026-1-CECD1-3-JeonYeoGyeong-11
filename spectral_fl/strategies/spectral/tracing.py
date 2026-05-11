"""Small tracing helpers for spectral strategy round logs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def matrix_log_if_small(
    matrix: np.ndarray,
    max_clients: int,
) -> Optional[List[List[float]]]:
    """Return a JSON-friendly matrix only when the client count is small."""
    if int(matrix.shape[0]) > int(max_clients):
        return None
    return [[float(v) for v in row] for row in matrix.tolist()]


def make_round_trace_payload(
    *,
    correction_family: str,
    control_graph_mode: str,
    graph_mode: str,
    alpha_mode: str,
    pre_post_round: Dict[str, Any],
) -> Dict[str, Any]:
    """Return normalized trace fields for correction-centric diagnostics."""
    return {
        "correction_family": str(correction_family),
        "control_graph_mode": str(control_graph_mode),
        "graph_mode_effective": str(graph_mode),
        "alpha_mode_effective": str(alpha_mode),
        "di_pre": float(pre_post_round.get("di_pre", np.nan)),
        "di_post": float(pre_post_round.get("di_post", np.nan)),
        "neff_pre": float(pre_post_round.get("neff_pre", np.nan)),
        "neff_post": float(pre_post_round.get("neff_post", np.nan)),
        "alignment_mean_pre": float(pre_post_round.get("align_mean_pre", np.nan)),
        "alignment_mean_post": float(pre_post_round.get("align_mean_post", np.nan)),
        "loo_mean_pre": float(pre_post_round.get("loo_mean_pre", np.nan)),
        "loo_mean_post": float(pre_post_round.get("loo_mean_post", np.nan)),
    }


__all__ = ["make_round_trace_payload", "matrix_log_if_small"]
