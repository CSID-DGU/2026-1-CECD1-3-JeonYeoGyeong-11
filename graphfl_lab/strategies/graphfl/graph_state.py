"""Graph state transition helpers for GraphFL strategies."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def select_round_graph(
    *,
    current_graph: np.ndarray,
    previous_graph_ema: Optional[np.ndarray],
    use_ema_graph: bool,
    in_warmup: bool,
    ema_alpha: float,
) -> Tuple[np.ndarray, str]:
    if use_ema_graph:
        if previous_graph_ema is None or in_warmup:
            graph = current_graph
        else:
            graph = float(ema_alpha) * previous_graph_ema + (
                1.0 - float(ema_alpha)
            ) * current_graph
    else:
        graph = current_graph

    if use_ema_graph and in_warmup:
        source = "warmup_current_graph"
    elif use_ema_graph:
        source = "ema_graph"
    else:
        source = "raw_current_graph"
    return graph, source


__all__ = ["select_round_graph"]
