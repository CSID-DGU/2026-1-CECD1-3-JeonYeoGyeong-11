"""Graph source selection for the FedSim-style baseline."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.graph.sources.config import GraphSourceConfig, normalize_key
from graphfl_lab.graph.sources.spectral import graph_vectors_for_spectral


def graph_vectors_for_fedsim(
    *,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    ema_updates: Optional[List[NDArrays]] = None,
    source: str = "update",
) -> Tuple[List[np.ndarray], str]:
    source_key = normalize_key(source)
    allowed = {
        "update",
        "delta",
        "update_delta",
        "pseudo_gradient",
        "pseudo_grad",
        "ema_update",
        "client_ema_update",
        "momentum_update",
        "momentum_smoothed_update",
        "temporal_update",
        "normalized_update",
        "normalized_delta",
        "normalized_ema_update",
        "ema_normalized_update",
        "normalized_client_ema_update",
        "client_ema_normalized_update",
        "weight",
        "weights",
        "model_weight",
        "model_weights",
        "state",
    }
    if source_key not in allowed:
        raise ValueError(
            "FedSim graph_source currently supports update, normalized_update, or weight; "
            f"got {source!r}"
        )
    return graph_vectors_for_spectral(
        local_weights=local_weights,
        local_updates=local_updates,
        ema_updates=ema_updates,
        config=GraphSourceConfig(source=source),
    )
