"""State containers for the graph-FL diagnostic strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from flwr.common import NDArrays


@dataclass
class GraphFLStrategyState:
    w_ema: Optional[np.ndarray] = None
    l_prev: Optional[np.ndarray] = None
    h_spec_ema: float = 0.0
    tau_signal_ema: float = 0.0
    client_update_ema: Optional[List[NDArrays]] = None
    client_update_ema_cids: Optional[List[str]] = None


SpectralState = GraphFLStrategyState


__all__ = ["GraphFLStrategyState", "SpectralState"]
