"""EMA helpers for GraphFL strategy state."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays


def update_client_update_ema(
    *,
    local_updates: List[NDArrays],
    cids: List[str],
    previous_updates: Optional[List[NDArrays]],
    previous_cids: Optional[List[str]],
    alpha: float,
) -> Tuple[List[NDArrays], str, List[NDArrays], List[str]]:
    alpha = min(max(float(alpha), 0.0), 1.0)
    if previous_updates is None or previous_cids != list(cids):
        ema_updates = [
            [np.array(arr, copy=True) for arr in update]
            for update in local_updates
        ]
        source = "initialized_current_update"
    else:
        ema_updates = []
        for old_update, current_update in zip(previous_updates, local_updates):
            ema_updates.append(
                [
                    alpha * old + (1.0 - alpha) * current
                    for old, current in zip(old_update, current_update)
                ]
            )
        source = "ema_update"

    stored_updates = [
        [np.array(arr, copy=True) for arr in update]
        for update in ema_updates
    ]
    return ema_updates, source, stored_updates, list(cids)


__all__ = ["update_client_update_ema"]
