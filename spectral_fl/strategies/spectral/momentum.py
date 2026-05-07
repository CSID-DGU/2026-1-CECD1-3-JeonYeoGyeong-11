"""Server-side momentum helpers for spectral strategy aggregation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from flwr.common import NDArrays


def apply_server_optimizer(
    *,
    current_global: Optional[NDArrays],
    candidate_global: NDArrays,
    server_learning_rate: float,
    server_momentum: float,
    server_momentum_vector: Optional[NDArrays],
) -> Tuple[NDArrays, Optional[NDArrays], Dict[str, Any]]:
    """Apply the existing FedAvgM-style server update to an aggregated model."""
    server_opt = (float(server_momentum) != 0.0) or (
        float(server_learning_rate) != 1.0
    )
    if not server_opt:
        return candidate_global, server_momentum_vector, {
            "server_optimizer": "none",
            "server_learning_rate": float(server_learning_rate),
            "server_momentum": float(server_momentum),
            "server_momentum_active": False,
        }
    if current_global is None:
        return candidate_global, server_momentum_vector, {
            "server_optimizer": "skipped_no_current_global",
            "server_learning_rate": float(server_learning_rate),
            "server_momentum": float(server_momentum),
            "server_momentum_active": False,
        }

    pseudo_gradient: NDArrays = [
        gp - cp for gp, cp in zip(current_global, candidate_global)
    ]
    next_momentum = server_momentum_vector
    if float(server_momentum) > 0.0:
        if next_momentum is None:
            next_momentum = [x.copy() for x in pseudo_gradient]
        else:
            next_momentum = [
                float(server_momentum) * old + grad
                for old, grad in zip(next_momentum, pseudo_gradient)
            ]
        step = next_momentum
        optimizer_name = "fedavgm_style_momentum"
    else:
        step = pseudo_gradient
        optimizer_name = "server_sgd"

    new_global = [
        gp - float(server_learning_rate) * grad
        for gp, grad in zip(current_global, step)
    ]
    candidate_delta_norm = float(
        np.sqrt(
            sum(
                float(np.sum((cp - gp) * (cp - gp)))
                for gp, cp in zip(current_global, candidate_global)
            )
        )
    )
    applied_delta_norm = float(
        np.sqrt(
            sum(
                float(np.sum((ng - gp) * (ng - gp)))
                for gp, ng in zip(current_global, new_global)
            )
        )
    )
    return new_global, next_momentum, {
        "server_optimizer": optimizer_name,
        "server_learning_rate": float(server_learning_rate),
        "server_momentum": float(server_momentum),
        "server_momentum_active": bool(float(server_momentum) > 0.0),
        "server_candidate_delta_norm": candidate_delta_norm,
        "server_applied_delta_norm": applied_delta_norm,
    }


__all__ = ["apply_server_optimizer"]
