"""Round-level spectral metric bundle for GraphFL strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from graphfl_lab.strategies.graphfl.diagnostics import (
    heterogeneity,
    spectral_energy_diagnostics,
)
from graphfl_lab.strategies.graphfl.filtering import laplacian


@dataclass(frozen=True)
class RoundSpectralMetrics:
    l_raw_current: np.ndarray
    l_curr: np.ndarray
    l_for_metric: np.ndarray
    metric_graph_source: str
    h_spec_raw_current: float
    h_spec_current: float
    h_spec: float
    metric_lambda_max: float
    h_spec_normalized: float
    h_spec_current_normalized: float
    h_spec_raw_current_normalized: float
    h_spec_ema_candidate: float
    spectral_diag: Dict[str, Any]


def compute_round_spectral_metrics(
    *,
    z_mat: np.ndarray,
    current_graph: np.ndarray,
    used_graph: np.ndarray,
    previous_laplacian: Optional[np.ndarray],
    previous_h_spec_ema: float,
) -> RoundSpectralMetrics:
    l_raw_current = laplacian(current_graph)
    l_curr = laplacian(used_graph)
    l_for_metric = previous_laplacian if previous_laplacian is not None else l_curr
    metric_graph_source = (
        "previous_round_graph"
        if previous_laplacian is not None
        else "current_round_graph"
    )
    h_spec_raw_current = heterogeneity(z_mat, l_raw_current)
    h_spec_current = heterogeneity(z_mat, l_curr)
    h_spec = heterogeneity(z_mat, l_for_metric)
    metric_eigvals = np.linalg.eigvalsh(l_for_metric)
    metric_lambda_max = float(max(np.max(metric_eigvals), 1e-12))
    h_spec_normalized = float(h_spec / (metric_lambda_max + 1e-12))
    h_spec_current_normalized = _normalized_h_spec(h_spec_current, l_curr)
    h_spec_raw_current_normalized = _normalized_h_spec(
        h_spec_raw_current,
        l_raw_current,
    )
    h_spec_ema_candidate = 0.9 * float(previous_h_spec_ema) + 0.1 * h_spec
    spectral_diag = spectral_energy_diagnostics(z_mat=z_mat, l_mat=l_curr)
    return RoundSpectralMetrics(
        l_raw_current=l_raw_current,
        l_curr=l_curr,
        l_for_metric=l_for_metric,
        metric_graph_source=metric_graph_source,
        h_spec_raw_current=float(h_spec_raw_current),
        h_spec_current=float(h_spec_current),
        h_spec=float(h_spec),
        metric_lambda_max=metric_lambda_max,
        h_spec_normalized=h_spec_normalized,
        h_spec_current_normalized=h_spec_current_normalized,
        h_spec_raw_current_normalized=h_spec_raw_current_normalized,
        h_spec_ema_candidate=float(h_spec_ema_candidate),
        spectral_diag=spectral_diag,
    )


def _normalized_h_spec(value: float, l_mat: np.ndarray) -> float:
    lambda_max = float(max(np.max(np.linalg.eigvalsh(l_mat)), 1e-12))
    return float(value / (lambda_max + 1e-12))


__all__ = ["RoundSpectralMetrics", "compute_round_spectral_metrics"]
