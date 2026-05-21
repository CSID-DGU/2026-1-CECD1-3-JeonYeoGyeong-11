"""Conflict and tau metric bundle for GraphFL strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from graphfl_lab.strategies.graphfl.aggregation import (
    TauSourceResolution,
    compute_conflict_weights,
    compute_tau,
    resolve_tau_source,
)
from graphfl_lab.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
    normalized_conflicts,
)


@dataclass(frozen=True)
class ConflictMetrics:
    z_tilde: np.ndarray
    filter_diag: Dict[str, Any]
    e: np.ndarray
    e_std_for_tau: float
    tau_source: TauSourceResolution
    tau: float
    e_z: np.ndarray
    conflict_weight: np.ndarray
    raw_cw: np.ndarray
    estd_disabled: bool
    e_mean: float
    e_std_raw: float


def compute_conflict_metric_bundle(
    *,
    z_mat: np.ndarray,
    l_mat: np.ndarray,
    filter_strength: float,
    tau_source_name: str,
    h_spec: float,
    h_spec_normalized: float,
    h_spec_ema: float,
    h_spec_ema_candidate: float,
    tau_signal_ema: float,
    tau_max: float,
    tau_gain: float,
    adaptive_tau: bool,
    fixed_tau: float,
    e_std_threshold: float,
) -> ConflictMetrics:
    z_tilde, filter_diag = apply_spectral_filter_with_diagnostics(
        z_mat=z_mat,
        l_mat=l_mat,
        filter_strength=filter_strength,
    )
    e = normalized_conflicts(z_mat=z_mat, z_tilde=z_tilde)
    e_std_for_tau = float(np.std(e))
    tau_source = resolve_tau_source(
        tau_source=tau_source_name,
        h_spec=h_spec,
        h_spec_normalized=h_spec_normalized,
        e_std=e_std_for_tau,
        h_spec_ema=h_spec_ema,
        h_spec_ema_candidate=h_spec_ema_candidate,
        tau_signal_ema=tau_signal_ema,
    )
    tau = compute_tau(
        h_spec_ema=tau_source.ema_value,
        tau_max=tau_max,
        tau_gain=tau_gain,
        adaptive=adaptive_tau,
        fixed_tau=fixed_tau,
    )
    e_z, conflict_weight, raw_cw, estd_disabled, e_mean, e_std_raw = (
        compute_conflict_weights(
            e=e,
            tau=tau,
            e_std_threshold=e_std_threshold,
        )
    )
    return ConflictMetrics(
        z_tilde=z_tilde,
        filter_diag=filter_diag,
        e=e,
        e_std_for_tau=e_std_for_tau,
        tau_source=tau_source,
        tau=tau,
        e_z=e_z,
        conflict_weight=conflict_weight,
        raw_cw=raw_cw,
        estd_disabled=bool(estd_disabled),
        e_mean=float(e_mean),
        e_std_raw=float(e_std_raw),
    )


__all__ = ["ConflictMetrics", "compute_conflict_metric_bundle"]
