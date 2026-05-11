"""Aggregation math for the spectral conflict-aware strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays
from spectral_fl.corrections.graph_free import resolve_graph_free_correction


@dataclass(frozen=True)
class TauSourceResolution:
    source_used: str
    source_signal: float
    ema_value: float
    ema_candidate: float


@dataclass(frozen=True)
class AggregationWeightSelection:
    alpha_raw: np.ndarray
    alpha_norm: np.ndarray
    conflict_weight: np.ndarray
    alpha_mode: str
    active_client_mask: np.ndarray


def _norm_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def compute_tau(
    h_spec_ema: float,
    tau_max: float,
    tau_gain: float,
    adaptive: bool,
    fixed_tau: float,
) -> float:
    """Return adaptive or fixed conflict temperature."""
    if not adaptive:
        return float(fixed_tau)
    return float(tau_max) * float(np.tanh(float(tau_gain) * float(h_spec_ema)))


def resolve_tau_source(
    *,
    tau_source: str,
    h_spec: float,
    h_spec_normalized: float,
    e_std: float,
    h_spec_ema: float,
    h_spec_ema_candidate: float,
    tau_signal_ema: float,
) -> TauSourceResolution:
    """Resolve which diagnostic signal drives adaptive tau."""
    tau_source_key = _norm_key(tau_source)
    if tau_source_key in {"h_spec", "hspec", "heterogeneity"}:
        return TauSourceResolution(
            source_used="h_spec",
            source_signal=float(h_spec),
            ema_value=float(h_spec_ema),
            ema_candidate=float(h_spec_ema_candidate),
        )

    if tau_source_key in {
        "h_spec_normalized",
        "hspec_normalized",
        "normalized_h_spec",
        "normalized_hspec",
    }:
        source_signal = float(h_spec_normalized)
        source_used = "h_spec_normalized"
    elif tau_source_key in {"e_std", "std_e", "conflict_std"}:
        source_signal = float(e_std)
        source_used = "e_std"
    elif tau_source_key in {
        "h_spec_normalized_times_e_std",
        "normalized_hspec_times_e_std",
        "h_spec_norm_e_std",
        "norm_hspec_e_std",
    }:
        source_signal = float(h_spec_normalized * e_std)
        source_used = "h_spec_normalized_times_e_std"
    else:
        raise ValueError(
            "Unknown tau_source "
            f"{tau_source!r}; expected h_spec, h_spec_normalized, "
            "e_std, or h_spec_normalized_times_e_std"
        )

    ema_candidate = 0.9 * float(tau_signal_ema) + 0.1 * source_signal
    return TauSourceResolution(
        source_used=source_used,
        source_signal=source_signal,
        ema_value=float(tau_signal_ema),
        ema_candidate=float(ema_candidate),
    )


def compute_conflict_weights(
    e: np.ndarray,
    tau: float,
    e_std_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, float]:
    """Compute standardized residuals and conflict weights."""
    e_mean = float(np.mean(e))
    e_std_raw = float(np.std(e))
    disabled = bool(e_std_raw < float(e_std_threshold))
    e_z = (e - e_mean) / (e_std_raw + 1e-8)
    if disabled:
        return e_z, np.ones_like(e), np.ones_like(e), True, e_mean, e_std_raw
    e_excess = np.maximum(0.0, e_z)
    cw = np.exp(-float(tau) * e_excess)
    return e_z, cw, cw, False, e_mean, e_std_raw


def select_aggregation_weights(
    *,
    n_examples: np.ndarray,
    conflict_weight: np.ndarray,
    diagnostic_only: bool,
    in_warmup: bool,
    estd_disabled: bool,
    graph_fallback_used: bool,
    conflict_mix: float,
    min_client_weight: float,
) -> AggregationWeightSelection:
    """Select and normalize client aggregation weights for one server round."""
    n_examples_arr = np.asarray(n_examples, dtype=np.float64)
    conflict_weight_arr = np.asarray(conflict_weight, dtype=np.float64)

    if diagnostic_only:
        alpha_raw = n_examples_arr.copy()
        conflict_weight_arr = np.ones_like(conflict_weight_arr)
        alpha_mode = "diagnostic_only_fedavg"
    elif in_warmup:
        alpha_raw = n_examples_arr.copy()
        conflict_weight_arr = np.ones_like(conflict_weight_arr)
        alpha_mode = "warmup_fedavg"
    elif estd_disabled:
        alpha_raw = n_examples_arr.copy()
        conflict_weight_arr = np.ones_like(conflict_weight_arr)
        alpha_mode = "weak_conflict_skip"
    elif graph_fallback_used:
        alpha_raw = n_examples_arr.copy()
        conflict_weight_arr = np.ones_like(conflict_weight_arr)
        alpha_mode = "graph_empty_fedavg"
    elif float(conflict_mix) <= 0.0:
        alpha_raw = n_examples_arr.copy()
        conflict_weight_arr = np.ones_like(conflict_weight_arr)
        alpha_mode = "sample_weight_no_conflict"
    else:
        alpha_raw = n_examples_arr * (
            (1.0 - float(conflict_mix)) + float(conflict_mix) * conflict_weight_arr
        )
        alpha_mode = "conflict_aware"

    alpha_norm = alpha_raw / (float(np.sum(alpha_raw)) + 1e-12)
    active_client_mask = n_examples_arr > 0.0
    alpha_norm = apply_min_client_weight(
        alpha_norm,
        float(min_client_weight),
        active_mask=active_client_mask,
    )
    return AggregationWeightSelection(
        alpha_raw=alpha_raw,
        alpha_norm=alpha_norm,
        conflict_weight=conflict_weight_arr,
        alpha_mode=alpha_mode,
        active_client_mask=active_client_mask,
    )


def apply_correction_family(
    *,
    correction_family: str,
    selection: AggregationWeightSelection,
    n_examples: np.ndarray,
    graph_free_mode: str = "none",
    graph_free_gamma: float = 1.0,
    contribution_cap: float = 0.0,
    clip_quantile: float = 0.9,
    update_norms: Optional[np.ndarray] = None,
) -> AggregationWeightSelection:
    """Apply family-level post processing on aggregation weights.

    This function intentionally keeps graph-free behavior lightweight for now so
    strategy integration remains stable while the richer graph-free module is
    developed in follow-up PRs.
    """
    fam = _norm_key(correction_family)
    if fam in {"real_graph", "control_graph", "clustering_only"}:
        return selection
    if fam != "graph_free":
        return selection

    mode = _norm_key(graph_free_mode)
    alpha = np.asarray(selection.alpha_norm, dtype=np.float64).copy()
    active = np.asarray(selection.active_client_mask, dtype=bool)
    alpha, mode_used = resolve_graph_free_correction(
        alpha=alpha,
        mode=mode,
        n_examples=np.asarray(n_examples, dtype=np.float64),
        update_norms=update_norms,
        clip_quantile=float(clip_quantile),
        contribution_cap=float(contribution_cap),
        gamma=float(graph_free_gamma),
    )
    alpha_mode = f"{selection.alpha_mode}|graph_free:{mode_used}"
    alpha = apply_min_client_weight(alpha, 0.0, active_mask=active)
    return AggregationWeightSelection(
        alpha_raw=selection.alpha_raw,
        alpha_norm=alpha,
        conflict_weight=selection.conflict_weight,
        alpha_mode=alpha_mode,
        active_client_mask=active,
    )


def weighted_average_by_alpha(
    local_updates: List[NDArrays],
    alphas: np.ndarray,
) -> NDArrays:
    total = float(np.sum(alphas))
    norm_alpha = alphas / (total + 1e-12)
    out: NDArrays = [np.zeros_like(arr) for arr in local_updates[0]]
    for client_idx, update in enumerate(local_updates):
        for p_idx, p in enumerate(update):
            out[p_idx] += norm_alpha[client_idx] * p
    return out


def compute_entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized entropy of a non-negative weight vector, in [0, 1]."""
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s <= eps:
        return 0.0
    p = w / s
    n = p.size
    ent = float(-np.sum(p * np.log(p + eps)))
    return ent / float(np.log(max(n, 2)))


def compute_effective_clients(weights: np.ndarray, eps: float = 1e-12) -> float:
    """Effective number of clients, 1 / sum_i p_i^2."""
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s <= eps:
        return 0.0
    p = w / s
    return float(1.0 / float(np.sum(p * p) + eps))


def apply_min_client_weight(
    alpha_norm: np.ndarray,
    min_w: float,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return normalized weights with a final per-active-client floor when feasible."""
    alpha = np.asarray(alpha_norm, dtype=np.float64)
    n = int(alpha.size)
    if n <= 0:
        return alpha_norm

    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != alpha.shape:
            raise ValueError("active_mask must have the same shape as alpha_norm")
        if not bool(np.all(mask)):
            out = np.zeros_like(alpha, dtype=np.float64)
            if not bool(np.any(mask)):
                return out
            out[mask] = apply_min_client_weight(alpha[mask], min_w)
            return out

    if min_w <= 0.0:
        return alpha_norm

    s = float(np.sum(alpha))
    if s <= 0.0:
        return np.ones_like(alpha, dtype=np.float64) / float(n)

    floor = min(float(min_w), 1.0 / float(n))
    remaining = 1.0 - floor * float(n)
    if remaining <= 1e-12:
        return np.ones_like(alpha, dtype=np.float64) / float(n)

    p = alpha / s
    excess = np.maximum(p - floor, 0.0)
    excess_sum = float(np.sum(excess))
    if excess_sum <= 1e-12:
        return np.ones_like(alpha, dtype=np.float64) / float(n)
    return floor + remaining * (excess / excess_sum)


__all__ = [
    "apply_min_client_weight",
    "AggregationWeightSelection",
    "compute_conflict_weights",
    "compute_effective_clients",
    "compute_entropy",
    "compute_tau",
    "resolve_tau_source",
    "apply_correction_family",
    "select_aggregation_weights",
    "TauSourceResolution",
    "weighted_average_by_alpha",
]
