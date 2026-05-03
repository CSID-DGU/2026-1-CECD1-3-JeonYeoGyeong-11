"""Aggregation helper functions for spectral FL strategies."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from flwr.common import NDArrays


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


def weighted_average_by_alpha(local_updates: List[NDArrays], alphas: np.ndarray) -> NDArrays:
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


def apply_min_client_weight(alpha_norm: np.ndarray, min_w: float) -> np.ndarray:
    """Return normalized weights with a final per-client floor when feasible."""
    if min_w <= 0.0:
        return alpha_norm

    alpha = np.asarray(alpha_norm, dtype=np.float64)
    n = int(alpha.size)
    if n <= 0:
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
