"""Graph-free correction helpers for aggregation weights."""

from __future__ import annotations

import numpy as np


def _normalize(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s <= eps:
        return np.ones_like(w, dtype=np.float64) / max(float(w.size), 1.0)
    return w / s


def apply_norm_clip(
    alpha: np.ndarray,
    update_norms: np.ndarray,
    clip_quantile: float = 0.9,
    clip_norm: float = 0.0,
) -> np.ndarray:
    """Reduce weight on high-norm updates via clipping scale."""
    a = _normalize(np.asarray(alpha, dtype=np.float64))
    norms = np.asarray(update_norms, dtype=np.float64)
    if norms.size == 0:
        return a
    c = float(clip_norm)
    if c <= 0.0:
        q = min(max(float(clip_quantile), 0.0), 1.0)
        c = float(np.quantile(norms, q))
    scales = np.minimum(1.0, c / (norms + 1e-12))
    return _normalize(a * scales)


def compute_contribution_cap_weights(alpha: np.ndarray, cap: float) -> np.ndarray:
    """Apply per-client cap while preserving normalization."""
    a = _normalize(np.asarray(alpha, dtype=np.float64))
    c = float(cap)
    if c <= 0.0:
        return a
    n = int(a.size)
    if c * float(n) < 1.0 - 1e-12:
        return a
    out = np.zeros_like(a, dtype=np.float64)
    rem = np.arange(n, dtype=np.int64)
    rem_mass = 1.0
    while rem.size > 0:
        b = a[rem]
        b_sum = float(np.sum(b))
        if b_sum <= 1e-12:
            out[rem] = rem_mass / float(rem.size)
            break
        proposed = rem_mass * (b / b_sum)
        over = proposed > c + 1e-12
        if not bool(np.any(over)):
            out[rem] = proposed
            break
        over_idx = rem[over]
        out[over_idx] = c
        rem_mass -= c * float(over_idx.size)
        rem = rem[~over]
        if rem_mass <= 1e-12:
            break
    return _normalize(out)


def apply_dominance_reweight(
    alpha: np.ndarray,
    n_examples: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Downweight dominant sample-mass clients without graph structure."""
    a = _normalize(np.asarray(alpha, dtype=np.float64))
    p = _normalize(np.asarray(n_examples, dtype=np.float64))
    g = max(float(gamma), 0.0)
    return _normalize(a * np.power(np.clip(1.0 - p, 0.0, 1.0), g))


def resolve_graph_free_correction(
    *,
    alpha: np.ndarray,
    mode: str,
    n_examples: np.ndarray,
    update_norms: np.ndarray | None = None,
    clip_quantile: float = 0.9,
    contribution_cap: float = 0.0,
    gamma: float = 1.0,
) -> tuple[np.ndarray, str]:
    """Apply graph-free correction mode and return (weights, mode_used)."""
    key = str(mode).strip().lower().replace("-", "_")
    base = _normalize(np.asarray(alpha, dtype=np.float64))
    if key == "none":
        return base, "none"
    if key == "contribution_cap":
        return compute_contribution_cap_weights(base, float(contribution_cap)), key
    if key == "dominance_reweight":
        return apply_dominance_reweight(base, n_examples, gamma=float(gamma)), key
    if key == "norm_clip":
        if update_norms is None:
            return base, "norm_clip_no_update_norms"
        return apply_norm_clip(
            base,
            np.asarray(update_norms, dtype=np.float64),
            clip_quantile=float(clip_quantile),
        ), key
    return base, f"unknown_mode:{mode}"


__all__ = [
    "apply_dominance_reweight",
    "apply_norm_clip",
    "compute_contribution_cap_weights",
    "resolve_graph_free_correction",
]
