"""Core diagnostic metrics used by graph-gain attribution experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _as_prob(weights: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s <= eps:
        return np.ones_like(w, dtype=np.float64) / max(float(w.size), 1.0)
    return w / s


def compute_q(
    update_norms: Sequence[float],
    weights: Sequence[float],
    eps: float = 1e-12,
) -> np.ndarray:
    norms = np.asarray(update_norms, dtype=np.float64)
    p = _as_prob(weights, eps=eps)
    raw = p * norms
    return raw / (float(np.sum(raw)) + eps)


def compute_alignment(
    flat_updates: np.ndarray,
    global_delta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    g = np.asarray(flat_updates, dtype=np.float64)
    d = np.asarray(global_delta, dtype=np.float64)
    dn = float(np.linalg.norm(d))
    if dn <= eps:
        return np.zeros(g.shape[0], dtype=np.float64)
    gn = np.linalg.norm(g, axis=1) + eps
    return (g @ d) / (gn * dn)


def compute_dominance_index(q: Sequence[float]) -> float:
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.size == 0:
        return 0.0
    return float(np.max(q_arr))


def compute_effective_client_number(
    q: Sequence[float],
    eps: float = 1e-12,
) -> float:
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.size == 0:
        return 0.0
    return float(1.0 / (float(np.sum(q_arr * q_arr)) + eps))


def compute_loo_distortion(
    flat_updates: np.ndarray,
    weights: Sequence[float],
    global_delta: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    g = np.asarray(flat_updates, dtype=np.float64)
    p = _as_prob(weights, eps=eps)
    delta = np.asarray(global_delta, dtype=np.float64) if global_delta is not None else np.sum(
        p[:, None] * g, axis=0
    )
    dn = float(np.linalg.norm(delta))
    if dn <= eps:
        return np.zeros(g.shape[0], dtype=np.float64)

    out = np.zeros(g.shape[0], dtype=np.float64)
    for i in range(g.shape[0]):
        remain = 1.0 - float(p[i])
        if remain <= eps:
            out[i] = 0.0
            continue
        delta_minus = (delta - p[i] * g[i]) / remain
        dmn = float(np.linalg.norm(delta_minus))
        if dmn <= eps:
            out[i] = 1.0
            continue
        cos = float(np.dot(delta, delta_minus) / (dn * dmn + eps))
        out[i] = float(1.0 - cos)
    return out


def summarize_pre_post(
    *,
    flat_updates: np.ndarray,
    weights_pre: Sequence[float],
    weights_post: Sequence[float],
    loo_enabled: bool,
) -> Dict[str, object]:
    g = np.asarray(flat_updates, dtype=np.float64)
    p_pre = _as_prob(weights_pre)
    p_post = _as_prob(weights_post)
    norms = np.linalg.norm(g, axis=1)

    delta_pre = np.sum(p_pre[:, None] * g, axis=0)
    delta_post = np.sum(p_post[:, None] * g, axis=0)

    q_pre = compute_q(norms, p_pre)
    q_post = compute_q(norms, p_post)
    align_pre = compute_alignment(g, delta_pre)
    align_post = compute_alignment(g, delta_post)
    loo_pre = (
        compute_loo_distortion(g, p_pre, global_delta=delta_pre)
        if loo_enabled
        else np.zeros(g.shape[0], dtype=np.float64)
    )
    loo_post = (
        compute_loo_distortion(g, p_post, global_delta=delta_post)
        if loo_enabled
        else np.zeros(g.shape[0], dtype=np.float64)
    )

    alpha_entropy = float(-np.sum(p_post * np.log(np.maximum(p_post, 1e-12))))
    return {
        "q_pre": q_pre,
        "q_post": q_post,
        "align_pre": align_pre,
        "align_post": align_post,
        "loo_pre": loo_pre,
        "loo_post": loo_post,
        "round": {
            "di_pre": compute_dominance_index(q_pre),
            "di_post": compute_dominance_index(q_post),
            "neff_pre": compute_effective_client_number(q_pre),
            "neff_post": compute_effective_client_number(q_post),
            "align_mean_pre": float(np.mean(align_pre)),
            "align_mean_post": float(np.mean(align_post)),
            "loo_mean_pre": float(np.mean(loo_pre)),
            "loo_mean_post": float(np.mean(loo_post)),
            "alpha_entropy": alpha_entropy,
        },
        "norms": norms,
    }


__all__ = [
    "compute_alignment",
    "compute_dominance_index",
    "compute_effective_client_number",
    "compute_loo_distortion",
    "compute_q",
    "summarize_pre_post",
]
