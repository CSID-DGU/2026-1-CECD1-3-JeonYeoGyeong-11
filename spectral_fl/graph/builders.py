"""Named client graph construction modes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from spectral_fl.graph.similarity import (
    dense_absolute_cosine,
    dense_negative_cosine,
    dense_positive_cosine,
    pairwise_sq_dists,
    positive_upper_values,
    resolve_distance_sigma,
)
from spectral_fl.graph.sparsification import (
    keep_mutual_topk,
    keep_threshold,
    keep_topk,
    random_edges_matched_to_knn,
    uniform_graph,
)


def rbf_graph(z_mat: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian/RBF graph from Euclidean distance in the chosen update space."""
    d2 = pairwise_sq_dists(z_mat)
    sig = resolve_distance_sigma(d2, sigma)
    w = np.exp(-d2 / (2.0 * sig * sig))
    np.fill_diagonal(w, 0.0)
    return w.astype(np.float64)


def magnitude_aware_graph(
    z_mat: np.ndarray,
    base: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Cosine graph down-weighted by log update-norm mismatch."""
    norms = np.linalg.norm(z_mat, axis=1).astype(np.float64) + 1e-12
    log_norms = np.log(norms)
    sig = max(float(sigma), 1e-12)
    scale = np.exp(-np.abs(log_norms[:, None] - log_norms[None, :]) / sig)
    w = base * scale
    np.fill_diagonal(w, 0.0)
    return w


def global_alignment_graph(z_mat: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Cosine graph weighted by each client's alignment with the mean signal."""
    center = np.mean(z_mat, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 1e-12:
        return base
    norms = np.linalg.norm(z_mat, axis=1) + 1e-12
    align = np.maximum(0.0, (z_mat @ center) / (norms * center_norm))
    pair_align = np.sqrt(np.outer(align, align))
    w = base * pair_align
    np.fill_diagonal(w, 0.0)
    return w


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    if not bool(np.any(cond)):
        theta = float(cssv[-1] / float(v.size))
    else:
        rho_idx = np.nonzero(cond)[0][-1]
        theta = float(cssv[rho_idx] / ind[rho_idx])
    return np.maximum(v - theta, 0.0)


def learned_smooth_graph(z_mat: np.ndarray, learned_lambda: float) -> np.ndarray:
    """Row-wise smoothness graph on a simplex.

    Each row solves a constrained smoothness surrogate,
    min_w sum_j w_j ||z_i-z_j||^2 + lambda/2 ||w||_2^2,
    with w_j >= 0, sum_j w_j = 1, and w_i = 0; rows are then symmetrized.
    """
    n = z_mat.shape[0]
    w = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return w
    d2 = pairwise_sq_dists(z_mat)
    scale_vals = positive_upper_values(d2)
    dist_scale = max(float(np.median(scale_vals)), 1e-12) if scale_vals.size else 1.0
    d2 = d2 / dist_scale
    lam = max(float(learned_lambda), 1e-12)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scores = -d2[i, mask] / lam
        w[i, mask] = project_simplex(scores)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    return w


def build_client_graph(
    z_mat: np.ndarray,
    mode: str = "dense",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
) -> np.ndarray:
    """Construct a symmetric non-negative client relation graph from updates."""
    mode_key = str(mode).strip().lower().replace("-", "_")
    n = z_mat.shape[0]
    if mode_key == "uniform":
        return uniform_graph(n)
    if mode_key in {"signed_abs", "absolute", "abs_cosine", "dense_abs"}:
        return dense_absolute_cosine(z_mat)
    if mode_key in {"signed_abs_knn", "abs_knn", "absolute_knn"}:
        return keep_topk(dense_absolute_cosine(z_mat), knn_k)
    if mode_key in {"negative", "negative_cosine", "anti", "anti_alignment"}:
        return dense_negative_cosine(z_mat)
    if mode_key in {"negative_knn", "anti_knn", "anti_alignment_knn"}:
        return keep_topk(dense_negative_cosine(z_mat), knn_k)
    if mode_key in {"rbf", "gaussian", "gaussian_rbf"}:
        return rbf_graph(z_mat, sigma=graph_scale_sigma)
    if mode_key in {"rbf_knn", "gaussian_knn", "gaussian_rbf_knn"}:
        return keep_topk(rbf_graph(z_mat, sigma=graph_scale_sigma), knn_k)
    if mode_key in {"learned_smooth", "smooth_graph", "learned"}:
        return learned_smooth_graph(
            z_mat=z_mat, learned_lambda=learned_graph_lambda
        )
    if mode_key in {"learned_smooth_knn", "smooth_graph_knn", "learned_knn"}:
        return keep_topk(
            learned_smooth_graph(
                z_mat=z_mat, learned_lambda=learned_graph_lambda
            ),
            knn_k,
        )
    base = dense_positive_cosine(z_mat)
    if mode_key == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        return random_edges_matched_to_knn(base=base, k=knn_k, rng=rng)
    if mode_key == "dense":
        return base
    if mode_key == "knn":
        return keep_topk(base, knn_k)
    if mode_key == "mutual_knn":
        return keep_mutual_topk(base, knn_k)
    if mode_key == "threshold":
        return keep_threshold(base, edge_threshold)
    if mode_key in {"magnitude", "magnitude_aware"}:
        return magnitude_aware_graph(
            z_mat=z_mat, base=base, sigma=graph_scale_sigma
        )
    if mode_key in {"magnitude_knn", "magnitude_aware_knn"}:
        return keep_topk(
            magnitude_aware_graph(
                z_mat=z_mat, base=base, sigma=graph_scale_sigma
            ),
            knn_k,
        )
    if mode_key in {"global", "global_alignment"}:
        return global_alignment_graph(z_mat=z_mat, base=base)
    raise ValueError(f"Unknown graph mode: {mode}")
