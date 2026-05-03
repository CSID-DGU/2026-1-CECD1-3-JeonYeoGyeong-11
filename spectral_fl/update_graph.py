"""Client update graph construction and graph-level diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def cosine_nonnegative(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    cos = float(np.dot(a, b) / denom)
    return max(0.0, cos)


def dense_positive_cosine(z_mat: np.ndarray) -> np.ndarray:
    """Pairwise max(0, cosine) similarity, zero diagonal, symmetric."""
    norms = np.linalg.norm(z_mat, axis=1) + 1e-12
    z_n = z_mat / norms[:, None]
    sim = z_n @ z_n.T
    sim = np.clip(sim, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    sim = 0.5 * (sim + sim.T)
    return sim.astype(np.float64)


def _knn_keep_topk(base: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k positive neighbors per row, then symmetrize by max."""
    n = base.shape[0]
    w = np.zeros_like(base)
    if k <= 0 or n <= 1:
        return w
    k_eff = min(int(k), n - 1)
    for i in range(n):
        sims = base[i].copy()
        sims[i] = -np.inf
        order = np.argsort(sims)[::-1]
        kept = 0
        for j in order:
            if kept >= k_eff:
                break
            v = float(base[i, j])
            if v <= 0.0:
                break
            w[i, j] = max(w[i, j], v)
            w[j, i] = max(w[j, i], v)
            kept += 1
    np.fill_diagonal(w, 0.0)
    return w


def _directed_topk_mask(base: np.ndarray, k: int) -> np.ndarray:
    """Return a directed mask for positive top-k neighbors per row."""
    n = base.shape[0]
    keep = np.zeros_like(base, dtype=bool)
    if k <= 0 or n <= 1:
        return keep
    k_eff = min(int(k), n - 1)
    for i in range(n):
        sims = base[i].copy()
        sims[i] = -np.inf
        order = np.argsort(sims)[::-1]
        kept = 0
        for j in order:
            if kept >= k_eff:
                break
            if float(base[i, j]) <= 0.0:
                break
            keep[i, j] = True
            kept += 1
    return keep


def _mutual_knn_keep_topk(base: np.ndarray, k: int) -> np.ndarray:
    """Keep edge i-j only when both endpoints choose each other in top-k."""
    keep = _directed_topk_mask(base, k)
    mutual = keep & keep.T
    w = np.where(mutual, base, 0.0)
    np.fill_diagonal(w, 0.0)
    return w


def _threshold_keep(base: np.ndarray, theta: float) -> np.ndarray:
    """Keep only edges whose similarity is greater than theta."""
    w = np.where(base > theta, base, 0.0)
    np.fill_diagonal(w, 0.0)
    w = 0.5 * (w + w.T)
    return w


def _uniform_graph(n: int) -> np.ndarray:
    """All off-diagonal entries are 1, diagonal is 0."""
    w = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(w, 0.0)
    return w


def _random_edges_with_edge_count(
    n: int, num_edges: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform random simple graph with exactly num_edges undirected edges."""
    w = np.zeros((n, n), dtype=np.float64)
    if n <= 1 or num_edges <= 0:
        return w
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ne = min(int(num_edges), len(pairs))
    rng.shuffle(pairs)
    for i, j in pairs[:ne]:
        w[i, j] = 1.0
        w[j, i] = 1.0
    return w


def _random_edges_matched_to_knn(
    base: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Random binary graph with the same undirected edge count as kNN(base, k)."""
    n = base.shape[0]
    w_knn = _knn_keep_topk(base, k)
    iu = np.triu_indices(n, k=1)
    n_edges = int(np.sum(w_knn[iu] > 0.0))
    return _random_edges_with_edge_count(n, n_edges, rng)


def _magnitude_aware_graph(z_mat: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Cosine graph down-weighted when client signal magnitudes differ."""
    norms = np.linalg.norm(z_mat, axis=1).astype(np.float64)
    n = z_mat.shape[0]
    scale = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            denom = max(float(norms[i]), float(norms[j]), 1e-12)
            ratio = min(float(norms[i]), float(norms[j])) / denom
            scale[i, j] = ratio
            scale[j, i] = ratio
    w = base * scale
    np.fill_diagonal(w, 0.0)
    return w


def _global_alignment_graph(z_mat: np.ndarray, base: np.ndarray) -> np.ndarray:
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


def build_client_graph(
    z_mat: np.ndarray,
    mode: str = "dense",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Construct a symmetric non-negative client relation graph from updates."""
    mode_key = str(mode).strip().lower().replace("-", "_")
    n = z_mat.shape[0]
    if mode_key == "uniform":
        return _uniform_graph(n)
    base = dense_positive_cosine(z_mat)
    if mode_key == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        return _random_edges_matched_to_knn(base=base, k=knn_k, rng=rng)
    if mode_key == "dense":
        return base
    if mode_key == "knn":
        return _knn_keep_topk(base, knn_k)
    if mode_key == "mutual_knn":
        return _mutual_knn_keep_topk(base, knn_k)
    if mode_key == "threshold":
        return _threshold_keep(base, edge_threshold)
    if mode_key in {"magnitude", "magnitude_aware"}:
        return _magnitude_aware_graph(z_mat=z_mat, base=base)
    if mode_key in {"global", "global_alignment"}:
        return _global_alignment_graph(z_mat=z_mat, base=base)
    raise ValueError(f"Unknown graph mode: {mode}")


def compute_graph_diagnostics(w: np.ndarray) -> Dict[str, Any]:
    """Return density, degree list, edge count, and emptiness flag."""
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    upper = w[iu]
    n_edges = int(np.sum(upper > 0.0))
    n_possible = max(int(n * (n - 1) // 2), 1)
    density = float(n_edges / n_possible)
    degrees = np.sum(w > 0.0, axis=1).astype(int).tolist()
    return {
        "graph_density": float(density),
        "graph_degree_list": [int(x) for x in degrees],
        "number_of_edges": int(n_edges),
        "graph_empty": bool(n_edges == 0),
    }
