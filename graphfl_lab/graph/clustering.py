"""Client clustering helpers for clustering-only graph controls."""

from __future__ import annotations

from typing import Optional

import numpy as np

from graphfl_lab.graph.similarity import dense_positive_cosine


def _resolve_cluster_k(num_clients: int, cluster_k: int, auto_k: bool) -> int:
    n = max(int(num_clients), 1)
    if int(cluster_k) > 0:
        return max(1, min(int(cluster_k), n))
    if bool(auto_k):
        return max(2, min(n, int(round(np.sqrt(n)))))
    return max(1, min(2, n))


def _kmeans_labels(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 30,
) -> np.ndarray:
    n = x.shape[0]
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int64)
    init_idx = rng.choice(np.arange(n, dtype=np.int64), size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cid in range(k):
            mask = labels == cid
            if bool(np.any(mask)):
                centers[cid] = np.mean(x[mask], axis=0)
            else:
                centers[cid] = x[int(rng.integers(0, n))]
    return labels


def cluster_clients(
    z_mat: np.ndarray,
    method: str = "kmeans",
    k: int = 0,
    seed: int = 0,
    auto_k: bool = False,
) -> np.ndarray:
    """Return integer cluster ids for each client vector."""
    z = np.asarray(z_mat, dtype=np.float64)
    n = int(z.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    k_resolved = _resolve_cluster_k(n, k, auto_k)
    key = str(method).strip().lower().replace("-", "_")

    if key in {"none", "kmeans"}:
        return _kmeans_labels(z, k_resolved, rng=rng)

    if key == "hierarchical":
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
        except Exception:
            return _kmeans_labels(z, k_resolved, rng=rng)
        z_center = z - np.mean(z, axis=0, keepdims=True)
        norms = np.linalg.norm(z_center, axis=1, keepdims=True) + 1e-12
        z_norm = z_center / norms
        tree = linkage(z_norm, method="average", metric="euclidean")
        labels = fcluster(tree, t=k_resolved, criterion="maxclust") - 1
        return labels.astype(np.int64)

    if key == "spectral":
        w = dense_positive_cosine(z)
        d = np.sum(w, axis=1)
        l = np.diag(d) - w
        _, eigvecs = np.linalg.eigh(l)
        emb = eigvecs[:, :k_resolved]
        return _kmeans_labels(emb, k_resolved, rng=rng)

    raise ValueError(f"Unknown cluster method: {method}")


def build_block_uniform_graph(
    cluster_ids: np.ndarray,
    intra: float = 1.0,
    inter: float = 0.0,
) -> np.ndarray:
    """Construct a block-uniform graph from cluster assignments."""
    c = np.asarray(cluster_ids, dtype=np.int64)
    n = int(c.size)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.full((n, n), float(inter), dtype=np.float64)
    for cid in np.unique(c):
        idx = np.where(c == cid)[0]
        out[np.ix_(idx, idx)] = float(intra)
    np.fill_diagonal(out, 0.0)
    out = 0.5 * (out + out.T)
    return out


__all__ = ["build_block_uniform_graph", "cluster_clients"]
