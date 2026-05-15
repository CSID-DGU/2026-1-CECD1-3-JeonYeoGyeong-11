"""Named client graph construction modes."""

from __future__ import annotations

from typing import Optional, Sequence

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
from spectral_fl.graph.controls import build_control_graph
from spectral_fl.graph.clustering import build_block_uniform_graph, cluster_clients
from spectral_fl.graph.registry import (
    GraphBuildContext,
    build_registered_graph,
    normalize_graph_mode,
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


def _normalized_sample_weights(
    n: int,
    sample_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if sample_weights is None:
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.size != n:
        raise ValueError(
            f"client_sample_weights has length {weights.size}; expected {n}"
        )
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    return weights / total


def pfedgraph_qp_graph(
    z_mat: np.ndarray,
    *,
    learned_lambda: float = 1.0,
    sample_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """pFedGraph-inspired collaboration graph.

    Official pFedGraph optimizes each collaboration row on the probability
    simplex with a cosine-difference term and a data-size prior. This function
    keeps that estimator, then projects it into the symmetric zero-diagonal
    adjacency expected by the current Laplacian-based diagnostic strategy.
    """
    n = int(z_mat.shape[0])
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    p = _normalized_sample_weights(n, sample_weights)
    norms = np.linalg.norm(z_mat, axis=1, keepdims=True)
    z_safe = z_mat / np.maximum(norms, 1e-12)
    cosine = np.clip(z_safe @ z_safe.T, -1.0, 1.0)
    difference = -cosine
    difference[difference < -0.9] = -1.0

    lam = max(float(learned_lambda), 1e-12)
    directed = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = project_simplex(p - difference[i] / (2.0 * lam))
        row[i] = 0.0
        row_sum = float(np.sum(row))
        if row_sum > 1e-12:
            row = row / row_sum
        else:
            row = np.full(n, 1.0 / float(n - 1), dtype=np.float64)
            row[i] = 0.0
        directed[i] = row

    w = 0.5 * (directed + directed.T)
    np.fill_diagonal(w, 0.0)
    return w


def _build_legacy_base_client_graph(
    z_mat: np.ndarray,
    mode: str = "dense",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
    client_sample_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Construct a symmetric non-negative client relation graph from updates."""
    mode_key = normalize_graph_mode(mode)
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
    if mode_key in {"pfedgraph_qp", "pfedgraph_simplex", "collaboration_qp"}:
        return pfedgraph_qp_graph(
            z_mat=z_mat,
            learned_lambda=learned_graph_lambda,
            sample_weights=client_sample_weights,
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


def _build_base_client_graph_with_meta(
    z_mat: np.ndarray,
    mode: str = "dense",
    graph_source: str = "unknown",
    aggregation_target: str = "unknown",
    correction_family: str = "real_graph",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
    client_sample_weights: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, dict]:
    """Construct a base graph, consulting registered plugin builders first."""
    context = GraphBuildContext(
        z_mat=z_mat,
        mode=mode,
        graph_source=str(graph_source),
        aggregation_target=str(aggregation_target),
        correction_family=str(correction_family),
        knn_k=int(knn_k),
        edge_threshold=float(edge_threshold),
        rng=rng,
        graph_scale_sigma=float(graph_scale_sigma),
        learned_graph_lambda=float(learned_graph_lambda),
        extras={"client_sample_weights": client_sample_weights},
    )
    registered = build_registered_graph(context)
    if registered is not None:
        return registered.adjacency.astype(np.float64, copy=False), dict(
            registered.metadata
        )

    from spectral_fl.lifecycle.relation import (
        estimate_relation_from_vectors,
        relation_kind_for_graph_mode,
        relation_trace_record,
    )
    from spectral_fl.lifecycle.topology import (
        build_topology_from_relation,
        topology_trace_record,
    )

    relation = estimate_relation_from_vectors(
        z_mat,
        relation_kind=relation_kind_for_graph_mode(mode),
        graph_scale_sigma=graph_scale_sigma,
        learned_graph_lambda=learned_graph_lambda,
        client_sample_weights=client_sample_weights,
    )
    topology = build_topology_from_relation(
        relation,
        mode=mode,
        knn_k=knn_k,
        edge_threshold=edge_threshold,
        rng=rng,
        graph_scale_sigma=graph_scale_sigma,
        learned_graph_lambda=learned_graph_lambda,
    )
    mode_key = normalize_graph_mode(mode)
    meta = {
        "base_graph_builder": "lifecycle",
        "base_graph_mode": mode_key,
        "base_relation_kind": relation.relation_kind,
        "base_topology_kind": topology.graph_kind,
        "lifecycle_trace": [
            relation_trace_record(relation).to_dict(),
            topology_trace_record(topology).to_dict(),
        ],
    }
    if mode_key in {"pfedgraph_qp", "pfedgraph_simplex", "collaboration_qp"}:
        meta["base_graph_kind"] = "pfedgraph_qp:symmetric_diagnostic_projection"
        meta["uses_client_sample_weights"] = True
    return topology.adjacency, meta


def _build_base_client_graph(
    z_mat: np.ndarray,
    mode: str = "dense",
    graph_source: str = "unknown",
    aggregation_target: str = "unknown",
    correction_family: str = "real_graph",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
    client_sample_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Backward-compatible base graph builder returning only adjacency."""
    adj, _ = _build_base_client_graph_with_meta(
        z_mat=z_mat,
        mode=mode,
        graph_source=graph_source,
        aggregation_target=aggregation_target,
        correction_family=correction_family,
        knn_k=knn_k,
        edge_threshold=edge_threshold,
        rng=rng,
        graph_scale_sigma=graph_scale_sigma,
        learned_graph_lambda=learned_graph_lambda,
        client_sample_weights=client_sample_weights,
    )
    return adj


def build_relation_graph(
    z_mat: np.ndarray,
    mode: str = "dense",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
    correction_family: str = "real_graph",
    control_graph_mode: str = "random",
    graph_source: str = "unknown",
    aggregation_target: str = "unknown",
    cluster_method: str = "none",
    cluster_k: int = 0,
    cluster_auto_k: bool = False,
    cluster_seed: int = 0,
    client_sample_weights: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, dict]:
    """Build relation graph and return (adjacency, metadata)."""
    base_graph, base_meta = _build_base_client_graph_with_meta(
        z_mat=z_mat,
        mode=mode,
        graph_source=graph_source,
        aggregation_target=aggregation_target,
        correction_family=correction_family,
        knn_k=knn_k,
        edge_threshold=edge_threshold,
        rng=rng,
        graph_scale_sigma=graph_scale_sigma,
        learned_graph_lambda=learned_graph_lambda,
        client_sample_weights=client_sample_weights,
    )
    fam = str(correction_family).strip().lower().replace("-", "_")
    meta = {
        "correction_family": fam,
        "graph_mode": str(mode),
        "graph_source": str(graph_source),
        "aggregation_target": str(aggregation_target),
        "control_graph_mode": str(control_graph_mode),
    }
    meta.update(base_meta)
    if fam == "control_graph":
        adj = build_control_graph(
            reference_adj=base_graph,
            control_mode=control_graph_mode,
            rng=rng,
        )
        meta["graph_kind"] = f"control:{str(control_graph_mode).strip().lower()}"
        return adj.astype(np.float64, copy=False), meta

    if fam == "clustering_only":
        cluster_ids = cluster_clients(
            z_mat=z_mat,
            method=cluster_method,
            k=int(cluster_k),
            seed=int(cluster_seed),
            auto_k=bool(cluster_auto_k),
        )
        same_mask = (cluster_ids[:, None] == cluster_ids[None, :]) & (~np.eye(cluster_ids.size, dtype=bool))
        intra_vals = base_graph[same_mask]
        intra = float(np.mean(intra_vals[intra_vals > 0.0])) if np.any(intra_vals > 0.0) else 1.0
        adj = build_block_uniform_graph(cluster_ids, intra=intra, inter=0.0)
        meta["graph_kind"] = "clustering_only:block_uniform"
        meta["cluster_method"] = str(cluster_method)
        meta["cluster_k"] = int(len(np.unique(cluster_ids)))
        meta["cluster_ids"] = [int(x) for x in cluster_ids.tolist()]
        return adj.astype(np.float64, copy=False), meta

    meta["graph_kind"] = "real_graph"
    return base_graph.astype(np.float64, copy=False), meta


def build_client_graph(
    z_mat: np.ndarray,
    mode: str = "dense",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
    correction_family: str = "real_graph",
    control_graph_mode: str = "random",
    graph_source: str = "unknown",
    aggregation_target: str = "unknown",
    cluster_method: str = "none",
    cluster_k: int = 0,
    cluster_auto_k: bool = False,
    cluster_seed: int = 0,
    client_sample_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Backward-compatible graph builder returning only adjacency."""
    adj, _ = build_relation_graph(
        z_mat=z_mat,
        mode=mode,
        knn_k=knn_k,
        edge_threshold=edge_threshold,
        rng=rng,
        graph_scale_sigma=graph_scale_sigma,
        learned_graph_lambda=learned_graph_lambda,
        correction_family=correction_family,
        control_graph_mode=control_graph_mode,
        graph_source=graph_source,
        aggregation_target=aggregation_target,
        cluster_method=cluster_method,
        cluster_k=cluster_k,
        cluster_auto_k=cluster_auto_k,
        cluster_seed=cluster_seed,
        client_sample_weights=client_sample_weights,
    )
    return adj
