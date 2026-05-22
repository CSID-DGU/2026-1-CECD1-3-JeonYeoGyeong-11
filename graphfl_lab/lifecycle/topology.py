"""Topology operators for lifecycle graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .context import TopologyContext
from .modules import ModuleResult
from .relation import RelationOutput
from .traces import TraceRecord


@dataclass(frozen=True)
class TopologyOutput:
    adjacency: np.ndarray
    graph_kind: str
    is_directed: bool = False
    is_weighted: bool = True
    is_dynamic: bool = False
    is_layerwise: bool = False
    cluster_ids: np.ndarray | None = None
    masks: Any | None = None
    layerwise_adjacency: Mapping[str, np.ndarray] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        adjacency = np.asarray(self.adjacency, dtype=np.float64)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "graph_kind", str(self.graph_kind))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _zero_diag(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(out, 0.0)
    return out


def _sym(matrix: np.ndarray) -> np.ndarray:
    return _zero_diag(0.5 * (matrix + matrix.T))


def _keep_topk(w: np.ndarray, k: int) -> np.ndarray:
    n = w.shape[0]
    if k <= 0 or n <= 1:
        return np.zeros_like(w, dtype=np.float64)
    out = np.zeros_like(w, dtype=np.float64)
    for i in range(n):
        row = w[i].copy()
        row[i] = 0.0
        positive = np.flatnonzero(row > 0.0)
        if positive.size == 0:
            continue
        chosen = positive[np.argsort(row[positive])[-min(k, positive.size):]]
        out[i, chosen] = row[chosen]
    return _sym(np.maximum(out, out.T))


def _keep_mutual_topk(w: np.ndarray, k: int) -> np.ndarray:
    n = w.shape[0]
    directed = np.zeros_like(w, dtype=np.float64)
    for i in range(n):
        row = w[i].copy()
        row[i] = 0.0
        positive = np.flatnonzero(row > 0.0)
        if positive.size == 0:
            continue
        chosen = positive[np.argsort(row[positive])[-min(k, positive.size):]]
        directed[i, chosen] = row[chosen]
    return _zero_diag(np.minimum(directed, directed.T))


def _keep_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    out = np.where(w > float(threshold), w, 0.0)
    return _sym(out)


def _uniform_graph(n: int) -> np.ndarray:
    w = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(w, 0.0)
    return w


def _random_edges_matched_to_knn(base: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    reference = _keep_topk(base, k)
    n = base.shape[0]
    edge_count = int(np.sum(reference[np.triu_indices(n, k=1)] > 0.0))
    if edge_count <= 0 or n <= 1:
        return np.zeros_like(base, dtype=np.float64)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    out = np.zeros_like(base, dtype=np.float64)
    for i, j in pairs[: min(edge_count, len(pairs))]:
        out[i, j] = out[j, i] = 1.0
    return out


def _magnitude_aware(z_mat: np.ndarray, base: np.ndarray, sigma: float) -> np.ndarray:
    norms = np.linalg.norm(z_mat, axis=1).astype(np.float64) + 1e-12
    log_norms = np.log(norms)
    scale = np.exp(-np.abs(log_norms[:, None] - log_norms[None, :]) / max(float(sigma), 1e-12))
    return _zero_diag(base * scale)


def _global_alignment(z_mat: np.ndarray, base: np.ndarray) -> np.ndarray:
    center = np.mean(z_mat, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 1e-12:
        return base
    norms = np.linalg.norm(z_mat, axis=1) + 1e-12
    align = np.maximum(0.0, (z_mat @ center) / (norms * center_norm))
    return _zero_diag(base * np.sqrt(np.outer(align, align)))


def _project_simplex(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    theta = float(cssv[-1] / float(v.size)) if not bool(np.any(cond)) else float(cssv[np.nonzero(cond)[0][-1]] / ind[np.nonzero(cond)[0][-1]])
    return np.maximum(v - theta, 0.0)


def _learned_smooth_from_d2(d2: np.ndarray, learned_lambda: float) -> np.ndarray:
    n = d2.shape[0]
    w = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return w
    upper = d2[np.triu_indices(n, k=1)]
    vals = upper[upper > 1e-12]
    dist_scale = max(float(np.median(vals)), 1e-12) if vals.size else 1.0
    scaled = d2 / dist_scale
    lam = max(float(learned_lambda), 1e-12)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w[i, mask] = _project_simplex(-scaled[i, mask] / lam)
    return _sym(w)


def _block_uniform(cluster_ids: np.ndarray, *, intra: float = 1.0, inter: float = 0.0) -> np.ndarray:
    ids = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    same = ids[:, None] == ids[None, :]
    out = np.where(same, float(intra), float(inter)).astype(np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def _diagnostics(w: np.ndarray) -> dict[str, Any]:
    n = w.shape[0]
    upper = w[np.triu_indices(n, k=1)]
    n_edges = int(np.sum(upper > 0.0))
    n_possible = max(int(n * (n - 1) // 2), 1)
    degrees = np.sum(w > 0.0, axis=1).astype(int)
    edge_weights = upper[upper > 0.0]
    if edge_weights.size <= 1:
        graph_entropy = 0.0
    else:
        p = edge_weights / (float(np.sum(edge_weights)) + 1e-12)
        graph_entropy = float(np.clip(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(edge_weights.size), 0.0, 1.0))
    row_sums = np.sum(w, axis=1, keepdims=True)
    row_probs = np.divide(w, np.maximum(row_sums, 1e-12), out=np.zeros_like(w), where=row_sums > 1e-12)
    row_entropy = []
    for row in row_probs:
        vals = row[row > 0.0]
        row_entropy.append(float(-np.sum(vals * np.log(np.maximum(vals, 1e-12))) / np.log(vals.size)) if vals.size > 1 else 0.0)
    return {
        "graph_density": float(n_edges / n_possible),
        "graph_entropy": graph_entropy,
        "degree_mean": float(np.mean(degrees)) if degrees.size else 0.0,
        "degree_min": int(np.min(degrees)) if degrees.size else 0,
        "degree_max": int(np.max(degrees)) if degrees.size else 0,
        "row_entropy_mean": float(np.mean(row_entropy)) if row_entropy else 0.0,
        "number_of_edges": n_edges,
    }


class GraphTopologyOperator:
    def __init__(
        self,
        mode: str = "dense",
        *,
        knn_k: int = 2,
        edge_threshold: float = 0.0,
        rng: np.random.Generator | None = None,
        graph_scale_sigma: float = 1.0,
        learned_graph_lambda: float = 1.0,
    ) -> None:
        self.mode = str(mode)
        self.knn_k = int(knn_k)
        self.edge_threshold = float(edge_threshold)
        self.rng = rng
        self.graph_scale_sigma = float(graph_scale_sigma)
        self.learned_graph_lambda = float(learned_graph_lambda)

    def run(self, context: TopologyContext) -> ModuleResult:
        try:
            output = build_topology_from_relation(
                context.relation_output,
                mode=self.mode,
                knn_k=self.knn_k,
                edge_threshold=self.edge_threshold,
                rng=self.rng,
                graph_scale_sigma=self.graph_scale_sigma,
                learned_graph_lambda=self.learned_graph_lambda,
            )
        except Exception as exc:  # pragma: no cover - exercised through ModuleResult contract
            return ModuleResult.error(exc, support_level="core-supported")
        trace = topology_trace_record(output, round_number=context.round_context.server_round)
        return ModuleResult.ok(output=output, trace_records=trace, support_level=output.metadata.get("support_level", "core-supported"))


class ClusterBlockTopologyOperator:
    def __init__(self, cluster_ids: Sequence[int], *, intra: float = 1.0, inter: float = 0.0) -> None:
        self.cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
        self.intra = float(intra)
        self.inter = float(inter)

    def run(self, context: TopologyContext) -> ModuleResult:
        adjacency = _block_uniform(self.cluster_ids, intra=self.intra, inter=self.inter)
        output = TopologyOutput(
            adjacency=adjacency,
            graph_kind="cluster_block",
            cluster_ids=self.cluster_ids,
            metadata={"topology_operator": "cluster_block", "support_level": "proxy-supported"},
        )
        trace = topology_trace_record(output, round_number=context.round_context.server_round)
        return ModuleResult.ok(output=output, support_level="proxy-supported", trace_records=trace)


class UnsupportedTopologyOperator:
    def __init__(self, name: str, *, support_level: str = "interface-target", reason: str = "") -> None:
        self.name = str(name)
        self.support_level = str(support_level)
        self.reason = reason or f"topology operator {self.name!r} is not executable yet"

    def run(self, context: TopologyContext) -> ModuleResult:
        trace = TraceRecord(
            phase="topology",
            module="unsupported",
            name=self.name,
            round=context.round_context.server_round,
            values={
                "status": "unsupported",
                "support_level": self.support_level,
                "component_kind": "TopologyOperator",
                "component_name": self.name,
                "reason": self.reason,
            },
        )
        return ModuleResult.unsupported(
            support_level=self.support_level,
            message=self.reason,
            trace_records=trace,
        )


def build_topology_from_relation(
    relation: RelationOutput,
    *,
    mode: str = "dense",
    knn_k: int = 2,
    edge_threshold: float = 0.0,
    rng: np.random.Generator | None = None,
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
) -> TopologyOutput:
    key = str(mode).strip().lower().replace("-", "_")
    base = _zero_diag(np.asarray(relation.relation_matrix, dtype=np.float64))
    z_mat = relation.relation_meta.get("state_vectors")
    if key == "uniform":
        adj = _uniform_graph(base.shape[0])
        graph_kind = "uniform"
    elif key in {"dense", "signed_abs", "absolute", "abs_cosine", "dense_abs", "negative", "negative_cosine", "anti", "anti_alignment", "rbf", "gaussian", "gaussian_rbf"}:
        adj = _sym(base)
        graph_kind = "dense"
    elif key in {"knn", "signed_abs_knn", "abs_knn", "absolute_knn", "negative_knn", "anti_knn", "anti_alignment_knn", "rbf_knn", "gaussian_knn", "gaussian_rbf_knn"}:
        adj = _keep_topk(base, int(knn_k))
        graph_kind = "knn"
    elif key == "mutual_knn":
        adj = _keep_mutual_topk(base, int(knn_k))
        graph_kind = "mutual_knn"
    elif key == "threshold":
        adj = _keep_threshold(base, float(edge_threshold))
        graph_kind = "threshold"
    elif key == "random":
        rng = np.random.default_rng(0) if rng is None else rng
        adj = _random_edges_matched_to_knn(base, int(knn_k), rng)
        graph_kind = "matched_random"
    elif key in {"magnitude", "magnitude_aware"}:
        if z_mat is None:
            raise ValueError("magnitude topology requires state vectors in relation_meta")
        adj = _magnitude_aware(np.asarray(z_mat, dtype=np.float64), base, graph_scale_sigma)
        graph_kind = "magnitude_aware"
    elif key in {"magnitude_knn", "magnitude_aware_knn"}:
        if z_mat is None:
            raise ValueError("magnitude topology requires state vectors in relation_meta")
        adj = _keep_topk(_magnitude_aware(np.asarray(z_mat, dtype=np.float64), base, graph_scale_sigma), int(knn_k))
        graph_kind = "magnitude_aware_knn"
    elif key in {"global", "global_alignment"}:
        if z_mat is None:
            raise ValueError("global alignment topology requires state vectors in relation_meta")
        adj = _global_alignment(np.asarray(z_mat, dtype=np.float64), base)
        graph_kind = "global_alignment"
    elif key in {"learned_smooth", "smooth_graph", "learned"}:
        adj = _learned_smooth_from_d2(base, learned_graph_lambda)
        graph_kind = "learned_smooth_proxy"
    elif key in {"learned_smooth_knn", "smooth_graph_knn", "learned_knn"}:
        adj = _keep_topk(_learned_smooth_from_d2(base, learned_graph_lambda), int(knn_k))
        graph_kind = "learned_smooth_proxy_knn"
    elif key in {"pfedgraph_qp", "pfedgraph_simplex", "collaboration_qp"}:
        adj = _sym(base)
        graph_kind = "pfedgraph_qp:symmetric_diagnostic_projection"
    else:
        raise ValueError(f"Unknown topology mode: {mode}")

    support_level = "proxy-supported" if ("proxy" in graph_kind or key in {"pfedgraph_qp", "pfedgraph_simplex", "collaboration_qp"}) else "core-supported"
    return TopologyOutput(
        adjacency=adj.astype(np.float64, copy=False),
        graph_kind=graph_kind,
        is_directed=False,
        is_weighted=True,
        metadata={
            "topology_operator": key,
            "support_level": support_level,
            **_diagnostics(adj),
        },
    )


def topology_trace_record(output: TopologyOutput, *, round_number: int | None = None) -> TraceRecord:
    values = {
        "status": "ok",
        "support_level": output.metadata.get("support_level", "core-supported"),
        "component_kind": "TopologyOperator",
        "component_name": output.metadata.get("topology_operator", output.graph_kind),
        "graph_kind": output.graph_kind,
        "is_directed": output.is_directed,
        "is_weighted": output.is_weighted,
        "is_dynamic": output.is_dynamic,
        "is_layerwise": output.is_layerwise,
    }
    values.update({key: value for key, value in output.metadata.items() if key in {"graph_density", "graph_entropy", "degree_mean", "degree_min", "degree_max", "row_entropy_mean", "number_of_edges"}})
    return TraceRecord(
        phase="topology",
        module="topology_operator",
        name=output.graph_kind,
        round=round_number,
        values=values,
    )


__all__ = [
    "ClusterBlockTopologyOperator",
    "GraphTopologyOperator",
    "TopologyOutput",
    "UnsupportedTopologyOperator",
    "build_topology_from_relation",
    "topology_trace_record",
]
