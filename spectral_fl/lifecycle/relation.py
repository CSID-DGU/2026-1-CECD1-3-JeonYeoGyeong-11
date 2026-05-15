"""Relation estimators for lifecycle graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .context import RelationContext
from .modules import ModuleResult
from .traces import TraceRecord


@dataclass(frozen=True)
class RelationOutput:
    relation_matrix: np.ndarray
    relation_kind: str
    is_symmetric: bool = True
    is_directed: bool = False
    is_learned: bool = False
    raw_scores: np.ndarray | None = None
    normalized_scores: np.ndarray | None = None
    relation_meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.relation_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("relation_matrix must be square")
        object.__setattr__(self, "relation_matrix", matrix)
        object.__setattr__(self, "relation_kind", str(self.relation_kind))
        object.__setattr__(self, "relation_meta", dict(self.relation_meta))


def _vectors_from_state(client_state_output: Any) -> np.ndarray:
    if isinstance(client_state_output, np.ndarray):
        return np.asarray(client_state_output, dtype=np.float64)
    if hasattr(client_state_output, "vector_matrix"):
        return np.asarray(client_state_output.vector_matrix(), dtype=np.float64)
    payload = getattr(client_state_output, "payload", None)
    vectors = getattr(payload, "vectors", None)
    if vectors is not None:
        return np.vstack([np.asarray(vector, dtype=np.float64).reshape(1, -1) for vector in vectors])
    raise ValueError("client_state_output does not expose vectors")


def _pairwise_sq_dists(z_mat: np.ndarray) -> np.ndarray:
    norms = np.sum(z_mat * z_mat, axis=1, keepdims=True)
    d2 = np.maximum(norms + norms.T - 2.0 * (z_mat @ z_mat.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _cosine(z_mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z_mat, axis=1, keepdims=True)
    safe = z_mat / np.maximum(norms, 1e-12)
    sim = np.clip(safe @ safe.T, -1.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _positive_upper_values(matrix: np.ndarray) -> np.ndarray:
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return upper[upper > 1e-12]


def _resolve_distance_sigma(d2: np.ndarray, sigma: float) -> float:
    if float(sigma) > 0.0:
        return max(float(sigma), 1e-12)
    vals = _positive_upper_values(d2)
    if vals.size == 0:
        return 1.0
    return max(float(np.sqrt(np.median(vals))), 1e-12)


def _normalized_sample_weights(n: int, sample_weights: Sequence[float] | None) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if sample_weights is None:
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.size != n:
        raise ValueError(f"client_sample_weights has length {weights.size}; expected {n}")
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    return weights / total


def _project_simplex(v: np.ndarray) -> np.ndarray:
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


def _entropy(matrix: np.ndarray) -> float:
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    values = values[values > 0.0]
    if values.size <= 1:
        return 0.0
    p = values / (float(np.sum(values)) + 1e-12)
    return float(np.clip(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(values.size), 0.0, 1.0))


def _relation_stats(matrix: np.ndarray) -> dict[str, Any]:
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return {
        "relation_score_mean": float(np.mean(upper)) if upper.size else 0.0,
        "relation_score_std": float(np.std(upper)) if upper.size else 0.0,
        "relation_entropy": _entropy(np.maximum(matrix, 0.0)),
    }


class GraphRelationEstimator:
    def __init__(
        self,
        relation_kind: str = "positive_cosine",
        *,
        graph_scale_sigma: float = 1.0,
        learned_graph_lambda: float = 1.0,
        client_sample_weights: Sequence[float] | None = None,
    ) -> None:
        self.relation_kind = str(relation_kind)
        self.graph_scale_sigma = float(graph_scale_sigma)
        self.learned_graph_lambda = float(learned_graph_lambda)
        self.client_sample_weights = client_sample_weights

    def run(self, context: RelationContext) -> ModuleResult:
        try:
            output = estimate_relation_from_vectors(
                _vectors_from_state(context.client_state_output),
                relation_kind=self.relation_kind,
                graph_scale_sigma=self.graph_scale_sigma,
                learned_graph_lambda=self.learned_graph_lambda,
                client_sample_weights=self.client_sample_weights,
            )
        except Exception as exc:  # pragma: no cover - exercised through ModuleResult contract
            return ModuleResult.error(exc, support_level="core-supported")
        trace = relation_trace_record(output, round_number=context.round_context.server_round)
        return ModuleResult.ok(output=output, trace_records=trace, support_level=output.relation_meta.get("support_level", "core-supported"))


class UnsupportedRelationEstimator:
    def __init__(self, name: str, *, support_level: str = "interface-target", reason: str = "") -> None:
        self.name = str(name)
        self.support_level = str(support_level)
        self.reason = reason or f"relation estimator {self.name!r} is not executable yet"

    def run(self, context: RelationContext) -> ModuleResult:
        trace = TraceRecord(
            phase="relation",
            module="unsupported",
            name=self.name,
            round=context.round_context.server_round,
            values={
                "status": "unsupported",
                "support_level": self.support_level,
                "component_kind": "RelationEstimator",
                "component_name": self.name,
                "reason": self.reason,
            },
        )
        return ModuleResult.unsupported(
            support_level=self.support_level,
            message=self.reason,
            trace_records=trace,
        )


def estimate_relation_from_vectors(
    z_mat: np.ndarray,
    *,
    relation_kind: str = "positive_cosine",
    graph_scale_sigma: float = 1.0,
    learned_graph_lambda: float = 1.0,
    client_sample_weights: Sequence[float] | None = None,
) -> RelationOutput:
    z = np.asarray(z_mat, dtype=np.float64)
    kind = str(relation_kind).strip().lower().replace("-", "_")
    if z.ndim != 2:
        raise ValueError("z_mat must be a 2D matrix")
    if kind in {"positive_cosine", "cosine", "dense", "knn", "mutual_knn", "threshold", "random", "magnitude", "magnitude_aware", "global", "global_alignment"}:
        raw = _cosine(z)
        matrix = np.maximum(raw, 0.0)
        return RelationOutput(
            relation_matrix=matrix,
            relation_kind="cosine",
            raw_scores=raw,
            normalized_scores=matrix,
            relation_meta={"metric": "cosine", "state_vectors": z, "support_level": "core-supported"},
        )
    if kind in {"absolute_cosine", "signed_abs", "abs_cosine", "dense_abs", "signed_abs_knn", "abs_knn", "absolute_knn"}:
        raw = _cosine(z)
        matrix = np.abs(raw)
        np.fill_diagonal(matrix, 0.0)
        return RelationOutput(
            relation_matrix=matrix,
            relation_kind="signed_conflict",
            raw_scores=raw,
            normalized_scores=matrix,
            relation_meta={"metric": "absolute_cosine", "state_vectors": z, "support_level": "core-supported"},
        )
    if kind in {"negative_cosine", "negative", "anti", "anti_alignment", "negative_knn", "anti_knn", "anti_alignment_knn"}:
        raw = _cosine(z)
        matrix = np.maximum(-raw, 0.0)
        np.fill_diagonal(matrix, 0.0)
        return RelationOutput(
            relation_matrix=matrix,
            relation_kind="signed_conflict",
            raw_scores=raw,
            normalized_scores=matrix,
            relation_meta={"metric": "negative_cosine", "state_vectors": z, "support_level": "core-supported"},
        )
    if kind in {"rbf", "gaussian", "gaussian_rbf", "rbf_knn", "gaussian_knn", "gaussian_rbf_knn"}:
        d2 = _pairwise_sq_dists(z)
        sigma = _resolve_distance_sigma(d2, graph_scale_sigma)
        matrix = np.exp(-d2 / (2.0 * sigma * sigma))
        np.fill_diagonal(matrix, 0.0)
        return RelationOutput(
            relation_matrix=matrix,
            relation_kind="rbf",
            raw_scores=d2,
            normalized_scores=matrix,
            relation_meta={"metric": "rbf", "temperature": sigma, "state_vectors": z, "support_level": "core-supported"},
        )
    if kind in {"learned_smooth", "smooth_graph", "learned", "learned_smooth_knn", "smooth_graph_knn", "learned_knn"}:
        d2 = _pairwise_sq_dists(z)
        return RelationOutput(
            relation_matrix=d2,
            relation_kind="euclidean_distance",
            raw_scores=d2,
            relation_meta={
                "metric": "squared_euclidean",
                "learned_graph_lambda": float(learned_graph_lambda),
                "state_vectors": z,
                "support_level": "proxy-supported",
            },
        )
    if kind in {"pfedgraph_qp", "pfedgraph_simplex", "collaboration_qp", "qp_collaboration"}:
        n = int(z.shape[0])
        p = _normalized_sample_weights(n, client_sample_weights)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z_safe = z / np.maximum(norms, 1e-12)
        cosine = np.clip(z_safe @ z_safe.T, -1.0, 1.0)
        difference = -cosine
        difference[difference < -0.9] = -1.0
        lam = max(float(learned_graph_lambda), 1e-12)
        directed = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            row = _project_simplex(p - difference[i] / (2.0 * lam))
            row[i] = 0.0
            row_sum = float(np.sum(row))
            if row_sum > 1e-12:
                row = row / row_sum
            elif n > 1:
                row = np.full(n, 1.0 / float(n - 1), dtype=np.float64)
                row[i] = 0.0
            directed[i] = row
        return RelationOutput(
            relation_matrix=directed,
            relation_kind="qp_collaboration",
            is_symmetric=False,
            is_directed=True,
            raw_scores=difference,
            normalized_scores=directed,
            relation_meta={
                "metric": "cosine_difference_qp",
                "prior_used": True,
                "sample_prior": p,
                "sample_prior_entropy": _sample_prior_entropy(p),
                "state_vectors": z,
                "learned_graph_lambda": float(learned_graph_lambda),
                "support_level": "proxy-supported",
            },
        )
    raise ValueError(f"Unknown relation kind: {relation_kind}")


def _sample_prior_entropy(prior: np.ndarray) -> float:
    if prior.size <= 1:
        return 0.0
    p = prior / (float(np.sum(prior)) + 1e-12)
    return float(np.clip(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(prior.size), 0.0, 1.0))


def relation_kind_for_graph_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    if key in {"signed_abs", "absolute", "abs_cosine", "dense_abs", "signed_abs_knn", "abs_knn", "absolute_knn"}:
        return "absolute_cosine"
    if key in {"negative", "negative_cosine", "anti", "anti_alignment", "negative_knn", "anti_knn", "anti_alignment_knn"}:
        return "negative_cosine"
    if key in {"rbf", "gaussian", "gaussian_rbf", "rbf_knn", "gaussian_knn", "gaussian_rbf_knn"}:
        return "rbf"
    if key in {"learned_smooth", "smooth_graph", "learned", "learned_smooth_knn", "smooth_graph_knn", "learned_knn"}:
        return "learned_smooth"
    if key in {"pfedgraph_qp", "pfedgraph_simplex", "collaboration_qp"}:
        return "pfedgraph_qp"
    return "positive_cosine"


def relation_trace_record(output: RelationOutput, *, round_number: int | None = None) -> TraceRecord:
    values = {
        "status": "ok",
        "support_level": output.relation_meta.get("support_level", "core-supported"),
        "component_kind": "RelationEstimator",
        "component_name": output.relation_kind,
        "relation_kind": output.relation_kind,
        "is_symmetric": output.is_symmetric,
        "is_directed": output.is_directed,
        "is_learned": output.is_learned,
        "uses_sample_prior": bool(output.relation_meta.get("prior_used", False)),
    }
    if "sample_prior_entropy" in output.relation_meta:
        values["sample_prior_entropy"] = output.relation_meta["sample_prior_entropy"]
    values.update(_relation_stats(np.maximum(output.relation_matrix, 0.0)))
    return TraceRecord(
        phase="relation",
        module="relation_estimator",
        name=output.relation_kind,
        round=round_number,
        values=values,
    )


__all__ = [
    "GraphRelationEstimator",
    "RelationOutput",
    "UnsupportedRelationEstimator",
    "estimate_relation_from_vectors",
    "relation_kind_for_graph_mode",
    "relation_trace_record",
]
