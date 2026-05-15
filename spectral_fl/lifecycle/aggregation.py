"""Aggregation operators for lifecycle graph-FL execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from spectral_fl.corrections.graph_free import resolve_graph_free_correction

from .context import AggregationContext
from .modules import ModuleResult
from .topology import TopologyOutput
from .traces import TraceRecord


@dataclass(frozen=True)
class AggregationResult:
    aggregation_target: str
    global_update: np.ndarray | None = None
    global_weights: list[np.ndarray] | None = None
    per_client_weights: list[list[np.ndarray]] | None = None
    alpha: np.ndarray | None = None
    alpha_matrix: np.ndarray | None = None
    cluster_ids: np.ndarray | None = None
    masks: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregation_target", str(self.aggregation_target))
        if self.global_update is not None:
            object.__setattr__(self, "global_update", np.asarray(self.global_update, dtype=np.float64))
        if self.alpha is not None:
            object.__setattr__(self, "alpha", np.asarray(self.alpha, dtype=np.float64))
        if self.alpha_matrix is not None:
            object.__setattr__(self, "alpha_matrix", np.asarray(self.alpha_matrix, dtype=np.float64))
        if self.cluster_ids is not None:
            object.__setattr__(self, "cluster_ids", np.asarray(self.cluster_ids, dtype=np.int64))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _normalize(weights: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= eps:
        return np.full(w.size, 1.0 / max(float(w.size), 1.0), dtype=np.float64)
    return w / total


def _as_matrix(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return np.vstack([np.asarray(value, dtype=np.float64).reshape(1, -1) for value in values])


def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    w = np.asarray(adjacency, dtype=np.float64)
    return np.diag(np.sum(w, axis=1)) - w


def _spectral_filter(z_mat: np.ndarray, adjacency: np.ndarray, filter_strength: float) -> np.ndarray:
    l_mat = _laplacian(adjacency)
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    lambda_max = float(max(np.max(eigvals), 1e-12)) if eigvals.size else 1.0
    base_gains = np.clip(1.0 - eigvals / lambda_max, 0.0, 1.0)
    gains = np.power(base_gains, max(float(filter_strength), 0.0))
    return eigvecs @ (gains[:, None] * (eigvecs.T @ z_mat))


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(matrix, dtype=np.float64), 0.0)
    row_sums = np.sum(w, axis=1, keepdims=True)
    return np.divide(w, np.maximum(row_sums, 1e-12), out=np.zeros_like(w), where=row_sums > 1e-12)


def _entropy(weights: np.ndarray) -> float:
    p = _normalize(weights)
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))))


def _contribution(alpha: np.ndarray, updates: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(updates, axis=1)
    raw = alpha * norms
    total = float(np.sum(raw))
    if total <= 1e-12:
        return np.full(alpha.size, 1.0 / max(float(alpha.size), 1.0), dtype=np.float64)
    return raw / total


class GraphAggregationOperator:
    def __init__(
        self,
        target: str = "global_update",
        *,
        filter_strength: float = 1.0,
        graph_free_mode: str = "none",
        dominance_gamma: float = 1.0,
        contribution_cap: float = 0.5,
    ) -> None:
        self.target = str(target)
        self.filter_strength = float(filter_strength)
        self.graph_free_mode = str(graph_free_mode)
        self.dominance_gamma = float(dominance_gamma)
        self.contribution_cap = float(contribution_cap)

    def run(self, context: AggregationContext) -> ModuleResult:
        target = self.target.strip().lower().replace("-", "_")
        try:
            result = self._run_core(context, target)
        except NotImplementedError as exc:
            trace = self._unsupported_trace(context, target, str(exc))
            return ModuleResult.unsupported(
                support_level="interface-target",
                message=str(exc),
                trace_records=trace,
            )
        except Exception as exc:  # pragma: no cover - error path is part of ModuleResult contract
            return ModuleResult.error(exc, support_level="core-supported")
        trace = aggregation_trace_record(result, round_number=context.round_context.server_round)
        return ModuleResult.ok(
            output=result,
            support_level=result.metadata.get("support_level", "core-supported"),
            trace_records=trace,
        )

    def _run_core(self, context: AggregationContext, target: str) -> AggregationResult:
        topology = context.topology_output
        adjacency = _adjacency(topology)
        updates = _as_matrix(context.local_updates)
        alpha = _normalize(context.num_examples)
        config = context.round_context.config
        metadata: dict[str, Any] = {"support_level": "core-supported"}

        if target in {"global_update", "update", "update_delta"}:
            post_updates = updates
            alpha_post = alpha
            target_name = "update"
        elif target in {"spectral_filtered_update", "filtered_update", "graph_filtered_update"}:
            post_updates = _spectral_filter(updates, adjacency, self.filter_strength)
            alpha_post = alpha
            target_name = "spectral_filtered_update"
        elif target in {
            "spectral_filtered_ema_update",
            "filtered_ema_update",
            "graph_filtered_ema_update",
        }:
            ema_updates = _as_matrix(config.get("ema_updates", context.local_updates))
            post_updates = _spectral_filter(ema_updates, adjacency, self.filter_strength)
            alpha_post = alpha
            target_name = "spectral_filtered_ema_update"
        elif target in {"weight", "weights"}:
            local_weights = _as_matrix(config["local_weights"])
            return AggregationResult(
                aggregation_target="weight",
                global_weights=[np.sum(alpha[:, None] * local_weights, axis=0)],
                alpha=alpha,
                metadata=metadata,
            )
        elif target in {
            "spectral_filtered_weight",
            "filtered_weight",
            "graph_filtered_weight",
        }:
            local_weights = _as_matrix(config["local_weights"])
            filtered = _spectral_filter(local_weights, adjacency, self.filter_strength)
            return AggregationResult(
                aggregation_target="spectral_filtered_weight",
                global_weights=[np.sum(alpha[:, None] * filtered, axis=0)],
                alpha=alpha,
                metadata=metadata,
            )
        elif target in {"graphfree_norm_clip", "graphfree_contribution_cap", "graphfree_dominance_reweight"}:
            mode = {
                "graphfree_norm_clip": "norm_clip",
                "graphfree_contribution_cap": "contribution_cap",
                "graphfree_dominance_reweight": "dominance_reweight",
            }[target]
            alpha_post, mode_used = resolve_graph_free_correction(
                alpha=alpha,
                mode=mode,
                n_examples=np.asarray(context.num_examples, dtype=np.float64),
                update_norms=np.linalg.norm(updates, axis=1),
                contribution_cap=self.contribution_cap,
                gamma=self.dominance_gamma,
            )
            post_updates = updates
            target_name = mode_used
        elif target in {"cluster_wise_update", "cluster_wise_weight"}:
            cluster_ids = getattr(topology, "cluster_ids", None)
            if cluster_ids is None:
                raise ValueError("cluster-wise aggregation requires topology_output.cluster_ids")
            metadata["support_level"] = "proxy-supported"
            post_updates = updates
            alpha_post = alpha
            target_name = target
            return AggregationResult(
                aggregation_target=target_name,
                global_update=np.sum(alpha_post[:, None] * post_updates, axis=0),
                alpha=alpha_post,
                cluster_ids=np.asarray(cluster_ids, dtype=np.int64),
                metadata=metadata,
            )
        elif target == "directed_neighbor_weight":
            alpha_matrix = _row_normalize(adjacency)
            return AggregationResult(
                aggregation_target="directed_neighbor_weight",
                alpha=alpha,
                alpha_matrix=alpha_matrix,
                metadata={"support_level": "proxy-supported"},
            )
        else:
            raise NotImplementedError(f"aggregation target {target!r} is an interface target")

        metadata["source_update_norms"] = np.linalg.norm(post_updates, axis=1)
        return AggregationResult(
            aggregation_target=target_name,
            global_update=np.sum(alpha_post[:, None] * post_updates, axis=0),
            alpha=alpha_post,
            metadata=metadata,
        )

    def _unsupported_trace(self, context: AggregationContext, target: str, reason: str) -> TraceRecord:
        return TraceRecord(
            phase="aggregation",
            module="aggregation_operator",
            name=target,
            round=context.round_context.server_round,
            values={
                "status": "unsupported",
                "support_level": "interface-target",
                "component_kind": "AggregationOperator",
                "component_name": target,
                "reason": reason,
            },
        )


def _adjacency(topology_output: Any) -> np.ndarray:
    if isinstance(topology_output, TopologyOutput):
        return topology_output.adjacency
    return np.asarray(getattr(topology_output, "adjacency", topology_output), dtype=np.float64)


def aggregation_trace_record(result: AggregationResult, *, round_number: int | None = None) -> TraceRecord:
    values: dict[str, Any] = {
        "status": "ok",
        "support_level": result.metadata.get("support_level", "core-supported"),
        "component_kind": "AggregationOperator",
        "component_name": result.aggregation_target,
        "aggregation_target": result.aggregation_target,
        "alpha": result.alpha,
        "alpha_matrix": result.alpha_matrix,
        "alpha_entropy": _entropy(result.alpha) if result.alpha is not None else 0.0,
    }
    if result.global_update is not None and result.alpha is not None:
        update_norms = result.metadata.get("source_update_norms")
        if update_norms is not None:
            raw = result.alpha * np.asarray(update_norms, dtype=np.float64)
            total = float(np.sum(raw))
            values["q_i"] = raw / total if total > 1e-12 else result.alpha
        values["pre_post_delta_norm"] = float(np.linalg.norm(result.global_update))
    return TraceRecord(
        phase="aggregation",
        module="aggregation_operator",
        name=result.aggregation_target,
        round=round_number,
        values=values,
    )


__all__ = [
    "AggregationResult",
    "GraphAggregationOperator",
    "aggregation_trace_record",
]
