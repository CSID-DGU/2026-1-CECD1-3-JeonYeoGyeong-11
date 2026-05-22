"""Side-effect-free counterfactual diagnostics for same-round artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from graphfl_lab.corrections.graph_free import resolve_graph_free_correction
from graphfl_lab.diagnostics.metrics import summarize_pre_post
from graphfl_lab.graph.diagnostics import compute_graph_diagnostics

from .counterfactuals import (
    CounterfactualResult,
    CounterfactualSpec,
    default_counterfactual_specs,
)
from .traces import TraceRecord


def _normalize(weights: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if total <= eps:
        return np.full(w.size, 1.0 / max(float(w.size), 1.0), dtype=np.float64)
    return w / total


def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    w = np.asarray(adjacency, dtype=np.float64)
    return np.diag(np.sum(w, axis=1)) - w


def _spectral_filter(z_mat: np.ndarray, adjacency: np.ndarray, filter_strength: float) -> np.ndarray:
    l_mat = _laplacian(adjacency)
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    lambda_max = float(max(np.max(eigvals), 1e-12)) if eigvals.size else 1.0
    base_gains = np.clip(1.0 - eigvals / lambda_max, 0.0, 1.0)
    strength = max(float(filter_strength), 0.0)
    gains = np.ones_like(base_gains) if strength <= 0.0 else np.power(base_gains, strength)
    return eigvecs @ (gains[:, None] * (eigvecs.T @ z_mat))


def _canonical_filtered_target(target: str) -> str:
    if target in {
        "spectral_filtered_update",
        "spectral_filtered_update_delta",
        "filtered_update",
        "filtered_update_delta",
        "graph_filtered_update",
    }:
        return "graph_filtered_update_delta"
    if target in {
        "spectral_filtered_ema_update",
        "spectral_filtered_client_ema_update_delta",
        "graph_filtered_ema_update",
    }:
        return "graph_filtered_client_ema_update_delta"
    if target in {
        "spectral_filtered_weight",
        "spectral_filtered_local_weight_delta",
        "filtered_weight",
        "filtered_local_weight_delta",
        "graph_filtered_weight",
    }:
        return "graph_filtered_local_weight_delta"
    return target


def _row_entropy_mean(matrix: np.ndarray) -> float:
    w = np.asarray(matrix, dtype=np.float64)
    row_sums = np.sum(w, axis=1, keepdims=True)
    probs = np.divide(w, np.maximum(row_sums, 1e-12), out=np.zeros_like(w), where=row_sums > 1e-12)
    entropies = []
    for row in probs:
        vals = row[row > 0.0]
        if vals.size <= 1:
            entropies.append(0.0)
        else:
            entropies.append(float(-np.sum(vals * np.log(np.maximum(vals, 1e-12))) / np.log(vals.size)))
    return float(np.mean(entropies)) if entropies else 0.0


@dataclass(frozen=True)
class MinimalAggregationOutput:
    weights_post: np.ndarray
    post_flat_updates: np.ndarray
    global_delta: np.ndarray
    metadata: Mapping[str, Any]


class MinimalAggregationAdapter:
    """Temporary aggregation bridge used only by counterfactual diagnostics."""

    def __init__(self, *, filter_strength: float = 1.0, dominance_gamma: float = 1.0) -> None:
        self.filter_strength = float(filter_strength)
        self.dominance_gamma = float(dominance_gamma)

    def run(
        self,
        *,
        flat_updates: np.ndarray,
        weights_pre: Sequence[float],
        adjacency: np.ndarray,
        aggregation_target: str,
        graph_free_mode: str = "none",
    ) -> MinimalAggregationOutput:
        updates = np.asarray(flat_updates, dtype=np.float64).copy()
        alpha = _normalize(weights_pre)
        target = str(aggregation_target).strip().lower().replace("-", "_")
        metadata: dict[str, Any] = {"aggregation_target": target}

        if target in {"global_update", "update", "update_delta"}:
            post_updates = updates
            weights_post = alpha
            target_used = "global_update"
        elif target in {
            "spectral_filtered_update",
            "spectral_filtered_update_delta",
            "spectral_filtered_ema_update",
            "spectral_filtered_client_ema_update_delta",
            "filtered_update",
            "filtered_update_delta",
            "graph_filtered_update",
            "graph_filtered_update_delta",
            "graph_filtered_ema_update",
            "graph_filtered_client_ema_update_delta",
            "spectral_filtered_weight",
            "spectral_filtered_local_weight_delta",
            "filtered_weight",
            "filtered_local_weight_delta",
            "graph_filtered_weight",
            "graph_filtered_local_weight_delta",
        }:
            post_updates = _spectral_filter(updates, np.asarray(adjacency, dtype=np.float64), self.filter_strength)
            weights_post = alpha
            target_used = _canonical_filtered_target(target)
        elif target in {"graphfree_dominance_reweight", "graph_free_dominance_reweight"}:
            norms = np.linalg.norm(updates, axis=1)
            weights_post, mode_used = resolve_graph_free_correction(
                alpha=alpha,
                mode=graph_free_mode or "dominance_reweight",
                n_examples=np.asarray(weights_pre, dtype=np.float64),
                update_norms=norms,
                gamma=self.dominance_gamma,
            )
            post_updates = updates
            target_used = mode_used
        elif target == "graphfree_norm_clip":
            norms = np.linalg.norm(updates, axis=1)
            weights_post, mode_used = resolve_graph_free_correction(
                alpha=alpha,
                mode="norm_clip",
                n_examples=np.asarray(weights_pre, dtype=np.float64),
                update_norms=norms,
            )
            post_updates = updates
            target_used = mode_used
        elif target == "graphfree_contribution_cap":
            weights_post, mode_used = resolve_graph_free_correction(
                alpha=alpha,
                mode="contribution_cap",
                n_examples=np.asarray(weights_pre, dtype=np.float64),
                contribution_cap=0.5,
            )
            post_updates = updates
            target_used = mode_used
        else:
            raise ValueError(f"Unsupported minimal aggregation target {aggregation_target!r}")

        global_delta = np.sum(weights_post[:, None] * post_updates, axis=0)
        metadata["aggregation_target_used"] = target_used
        return MinimalAggregationOutput(
            weights_post=weights_post,
            post_flat_updates=post_updates,
            global_delta=global_delta,
            metadata=metadata,
        )


def _matched_random(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(reference, dtype=np.float64)
    n = w.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    weights = np.sort(w[upper_idx][w[upper_idx] > 0.0])
    if weights.size == 0:
        return np.zeros_like(w)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    out = np.zeros_like(w)
    for (i, j), weight in zip(pairs, weights):
        out[i, j] = out[j, i] = float(weight)
    return out


def _shuffled(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(reference, dtype=np.float64)
    n = w.shape[0]
    perm = rng.permutation(n)
    return w[perm][:, perm].copy()


def _uniform(n: int) -> np.ndarray:
    out = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def _identity(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def _default_cluster_ids(flat_updates: np.ndarray) -> np.ndarray:
    n = int(flat_updates.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    order = np.argsort(np.linalg.norm(flat_updates, axis=1))
    labels = np.zeros(n, dtype=np.int64)
    labels[order[n // 2:]] = 1
    return labels


def _cluster_block(flat_updates: np.ndarray) -> np.ndarray:
    cluster_ids = _default_cluster_ids(flat_updates)
    same = cluster_ids[:, None] == cluster_ids[None, :]
    out = np.where(same, 1.0, 0.0).astype(np.float64)
    np.fill_diagonal(out, 0.0)
    return out


class CounterfactualDiagnosticRunner:
    def __init__(
        self,
        *,
        aggregation_adapter: MinimalAggregationAdapter | None = None,
        loo_enabled: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.aggregation_adapter = aggregation_adapter or MinimalAggregationAdapter()
        self.loo_enabled = bool(loo_enabled)
        self.rng = np.random.default_rng(0) if rng is None else rng

    def run(
        self,
        *,
        flat_updates: np.ndarray,
        weights_pre: Sequence[float],
        actual_adjacency: np.ndarray,
        specs: Sequence[CounterfactualSpec] | None = None,
        round_number: int | None = None,
    ) -> tuple[CounterfactualResult, ...]:
        updates = np.asarray(flat_updates, dtype=np.float64).copy()
        actual = np.asarray(actual_adjacency, dtype=np.float64).copy()
        selected = tuple(default_counterfactual_specs() if specs is None else specs)
        return tuple(
            self._run_one(
                spec,
                flat_updates=updates,
                weights_pre=weights_pre,
                actual_adjacency=actual,
                round_number=round_number,
            )
            for spec in selected
        )

    def _adjacency_for_spec(
        self,
        spec: CounterfactualSpec,
        *,
        flat_updates: np.ndarray,
        actual_adjacency: np.ndarray,
    ) -> np.ndarray:
        mode = spec.topology_mode
        n = int(actual_adjacency.shape[0])
        if spec.name == "actual" or mode == "actual":
            return actual_adjacency.copy()
        if mode == "matched_random":
            return _matched_random(actual_adjacency, self.rng)
        if mode == "shuffled":
            return _shuffled(actual_adjacency, self.rng)
        if mode == "uniform":
            return _uniform(n)
        if mode == "identity":
            return _identity(n)
        if mode in {"cluster_block", "clustering_only", "block_uniform"}:
            return _cluster_block(flat_updates)
        return actual_adjacency.copy()

    def _run_one(
        self,
        spec: CounterfactualSpec,
        *,
        flat_updates: np.ndarray,
        weights_pre: Sequence[float],
        actual_adjacency: np.ndarray,
        round_number: int | None,
    ) -> CounterfactualResult:
        try:
            adjacency = self._adjacency_for_spec(
                spec,
                flat_updates=flat_updates,
                actual_adjacency=actual_adjacency,
            )
            aggregation = self.aggregation_adapter.run(
                flat_updates=flat_updates,
                weights_pre=weights_pre,
                adjacency=adjacency,
                aggregation_target=spec.aggregation_target,
                graph_free_mode=spec.graph_free_mode,
            )
            metrics = self._metrics(
                spec,
                flat_updates=flat_updates,
                weights_pre=weights_pre,
                adjacency=adjacency,
                aggregation=aggregation,
            )
            trace = self._trace_record(spec, metrics=metrics, round_number=round_number)
            return CounterfactualResult(
                name=spec.name,
                adjacency=adjacency,
                weights_post=aggregation.weights_post,
                post_flat_updates=aggregation.post_flat_updates,
                metrics=metrics,
                trace_records=(trace,),
            )
        except Exception as exc:
            adjacency = np.asarray(actual_adjacency, dtype=np.float64).copy()
            weights = _normalize(weights_pre)
            metrics = {
                "counterfactual": spec.name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            trace = self._trace_record(spec, metrics=metrics, round_number=round_number)
            return CounterfactualResult(
                name=spec.name,
                adjacency=adjacency,
                weights_post=weights,
                post_flat_updates=np.asarray(flat_updates, dtype=np.float64).copy(),
                metrics=metrics,
                trace_records=(trace,),
            )

    def _metrics(
        self,
        spec: CounterfactualSpec,
        *,
        flat_updates: np.ndarray,
        weights_pre: Sequence[float],
        adjacency: np.ndarray,
        aggregation: MinimalAggregationOutput,
    ) -> dict[str, Any]:
        summary = summarize_pre_post(
            flat_updates=flat_updates,
            flat_updates_post=aggregation.post_flat_updates,
            weights_pre=weights_pre,
            weights_post=aggregation.weights_post,
            loo_enabled=self.loo_enabled,
        )
        graph_metrics = compute_graph_diagnostics(adjacency)
        metrics = {
            "counterfactual": spec.name,
            "status": "ok",
            **summary["round"],
            "graph_density": graph_metrics["graph_density"],
            "graph_entropy": graph_metrics["graph_entropy"],
            "alpha_matrix_entropy": _row_entropy_mean(adjacency),
            "graph_kind": spec.topology_mode,
            "graph_source": spec.graph_source,
            "aggregation_target": aggregation.metadata.get("aggregation_target", spec.aggregation_target),
            "aggregation_target_used": aggregation.metadata.get("aggregation_target_used", spec.aggregation_target),
        }
        return metrics

    def _trace_record(
        self,
        spec: CounterfactualSpec,
        *,
        metrics: Mapping[str, Any],
        round_number: int | None,
    ) -> TraceRecord:
        return TraceRecord(
            phase="counterfactual",
            module="diagnostic_runner",
            name=spec.name,
            round=round_number,
            values={
                "variant": spec.name,
                "status": metrics.get("status", "ok"),
                "support_level": "proxy-supported" if spec.name != "actual" else "core-supported",
                "correction_family": spec.correction_family,
                "topology_mode": spec.topology_mode,
                "aggregation_target": spec.aggregation_target,
                "metrics": dict(metrics),
            },
        )


__all__ = [
    "CounterfactualDiagnosticRunner",
    "MinimalAggregationAdapter",
    "MinimalAggregationOutput",
]
