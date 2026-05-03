"""Spectral client update graph strategy.

This module handles server-side orchestration for the current idea:

    1. server holds w_global
    2. for each client i, compute g_i = w_i - w_global
    3. compress to z_i via seeded Gaussian random projection
    4. build a client similarity graph W from the configured graph source
    5. compute Laplacian L_c, graph-update diagnostic H_spec, smoothed Z~,
       per-client conflict score e_i = ||z_i - z~_i|| / ||z_i||
    6. optionally convert spectral residuals into conservative weights
    7. w_global <- w_global + sum_i alpha_i * g_i        (NOT z_i)

Compressed z_i is used **only** for conflict measurement.  By default,
the actual update direction is the original full delta g_i.  Graph source,
aggregation target, graph mode, tau form, and conservative penalty options
are configurable via the strategy constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from spectral_fl.aggregation import (
    apply_min_client_weight,
    compute_conflict_weights,
    compute_effective_clients,
    compute_entropy,
    compute_tau,
    weighted_average_by_alpha,
)
from spectral_fl.projection import flatten_weights, make_gaussian_projection
from spectral_fl.spectral_diagnostics import (
    heterogeneity,
    laplacian,
    normalized_conflicts,
    spectral_energy_diagnostics,
    spectral_filter,
)
from spectral_fl.update_graph import build_client_graph, compute_graph_diagnostics


# =============================================================================
# Per-client eval tracing mixin.
# =============================================================================


class _EvalTracer:
    """Mixin providing eval_logs[round] = [{cid, accuracy, loss, num_examples}]."""

    def __init__(self) -> None:
        self.eval_logs: Dict[int, List[Dict[str, Any]]] = {}

    def _record_eval(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> None:
        per_client = []
        for proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            per_client.append(
                {
                    "cid": str(getattr(proxy, "cid", "?")),
                    "accuracy": float(metrics.get("accuracy", float("nan"))),
                    "loss": float(eval_res.loss),
                    "num_examples": int(eval_res.num_examples),
                }
            )
        self.eval_logs[int(server_round)] = per_client


class TracingFedAvg(_EvalTracer, fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


# =============================================================================
# Spectral client update graph strategy.
# =============================================================================


@dataclass
class SpectralState:
    w_ema: Optional[np.ndarray] = None
    l_prev: Optional[np.ndarray] = None
    h_spec_ema: float = 0.0


class SpectralConflictAwareStrategy(_EvalTracer, fl.server.strategy.FedAvg):
    def __init__(
        self,
        compression_dim: int = 256,
        compression_seed: int = 0,
        ema_alpha: float = 0.8,
        tau_gain: float = 2.0,
        tau_max: float = 2.0,
        conflict_mix: float = 0.5,
        warmup_rounds: int = 2,
        graph_mode: str = "dense",
        graph_source: str = "update",
        aggregation_target: str = "update",
        knn_k: int = 2,
        edge_threshold: float = 0.0,
        e_std_threshold: float = 0.0,
        graph_seed: int = 0,
        use_ema_graph: bool = True,
        adaptive_tau: bool = True,
        fixed_tau: float = 1.0,
        diagnostic_only: bool = False,
        min_client_weight: float = 0.0,
        log_w_matrix_max_clients: int = 20,
        **kwargs,
    ) -> None:
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        _EvalTracer.__init__(self)
        self.compression_dim = int(compression_dim)
        self.compression_seed = int(compression_seed)
        self.ema_alpha = float(ema_alpha)
        self.tau_gain = float(tau_gain)
        self.tau_max = float(tau_max)
        self.conflict_mix = float(conflict_mix)
        self.warmup_rounds = int(warmup_rounds)
        self.graph_mode = str(graph_mode)
        self.graph_source = str(graph_source)
        self.aggregation_target = str(aggregation_target)
        self.knn_k = int(knn_k)
        self.edge_threshold = float(edge_threshold)
        self.e_std_threshold = float(e_std_threshold)
        self.graph_seed = int(graph_seed)
        self.use_ema_graph = bool(use_ema_graph)
        self.adaptive_tau = bool(adaptive_tau)
        self.fixed_tau = float(fixed_tau)
        self.diagnostic_only = bool(diagnostic_only)
        self.min_client_weight = float(min_client_weight)
        self.log_w_matrix_max_clients = int(log_w_matrix_max_clients)
        self.state = SpectralState()
        self._current_global: Optional[NDArrays] = None
        self._proj_matrix: Optional[np.ndarray] = None
        self.round_logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ utils

    def _project(self, vec: np.ndarray) -> np.ndarray:
        if vec.size <= self.compression_dim:
            return vec.astype(np.float32, copy=False)
        if self._proj_matrix is None:
            self._proj_matrix = make_gaussian_projection(
                n_features=int(vec.size),
                n_dim=int(self.compression_dim),
                seed=int(self.compression_seed),
            )
        v = vec.astype(np.float32, copy=False)
        return v @ self._proj_matrix

    @staticmethod
    def _norm_key(value: str) -> str:
        return str(value).strip().lower().replace("-", "_")

    def _graph_vectors(
        self,
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
    ) -> Tuple[List[np.ndarray], str]:
        source = self._norm_key(self.graph_source)
        if source in {"update", "delta", "update_delta", "pseudo_gradient", "pseudo_grad"}:
            return [flatten_weights(g_i) for g_i in local_updates], "update_delta"
        if source in {"normalized_update", "normalized_delta"}:
            out = []
            for g_i in local_updates:
                flat = flatten_weights(g_i).astype(np.float32, copy=False)
                out.append(flat / (float(np.linalg.norm(flat)) + 1e-12))
            return out, "normalized_update_delta"
        if source in {"weight", "weights", "model_weight", "model_weights", "state"}:
            return [flatten_weights(w_i) for w_i in local_weights], "local_weight"
        raise ValueError(
            "Unknown graph_source "
            f"{self.graph_source!r}; expected update, normalized_update, or weight"
        )

    def _aggregate_target(
        self,
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
        alpha_norm: np.ndarray,
    ) -> Tuple[NDArrays, str]:
        target = self._norm_key(self.aggregation_target)
        if target in {"update", "delta", "update_delta"}:
            agg_delta = weighted_average_by_alpha(
                local_updates=local_updates, alphas=alpha_norm
            )
            return [gp + gd for gp, gd in zip(self._current_global, agg_delta)], "update_delta"
        if target in {"weight", "weights", "model_weight", "model_weights", "state"}:
            return (
                weighted_average_by_alpha(local_updates=local_weights, alphas=alpha_norm),
                "local_weight",
            )
        raise ValueError(
            "Unknown aggregation_target "
            f"{self.aggregation_target!r}; expected update or weight"
        )

    # ------------------------------------------------------------------ flwr

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._current_global = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        self._record_eval(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}
        if self._current_global is None:
            return super().aggregate_fit(server_round, results, failures)

        # ----------------- collect client local weights and metadata
        cids: List[str] = []
        local_weights: List[NDArrays] = []
        n_examples: List[int] = []
        client_metrics: List[Dict[str, Any]] = []
        for proxy, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            cids.append(str(metrics.get("cid", getattr(proxy, "cid", "?"))))
            local_weights.append(parameters_to_ndarrays(fit_res.parameters))
            n_examples.append(int(fit_res.num_examples))
            client_metrics.append(metrics)
        n_examples_arr = np.array(n_examples, dtype=np.float64)

        # ----------------- update space and projection
        local_updates: List[NDArrays] = [
            [lp - gp for lp, gp in zip(local, self._current_global)]
            for local in local_weights
        ]
        flat_deltas = [flatten_weights(g_i) for g_i in local_updates]
        delta_norms = np.array([float(np.linalg.norm(g)) for g in flat_deltas])
        flat_weights = [flatten_weights(w_i) for w_i in local_weights]
        weight_norms = np.array([float(np.linalg.norm(w)) for w in flat_weights])
        graph_vectors, graph_source_used = self._graph_vectors(
            local_weights=local_weights,
            local_updates=local_updates,
        )
        graph_source_norms = np.array([float(np.linalg.norm(g)) for g in graph_vectors])
        z_list = [self._project(g) for g in graph_vectors]
        z_mat = np.stack(z_list, axis=0)
        z_norms = np.array([float(np.linalg.norm(z)) for z in z_list])

        # ----------------- client similarity graph
        graph_rng = np.random.default_rng(
            int(self.graph_seed) * 1009 + int(server_round) * 13
        )
        w_curr = build_client_graph(
            z_mat=z_mat,
            mode=self.graph_mode,
            knn_k=self.knn_k,
            edge_threshold=self.edge_threshold,
            rng=graph_rng,
        )
        graph_diag_current = compute_graph_diagnostics(w_curr)
        if self.use_ema_graph:
            if self.state.w_ema is None:
                w_ema = w_curr
            else:
                w_ema = (
                    self.ema_alpha * self.state.w_ema
                    + (1.0 - self.ema_alpha) * w_curr
                )
        else:
            w_ema = w_curr

        graph_used_source = "ema_graph" if self.use_ema_graph else "raw_current_graph"
        graph_diag = compute_graph_diagnostics(w_ema)

        # Safety: if the graph is empty, fall back to FedAvg weights but still
        # log diagnostics.  This matches the spec ("don't crash; warn; record
        # graph_empty=true").
        graph_fallback_used = bool(graph_diag["graph_empty"])

        l_raw_current = laplacian(w_curr)
        l_curr = laplacian(w_ema)
        l_for_metric = self.state.l_prev if self.state.l_prev is not None else l_curr
        metric_graph_source = (
            "previous_round_graph" if self.state.l_prev is not None else "current_round_graph"
        )
        h_spec_raw_current = heterogeneity(z_mat, l_raw_current)
        h_spec_current = heterogeneity(z_mat, l_curr)
        h_spec = heterogeneity(z_mat, l_for_metric)
        self.state.h_spec_ema = 0.9 * self.state.h_spec_ema + 0.1 * h_spec
        spectral_diag = spectral_energy_diagnostics(z_mat=z_mat, l_mat=l_curr)

        # ----------------- spectral filtering and conflict score
        z_tilde = spectral_filter(z_mat, l_curr)
        e = normalized_conflicts(z_mat=z_mat, z_tilde=z_tilde)

        tau = compute_tau(
            h_spec_ema=self.state.h_spec_ema,
            tau_max=self.tau_max,
            tau_gain=self.tau_gain,
            adaptive=self.adaptive_tau,
            fixed_tau=self.fixed_tau,
        )

        e_z, conflict_weight, raw_cw, estd_disabled, e_mean_val, e_std_val_raw = (
            compute_conflict_weights(
                e=e, tau=tau, e_std_threshold=self.e_std_threshold
            )
        )

        # ----------------- aggregation weights
        if self.diagnostic_only:
            alpha_raw = n_examples_arr.copy()
            conflict_weight = np.ones_like(e)
            alpha_mode = "diagnostic_only_fedavg"
        elif server_round <= self.warmup_rounds:
            alpha_raw = n_examples_arr.copy()
            conflict_weight = np.ones_like(e)
            alpha_mode = "warmup_fedavg"
        elif estd_disabled:
            alpha_raw = n_examples_arr.copy()
            conflict_weight = np.ones_like(e)
            alpha_mode = "weak_conflict_skip"
        elif graph_fallback_used:
            alpha_raw = n_examples_arr.copy()
            conflict_weight = np.ones_like(e)
            alpha_mode = "graph_empty_fedavg"
        else:
            alpha_raw = n_examples_arr * (
                (1.0 - self.conflict_mix) + self.conflict_mix * conflict_weight
            )
            alpha_mode = "conflict_aware"

        # Normalize
        alpha_norm = alpha_raw / (float(np.sum(alpha_raw)) + 1e-12)
        # Optional: enforce minimum per-client weight (off by default).
        alpha_norm = apply_min_client_weight(alpha_norm, self.min_client_weight)

        # ----------------- aggregate configured target and apply
        new_global, aggregation_target_used = self._aggregate_target(
            local_weights=local_weights,
            local_updates=local_updates,
            alpha_norm=alpha_norm,
        )

        self.state.w_ema = w_ema
        self.state.l_prev = l_curr

        # ----------------- diagnostics
        n_clients = z_mat.shape[0]
        w_matrix_log: Optional[List[List[float]]] = None
        if n_clients <= self.log_w_matrix_max_clients:
            w_matrix_log = [[float(v) for v in row] for row in w_ema.tolist()]

        client_train_acc = self._extract_metric(client_metrics, "accuracy", "train_accuracy")
        client_train_loss = self._extract_metric(client_metrics, "loss", "train_loss")

        round_log: Dict[str, Any] = {
            "round": int(server_round),
            "cids": list(cids),

            # spectral signals
            "h_spec": float(h_spec),
            "h_spec_metric": float(h_spec),
            "h_spec_current": float(h_spec_current),
            "h_spec_used_graph": float(h_spec_current),
            "h_spec_raw_current_graph": float(h_spec_raw_current),
            "h_spec_ema": float(self.state.h_spec_ema),
            "h_spec_metric_graph_source": metric_graph_source,
            "h_spec_graph_uses_ema": bool(self.use_ema_graph),
            "graph_used_source": graph_used_source,
            "tau": float(tau),
            "adaptive_tau_enabled": bool(self.adaptive_tau),
            "fixed_tau": float(self.fixed_tau) if not self.adaptive_tau else None,
            **spectral_diag,

            # conflict
            "e_list": [float(x) for x in e.tolist()],
            "e_z_list": [float(x) for x in e_z.tolist()],
            "conflict_weight_list": [float(x) for x in conflict_weight.tolist()],
            "raw_conflict_weight_list": [float(x) for x in raw_cw.tolist()],
            "e_mean": float(e_mean_val),
            "e_std": float(e_std_val_raw),
            "min_e": float(np.min(e)),
            "max_e": float(np.max(e)),
            # legacy keys retained for older analysis scripts
            "mean_e": float(e_mean_val),
            "std_e": float(e_std_val_raw),
            "conflict_penalty_disabled_due_to_estd": bool(estd_disabled),
            "graph_fallback_used": bool(graph_fallback_used),

            # update-space stats
            "z_norm_list": [float(x) for x in z_norms.tolist()],
            "delta_norm_list": [float(x) for x in delta_norms.tolist()],
            "weight_norm_list": [float(x) for x in weight_norms.tolist()],
            "graph_source_norm_list": [float(x) for x in graph_source_norms.tolist()],

            # graph
            "graph_mode": str(self.graph_mode),
            "graph_source": str(self.graph_source),
            "graph_source_used": graph_source_used,
            "knn_k": int(self.knn_k),
            "edge_threshold": float(self.edge_threshold),
            "use_ema_graph": bool(self.use_ema_graph),
            "raw_current_graph_density": float(graph_diag_current["graph_density"]),
            "raw_current_graph_degree_list": list(graph_diag_current["graph_degree_list"]),
            "raw_current_number_of_edges": int(graph_diag_current["number_of_edges"]),
            "raw_current_graph_empty": bool(graph_diag_current["graph_empty"]),
            "graph_density": float(graph_diag["graph_density"]),
            "graph_degree_list": list(graph_diag["graph_degree_list"]),
            "number_of_edges": int(graph_diag["number_of_edges"]),
            "graph_empty": bool(graph_diag["graph_empty"]),
            "W_matrix": w_matrix_log,

            # alpha
            "alpha_mode": alpha_mode,
            "aggregation_target": str(self.aggregation_target),
            "aggregation_target_used": aggregation_target_used,
            "alpha_raw_list": [float(x) for x in alpha_raw.tolist()],
            "alpha_norm_list": [float(x) for x in alpha_norm.tolist()],
            # legacy alias used by some older analysis scripts
            "alpha_list": [float(x) for x in alpha_norm.tolist()],
            "min_alpha": float(np.min(alpha_norm)),
            "max_alpha": float(np.max(alpha_norm)),
            "entropy_alpha": float(compute_entropy(alpha_norm)),
            "effective_clients": float(compute_effective_clients(alpha_norm)),

            # client info
            "client_num_examples": [int(x) for x in n_examples_arr.tolist()],
            "client_train_accuracy_list": client_train_acc,
            "client_train_loss_list": client_train_loss,

            # config flags
            "conflict_mix": float(self.conflict_mix),
            "warmup_rounds": int(self.warmup_rounds),
            "e_std_threshold": float(self.e_std_threshold),
            "min_client_weight": float(self.min_client_weight),
            "diagnostic_only": bool(self.diagnostic_only),
            "graph_source_config": str(self.graph_source),
            "aggregation_target_config": str(self.aggregation_target),
        }
        self.round_logs.append(round_log)

        metrics = {
            "h_spec": float(h_spec),
            "h_spec_metric": float(h_spec),
            "h_spec_current": float(h_spec_current),
            "h_spec_used_graph": float(h_spec_current),
            "h_spec_raw_current_graph": float(h_spec_raw_current),
            "h_spec_ema": float(self.state.h_spec_ema),
            "tau": float(tau),
            "low_frequency_energy_ratio": float(
                spectral_diag["low_frequency_energy_ratio"]
            ),
            "mid_frequency_energy_ratio": float(
                spectral_diag["mid_frequency_energy_ratio"]
            ),
            "high_frequency_energy_ratio": float(
                spectral_diag["high_frequency_energy_ratio"]
            ),
            "high_to_low_energy_ratio": float(spectral_diag["high_to_low_energy_ratio"]),
            "dominant_frequency_mode_index": float(
                spectral_diag["dominant_frequency_mode_index"]
            ),
            "dominant_frequency_mode_lambda": float(
                spectral_diag["dominant_frequency_mode_lambda"]
            ),
            "dominant_frequency_energy_ratio": float(
                spectral_diag["dominant_frequency_energy_ratio"]
            ),
            "spectral_entropy": float(spectral_diag["spectral_entropy"]),
            "eigengap_max": float(spectral_diag["eigengap_max"]),
            "e_mean": float(e_mean_val),
            "e_std": float(e_std_val_raw),
            "e_min": float(np.min(e)),
            "e_max": float(np.max(e)),
            "alpha_min": float(np.min(alpha_norm)),
            "alpha_max": float(np.max(alpha_norm)),
            "alpha_entropy": float(compute_entropy(alpha_norm)),
            "alpha_effective_clients": float(compute_effective_clients(alpha_norm)),
            "raw_current_graph_density": float(graph_diag_current["graph_density"]),
            "raw_current_graph_empty": float(graph_diag_current["graph_empty"]),
            "raw_current_number_of_edges": float(graph_diag_current["number_of_edges"]),
            "graph_density": float(graph_diag["graph_density"]),
            "graph_empty": float(graph_diag["graph_empty"]),
            "number_of_edges": float(graph_diag["number_of_edges"]),
        }
        return ndarrays_to_parameters(new_global), metrics

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _extract_metric(
        client_metrics: List[Dict[str, Any]], *keys: str
    ) -> Optional[List[Optional[float]]]:
        """Look up a metric by any of the given key aliases per client.

        Returns None if no client provides the metric (so analysis scripts
        can store null instead of misleading zeros).  Returns a list of
        floats / None otherwise.
        """
        out: List[Optional[float]] = []
        any_found = False
        for m in client_metrics:
            v: Optional[float] = None
            for k in keys:
                if k in m:
                    try:
                        v = float(m[k])
                        any_found = True
                        break
                    except (TypeError, ValueError):
                        continue
            out.append(v)
        return out if any_found else None
