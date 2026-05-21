"""Composable graph-FL diagnostic strategy.

This module handles server-side orchestration for graph-based FL designs:

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

from dataclasses import replace
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from graphfl_lab.graph.builders import build_relation_graph
from graphfl_lab.graph.registry import load_graph_plugins
from graphfl_lab.diagnostics.logging import (
    append_client_metrics_csv,
    append_counterfactual_metrics_csv,
    append_graph_stats_csv,
    append_module_traces_jsonl,
    append_round_metrics_csv,
    init_artifact_dir,
)
from graphfl_lab.diagnostics.metrics import summarize_pre_post
from graphfl_lab.diagnostics.schema import (
    ClientRoundDiagnostics,
    RoundDiagnostics,
)
from graphfl_lab.graph.diagnostics import compute_graph_diagnostics
from graphfl_lab.lifecycle.counterfactuals import default_counterfactual_specs
from graphfl_lab.lifecycle.diagnostic_runner import (
    CounterfactualDiagnosticRunner,
    MinimalAggregationAdapter,
)
from graphfl_lab.strategies.graphfl.aggregation import (
    apply_correction_family,
    compute_conflict_weights,
    compute_tau,
    resolve_tau_source,
    select_aggregation_weights,
)
from graphfl_lab.graph.sources import (
    GraphSourceConfig,
    graph_vectors_for_spectral,
    normalize_key,
)
from graphfl_lab.projection import flatten_weights
from graphfl_lab.strategies.graphfl.client_metrics import (
    extract_metric,
    weighted_optional_mean,
)
from graphfl_lab.strategies.graphfl.config import GraphFLStrategyState
from graphfl_lab.strategies.graphfl.diagnostics import (
    build_fit_metrics,
    build_round_log,
    heterogeneity,
    spectral_energy_diagnostics,
)
from graphfl_lab.strategies.graphfl.diagnostic_targets import (
    flatten_diagnostic_post_updates,
)
from graphfl_lab.strategies.graphfl.ema import update_client_update_ema
from graphfl_lab.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
    laplacian,
    normalized_conflicts,
)
from graphfl_lab.strategies.graphfl.momentum import apply_server_optimizer
from graphfl_lab.strategies.graphfl.projection import project_with_cached_matrix
from graphfl_lab.strategies.baselines import (
    TracingFedAdagrad,
    TracingFedAdam,
    TracingFedAvg,
    TracingFedAvgM,
    TracingFedMedian,
    TracingFedNova,
    TracingFedProx,
    TracingFedSim,
    TracingFedTrimmedAvg,
    TracingFedYogi,
    _EvalTracer,
    _fit_result_cid_key,
    sort_fit_results_by_cid as _sort_fit_results_by_cid,
)
from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregate_target,
)
from graphfl_lab.strategies.graphfl.tracing import (
    make_round_trace_payload,
    matrix_log_if_small,
)


# =============================================================================
# Graph-FL diagnostic strategy.
# =============================================================================


class GraphFLDiagnosticStrategy(_EvalTracer, fl.server.strategy.FedAvg):
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
        graph_scale_sigma: float = 1.0,
        learned_graph_lambda: float = 1.0,
        graph_layer_start: int = 0,
        graph_layer_end: int = 0,
        e_std_threshold: float = 0.0,
        graph_seed: int = 0,
        graph_plugin_modules: str = "",
        graph_method: str = "none",
        correction_family: str = "real_graph",
        control_graph_mode: str = "random",
        cluster_method: str = "none",
        cluster_k: int = 0,
        cluster_auto_k: bool = False,
        use_ema_graph: bool = True,
        adaptive_tau: bool = True,
        fixed_tau: float = 1.0,
        tau_source: str = "h_spec",
        spectral_filter_strength: Optional[float] = None,
        graph_filter_strength: float = 1.0,
        client_update_ema_alpha: float = 0.8,
        diagnostics_enable: bool = False,
        loo_enabled: bool = False,
        diagnostics_artifact_dir: str = "",
        diagnostics_run_id: str = "",
        diagnostics_variant: str = "",
        diagnostics_seed: int = -1,
        graph_free_mode: str = "none",
        graph_free_gamma: float = 1.0,
        clip_quantile: float = 0.9,
        contribution_cap: float = 0.0,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
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
        self.graph_scale_sigma = float(graph_scale_sigma)
        self.learned_graph_lambda = float(learned_graph_lambda)
        self.graph_layer_start = int(graph_layer_start)
        self.graph_layer_end = int(graph_layer_end)
        self.e_std_threshold = float(e_std_threshold)
        self.graph_seed = int(graph_seed)
        self.graph_plugin_modules = str(graph_plugin_modules)
        load_graph_plugins(self.graph_plugin_modules)
        self.graph_method = str(graph_method)
        self.correction_family = str(correction_family)
        self.control_graph_mode = str(control_graph_mode)
        self.cluster_method = str(cluster_method)
        self.cluster_k = int(cluster_k)
        self.cluster_auto_k = bool(cluster_auto_k)
        self.use_ema_graph = bool(use_ema_graph)
        self.adaptive_tau = bool(adaptive_tau)
        self.fixed_tau = float(fixed_tau)
        self.tau_source = str(tau_source)
        filter_strength = (
            graph_filter_strength
            if spectral_filter_strength is None
            else spectral_filter_strength
        )
        self.graph_filter_strength = max(float(filter_strength), 0.0)
        self.spectral_filter_strength = self.graph_filter_strength
        self.client_update_ema_alpha = float(client_update_ema_alpha)
        self.diagnostics_enable = bool(diagnostics_enable)
        self.loo_enabled = bool(loo_enabled)
        self.diagnostics_run_id = str(diagnostics_run_id).strip() or "run"
        self.diagnostics_variant = str(diagnostics_variant).strip() or "ours"
        self.diagnostics_seed = int(diagnostics_seed)
        self.graph_free_mode = str(graph_free_mode)
        self.graph_free_gamma = float(graph_free_gamma)
        self.clip_quantile = float(clip_quantile)
        self.contribution_cap = float(contribution_cap)
        diag_root = str(diagnostics_artifact_dir).strip()
        self.diagnostics_artifact_dir = (
            init_artifact_dir(Path(diag_root))
            if diag_root
            else None
        )
        self.server_learning_rate = float(server_learning_rate)
        self.server_momentum = float(server_momentum)
        self.server_opt = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )
        self.server_momentum_vector: Optional[NDArrays] = None
        self.diagnostic_only = bool(diagnostic_only)
        self.min_client_weight = float(min_client_weight)
        self.log_w_matrix_max_clients = int(log_w_matrix_max_clients)
        self.state = GraphFLStrategyState()
        self._current_global: Optional[NDArrays] = None
        self._proj_matrix: Optional[np.ndarray] = None
        self.round_logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ utils

    def _project(self, vec: np.ndarray) -> np.ndarray:
        projected, self._proj_matrix = project_with_cached_matrix(
            vec,
            projection_matrix=self._proj_matrix,
            compression_dim=self.compression_dim,
            compression_seed=self.compression_seed,
        )
        return projected

    def _update_client_update_ema(
        self,
        local_updates: List[NDArrays],
        cids: List[str],
    ) -> Tuple[List[NDArrays], str]:
        ema_updates, source, stored_updates, stored_cids = update_client_update_ema(
            local_updates=local_updates,
            cids=cids,
            previous_updates=self.state.client_update_ema,
            previous_cids=self.state.client_update_ema_cids,
            alpha=self.client_update_ema_alpha,
        )
        self.state.client_update_ema = stored_updates
        self.state.client_update_ema_cids = stored_cids
        return ema_updates, source

    @staticmethod
    def _norm_key(value: str) -> str:
        return normalize_key(value)

    def _graph_vectors(
        self,
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
        ema_updates: Optional[List[NDArrays]] = None,
    ) -> Tuple[List[np.ndarray], str]:
        return graph_vectors_for_spectral(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            config=GraphSourceConfig(
                source=self.graph_source,
                layer_start=self.graph_layer_start,
                layer_end=self.graph_layer_end,
            ),
        )

    def _aggregate_target(
        self,
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
        alpha_norm: np.ndarray,
        l_mat: Optional[np.ndarray] = None,
        target_override: Optional[str] = None,
        ema_updates: Optional[List[NDArrays]] = None,
    ) -> Tuple[NDArrays, str, Dict[str, Any]]:
        if self._current_global is None:
            raise ValueError("aggregation target requires current global parameters")
        return aggregate_target(
            current_global=self._current_global,
            local_weights=local_weights,
            local_updates=local_updates,
            alpha_norm=alpha_norm,
            config=AggregationTargetConfig(
                target=target_override or self.aggregation_target,
                filter_strength=self.graph_filter_strength,
            ),
            l_mat=l_mat,
            ema_updates=ema_updates,
        )

    def _diagnostic_post_flat_updates(
        self,
        *,
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
        ema_updates: List[NDArrays],
        l_mat: np.ndarray,
        target_override: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        if self._current_global is None:
            raise ValueError("diagnostic target requires current global parameters")
        return flatten_diagnostic_post_updates(
            current_global=self._current_global,
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            l_mat=l_mat,
            aggregation_target=self.aggregation_target,
            filter_strength=self.graph_filter_strength,
            target_override=target_override,
        )

    def _apply_server_optimizer(
        self,
        candidate_global: NDArrays,
    ) -> Tuple[NDArrays, Dict[str, Any]]:
        """Apply FedAvgM-style server momentum to an already aggregated model."""
        new_global, self.server_momentum_vector, diagnostics = apply_server_optimizer(
            current_global=self._current_global,
            candidate_global=candidate_global,
            server_learning_rate=self.server_learning_rate,
            server_momentum=self.server_momentum,
            server_momentum_vector=self.server_momentum_vector,
        )
        return new_global, diagnostics

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
        aggregate_started_at = time.perf_counter()
        if not results:
            return None, {}
        if self._current_global is None:
            return super().aggregate_fit(server_round, results, failures)

        # ----------------- collect client local weights and metadata
        ordered_results = _sort_fit_results_by_cid(results)
        cids: List[str] = []
        local_weights: List[NDArrays] = []
        n_examples: List[int] = []
        client_metrics: List[Dict[str, Any]] = []
        for proxy, fit_res in ordered_results:
            metrics = dict(fit_res.metrics or {})
            cids.append(str(metrics.get("cid", getattr(proxy, "cid", "?"))))
            local_weights.append(parameters_to_ndarrays(fit_res.parameters))
            n_examples.append(int(fit_res.num_examples))
            client_metrics.append(metrics)
        n_examples_arr = np.array(n_examples, dtype=np.float64)
        in_warmup = int(server_round) <= int(self.warmup_rounds)

        # ----------------- update space and projection
        local_updates: List[NDArrays] = [
            [lp - gp for lp, gp in zip(local, self._current_global)]
            for local in local_weights
        ]
        ema_updates, ema_update_source = self._update_client_update_ema(
            local_updates=local_updates,
            cids=cids,
        )
        flat_deltas = [flatten_weights(g_i) for g_i in local_updates]
        delta_norms = np.array([float(np.linalg.norm(g)) for g in flat_deltas])
        flat_ema_deltas = [flatten_weights(g_i) for g_i in ema_updates]
        ema_delta_norms = np.array([float(np.linalg.norm(g)) for g in flat_ema_deltas])
        flat_weights = [flatten_weights(w_i) for w_i in local_weights]
        weight_norms = np.array([float(np.linalg.norm(w)) for w in flat_weights])
        graph_vectors, graph_source_used = self._graph_vectors(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
        )
        graph_source_norms = np.array([float(np.linalg.norm(g)) for g in graph_vectors])
        z_list = [self._project(g) for g in graph_vectors]
        z_mat = np.stack(z_list, axis=0)
        z_norms = np.array([float(np.linalg.norm(z)) for z in z_list])

        # ----------------- client similarity graph
        graph_rng = np.random.default_rng(
            int(self.graph_seed) * 1009 + int(server_round) * 13
        )
        pre_weights = n_examples_arr / (float(np.sum(n_examples_arr)) + 1e-12)
        w_curr, graph_meta = build_relation_graph(
            z_mat=z_mat,
            mode=self.graph_mode,
            knn_k=self.knn_k,
            edge_threshold=self.edge_threshold,
            rng=graph_rng,
            graph_scale_sigma=self.graph_scale_sigma,
            learned_graph_lambda=self.learned_graph_lambda,
            correction_family=self.correction_family,
            control_graph_mode=self.control_graph_mode,
            graph_source=graph_source_used,
            aggregation_target=self.aggregation_target,
            cluster_method=self.cluster_method,
            cluster_k=self.cluster_k,
            cluster_auto_k=self.cluster_auto_k,
            cluster_seed=int(self.graph_seed) + int(server_round),
            client_sample_weights=pre_weights,
        )
        cluster_ids = graph_meta.get("cluster_ids")
        if isinstance(cluster_ids, list) and len(cluster_ids) == len(cids):
            client_cluster_ids = [int(x) for x in cluster_ids]
        else:
            client_cluster_ids = [-1 for _ in cids]
        graph_diag_current = compute_graph_diagnostics(w_curr)
        if self.use_ema_graph:
            if self.state.w_ema is None or in_warmup:
                w_ema = w_curr
            else:
                w_ema = (
                    self.ema_alpha * self.state.w_ema
                    + (1.0 - self.ema_alpha) * w_curr
                )
        else:
            w_ema = w_curr

        if self.use_ema_graph and in_warmup:
            graph_used_source = "warmup_current_graph"
        elif self.use_ema_graph:
            graph_used_source = "ema_graph"
        else:
            graph_used_source = "raw_current_graph"
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
        metric_eigvals = np.linalg.eigvalsh(l_for_metric)
        metric_lambda_max = float(max(np.max(metric_eigvals), 1e-12))
        h_spec_normalized = float(h_spec / (metric_lambda_max + 1e-12))
        h_spec_current_normalized = float(
            h_spec_current / (float(max(np.max(np.linalg.eigvalsh(l_curr)), 1e-12)) + 1e-12)
        )
        h_spec_raw_current_normalized = float(
            h_spec_raw_current
            / (float(max(np.max(np.linalg.eigvalsh(l_raw_current)), 1e-12)) + 1e-12)
        )
        h_spec_ema_candidate = 0.9 * self.state.h_spec_ema + 0.1 * h_spec
        if not in_warmup:
            self.state.h_spec_ema = h_spec_ema_candidate
        spectral_diag = spectral_energy_diagnostics(z_mat=z_mat, l_mat=l_curr)

        # ----------------- spectral filtering and conflict score
        z_tilde, filter_diag = apply_spectral_filter_with_diagnostics(
            z_mat=z_mat,
            l_mat=l_curr,
            filter_strength=self.graph_filter_strength,
        )
        e = normalized_conflicts(z_mat=z_mat, z_tilde=z_tilde)
        e_std_for_tau = float(np.std(e))

        tau_source = resolve_tau_source(
            tau_source=self.tau_source,
            h_spec=h_spec,
            h_spec_normalized=h_spec_normalized,
            e_std=e_std_for_tau,
            h_spec_ema=self.state.h_spec_ema,
            h_spec_ema_candidate=h_spec_ema_candidate,
            tau_signal_ema=self.state.tau_signal_ema,
        )
        if tau_source.source_used != "h_spec" and not in_warmup:
            self.state.tau_signal_ema = tau_source.ema_candidate

        tau = compute_tau(
            h_spec_ema=tau_source.ema_value,
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
        weight_selection = select_aggregation_weights(
            n_examples=n_examples_arr,
            conflict_weight=conflict_weight,
            diagnostic_only=self.diagnostic_only,
            in_warmup=in_warmup,
            estd_disabled=estd_disabled,
            graph_fallback_used=graph_fallback_used,
            conflict_mix=self.conflict_mix,
            min_client_weight=self.min_client_weight,
        )
        weight_selection = apply_correction_family(
            correction_family=self.correction_family,
            selection=weight_selection,
            n_examples=n_examples_arr,
            graph_free_mode=self.graph_free_mode,
            graph_free_gamma=self.graph_free_gamma,
            contribution_cap=self.contribution_cap,
            clip_quantile=self.clip_quantile,
            update_norms=delta_norms,
        )
        alpha_raw = weight_selection.alpha_raw
        alpha_norm = weight_selection.alpha_norm
        conflict_weight = weight_selection.conflict_weight
        alpha_mode = weight_selection.alpha_mode
        active_client_mask = weight_selection.active_client_mask

        target_override_for_round = "update" if (self.diagnostic_only or in_warmup) else None
        post_flat_updates, diagnostic_target_used, diagnostic_filter_diag = (
            self._diagnostic_post_flat_updates(
                local_weights=local_weights,
                local_updates=local_updates,
                ema_updates=ema_updates,
                l_mat=l_curr,
                target_override=target_override_for_round,
            )
        )
        pre_post = summarize_pre_post(
            flat_updates=np.stack(flat_deltas, axis=0),
            flat_updates_post=post_flat_updates,
            weights_pre=pre_weights,
            weights_post=alpha_norm,
            loo_enabled=self.loo_enabled,
        )
        client_train_acc = extract_metric(client_metrics, "accuracy", "train_accuracy")
        client_train_loss = extract_metric(client_metrics, "loss", "train_loss")
        round_accuracy = weighted_optional_mean(client_train_acc, n_examples_arr)
        round_loss = weighted_optional_mean(client_train_loss, n_examples_arr)

        # ----------------- diagnostic artifact rows
        if self.diagnostics_enable and self.diagnostics_artifact_dir is not None:
            graph_meta_kind = str(graph_meta.get("graph_kind", graph_meta.get("kind", "")))
            wall_time_sec = float(time.perf_counter() - aggregate_started_at)
            round_diag = RoundDiagnostics(
                run_id=self.diagnostics_run_id,
                variant=self.diagnostics_variant,
                seed=int(self.diagnostics_seed),
                round=int(server_round),
                accuracy=float(round_accuracy),
                loss=float(round_loss),
                di_pre=float(pre_post["round"]["di_pre"]),
                di_post=float(pre_post["round"]["di_post"]),
                neff_pre=float(pre_post["round"]["neff_pre"]),
                neff_post=float(pre_post["round"]["neff_post"]),
                align_mean_pre=float(pre_post["round"]["align_mean_pre"]),
                align_mean_post=float(pre_post["round"]["align_mean_post"]),
                loo_mean_pre=float(pre_post["round"]["loo_mean_pre"]),
                loo_mean_post=float(pre_post["round"]["loo_mean_post"]),
                graph_density=float(graph_diag["graph_density"]),
                graph_entropy=float(graph_diag["graph_entropy"]),
                alpha_entropy=float(pre_post["round"]["alpha_entropy"]),
                wall_time_sec=wall_time_sec,
                graph_method=str(self.graph_method),
                correction_family=str(self.correction_family),
                graph_source=str(graph_source_used),
                graph_variant=str(self.graph_mode),
                aggregation_target=str(diagnostic_target_used),
                graph_kind=graph_meta_kind,
            )
            append_round_metrics_csv(
                self.diagnostics_artifact_dir / "round_metrics.csv",
                round_diag.to_dict(),
            )
            append_graph_stats_csv(
                self.diagnostics_artifact_dir / "graph_stats.csv",
                {
                    "run_id": self.diagnostics_run_id,
                    "variant": self.diagnostics_variant,
                    "seed": int(self.diagnostics_seed),
                    "round": int(server_round),
                    "graph_method": str(self.graph_method),
                    "correction_family": str(self.correction_family),
                    "graph_source": str(graph_source_used),
                    "graph_variant": str(self.graph_mode),
                    "aggregation_target": str(diagnostic_target_used),
                    "graph_kind": graph_meta_kind,
                    "graph_used_source": str(graph_used_source),
                    "graph_source_used": str(graph_source_used),
                    "graph_density": float(graph_diag["graph_density"]),
                    "graph_entropy": float(graph_diag["graph_entropy"]),
                    "graph_num_nodes": int(graph_diag["graph_num_nodes"]),
                    "number_of_edges": int(graph_diag["number_of_edges"]),
                    "graph_degree_mean": float(graph_diag["graph_degree_mean"]),
                    "graph_degree_min": int(graph_diag["graph_degree_min"]),
                    "graph_degree_max": int(graph_diag["graph_degree_max"]),
                    "graph_empty": bool(graph_diag["graph_empty"]),
                    "control_graph_mode": str(self.control_graph_mode),
                    "cluster_method": str(self.cluster_method),
                    "cluster_k_config": int(self.cluster_k),
                    "cluster_auto_k": bool(self.cluster_auto_k),
                },
            )
            client_rows: List[Dict[str, object]] = []
            norms_pre = pre_post["norms_pre"]
            norms_post = pre_post["norms_post"]
            for i, cid in enumerate(cids):
                row = ClientRoundDiagnostics(
                    run_id=self.diagnostics_run_id,
                    variant=self.diagnostics_variant,
                    seed=int(self.diagnostics_seed),
                    round=int(server_round),
                    cid=str(cid),
                    num_examples=int(n_examples_arr[i]),
                    update_norm_raw=float(norms_pre[i]),
                    update_norm_corrected=float(norms_post[i]),
                    q_raw=float(pre_post["q_pre"][i]),
                    q_corrected=float(pre_post["q_post"][i]),
                    alignment_raw=float(pre_post["align_pre"][i]),
                    alignment_corrected=float(pre_post["align_post"][i]),
                    loo_raw=float(pre_post["loo_pre"][i]),
                    loo_corrected=float(pre_post["loo_post"][i]),
                    cluster_id=int(client_cluster_ids[i]),
                )
                client_rows.append(row.to_dict())
            append_client_metrics_csv(
                self.diagnostics_artifact_dir / "client_metrics.csv",
                client_rows,
            )
            counterfactual_target = str(diagnostic_target_used or self.aggregation_target)
            counterfactual_specs = tuple(
                (
                    spec
                    if spec.name == "graphfree_dominance_reweight"
                    else replace(spec, aggregation_target=counterfactual_target)
                )
                for spec in default_counterfactual_specs()
            )
            seed_base = (
                int(self.diagnostics_seed)
                if int(self.diagnostics_seed) >= 0
                else int(self.graph_seed)
            )
            counterfactual_runner = CounterfactualDiagnosticRunner(
                aggregation_adapter=MinimalAggregationAdapter(
                    filter_strength=self.graph_filter_strength,
                    dominance_gamma=self.graph_free_gamma,
                ),
                loo_enabled=self.loo_enabled,
                rng=np.random.default_rng(seed_base * 4099 + int(server_round)),
            )
            counterfactual_results = counterfactual_runner.run(
                flat_updates=np.stack(flat_deltas, axis=0),
                weights_pre=pre_weights,
                actual_adjacency=w_ema,
                specs=counterfactual_specs,
                round_number=int(server_round),
            )
            counterfactual_rows = []
            module_trace_rows = []

            def with_run_context(payload: Dict[str, Any]) -> Dict[str, Any]:
                enriched = dict(payload)
                if enriched.get("round") is None:
                    enriched["round"] = int(server_round)
                values = dict(enriched.get("values") or {})
                values.setdefault("run_id", self.diagnostics_run_id)
                values.setdefault("variant", self.diagnostics_variant)
                values.setdefault("seed", int(self.diagnostics_seed))
                enriched["values"] = values
                return enriched

            for trace in graph_meta.get("lifecycle_trace", []) or []:
                module_trace_rows.append(with_run_context(dict(trace)))
            for result in counterfactual_results:
                counterfactual_row = {
                    "run_id": self.diagnostics_run_id,
                    "variant": self.diagnostics_variant,
                    "seed": int(self.diagnostics_seed),
                    "round": int(server_round),
                    "graph_method": str(self.graph_method),
                    "graph_variant": str(self.graph_mode),
                    **dict(result.metrics),
                }
                counterfactual_rows.append(counterfactual_row)
                for trace in result.trace_records:
                    module_trace_rows.append(with_run_context(trace.to_dict()))
            append_counterfactual_metrics_csv(
                self.diagnostics_artifact_dir / "counterfactual_metrics.csv",
                counterfactual_rows,
            )
            append_module_traces_jsonl(
                self.diagnostics_artifact_dir / "module_traces.jsonl",
                module_trace_rows,
            )

        # ----------------- aggregate configured target and apply
        candidate_global, aggregation_target_used, target_filter_diag = self._aggregate_target(
            local_weights=local_weights,
            local_updates=local_updates,
            alpha_norm=alpha_norm,
            l_mat=l_curr,
            target_override=target_override_for_round,
            ema_updates=ema_updates,
        )
        new_global, server_opt_diag = self._apply_server_optimizer(candidate_global)

        if not in_warmup:
            self.state.w_ema = w_ema
            self.state.l_prev = l_curr

        # ----------------- diagnostics
        w_matrix_log = matrix_log_if_small(
            w_ema,
            max_clients=self.log_w_matrix_max_clients,
        )

        spectral_context = {
            "h_spec": h_spec,
            "h_spec_current": h_spec_current,
            "h_spec_raw_current": h_spec_raw_current,
            "h_spec_normalized": h_spec_normalized,
            "h_spec_current_normalized": h_spec_current_normalized,
            "h_spec_raw_current_normalized": h_spec_raw_current_normalized,
            "metric_lambda_max": metric_lambda_max,
            "h_spec_ema": self.state.h_spec_ema,
            "h_spec_ema_candidate": h_spec_ema_candidate,
            "metric_graph_source": metric_graph_source,
            "in_warmup": in_warmup,
            "tau": tau,
            "tau_source_used": tau_source.source_used,
            "tau_source_signal": tau_source.source_signal,
            "tau_source_ema": tau_source.ema_value,
            "tau_source_ema_candidate": tau_source.ema_candidate,
            "spectral_diag": spectral_diag,
            "filter_diag": filter_diag,
            "target_filter_diag": target_filter_diag,
            "diagnostic_filter_diag": diagnostic_filter_diag,
        }
        conflict_context = {
            "e": e,
            "e_z": e_z,
            "conflict_weight": conflict_weight,
            "raw_cw": raw_cw,
            "e_mean": e_mean_val,
            "e_std": e_std_val_raw,
            "estd_disabled": estd_disabled,
            "graph_fallback_used": graph_fallback_used,
        }
        update_context = {
            "z_norms": z_norms,
            "delta_norms": delta_norms,
            "ema_delta_norms": ema_delta_norms,
            "weight_norms": weight_norms,
            "graph_source_norms": graph_source_norms,
            "ema_update_source": ema_update_source,
        }
        graph_context = {
            "graph_source_used": graph_source_used,
            "graph_used_source": graph_used_source,
            "graph_meta": graph_meta,
            "graph_diag_current": graph_diag_current,
            "graph_diag": graph_diag,
            "w_matrix_log": w_matrix_log,
        }
        alpha_context = {
            "alpha_raw": alpha_raw,
            "alpha_norm": alpha_norm,
            "alpha_mode": alpha_mode,
            "active_client_mask": active_client_mask,
            "aggregation_target_used": aggregation_target_used,
            "diagnostic_target_used": diagnostic_target_used,
            "server_opt_diag": server_opt_diag,
            "pre_post_round": pre_post["round"],
            "q_pre": pre_post["q_pre"],
            "q_post": pre_post["q_post"],
            "align_pre": pre_post["align_pre"],
            "align_post": pre_post["align_post"],
            "loo_pre": pre_post["loo_pre"],
            "loo_post": pre_post["loo_post"],
        }
        client_context = {
            "n_examples_arr": n_examples_arr,
            "client_train_acc": client_train_acc,
            "client_train_loss": client_train_loss,
        }
        config_context = {
            "adaptive_tau": self.adaptive_tau,
            "aggregation_target": self.aggregation_target,
            "client_update_ema_alpha": self.client_update_ema_alpha,
            "conflict_mix": self.conflict_mix,
            "diagnostic_only": self.diagnostic_only,
            "diagnostics_enable": self.diagnostics_enable,
            "loo_enabled": self.loo_enabled,
            "edge_threshold": self.edge_threshold,
            "e_std_threshold": self.e_std_threshold,
            "fixed_tau": self.fixed_tau,
            "graph_layer_end": self.graph_layer_end,
            "graph_layer_start": self.graph_layer_start,
            "graph_mode": self.graph_mode,
            "graph_method": self.graph_method,
            "graph_scale_sigma": self.graph_scale_sigma,
            "graph_source": self.graph_source,
            "correction_family": self.correction_family,
            "control_graph_mode": self.control_graph_mode,
            "cluster_method": self.cluster_method,
            "cluster_k": self.cluster_k,
            "cluster_auto_k": self.cluster_auto_k,
            "graph_free_mode": self.graph_free_mode,
            "graph_free_gamma": self.graph_free_gamma,
            "clip_quantile": self.clip_quantile,
            "contribution_cap": self.contribution_cap,
            "knn_k": self.knn_k,
            "learned_graph_lambda": self.learned_graph_lambda,
            "min_client_weight": self.min_client_weight,
            "server_learning_rate": self.server_learning_rate,
            "server_momentum": self.server_momentum,
            "graph_filter_strength": self.graph_filter_strength,
            "spectral_filter_strength": self.graph_filter_strength,
            "tau_source": self.tau_source,
            "use_ema_graph": self.use_ema_graph,
            "warmup_rounds": self.warmup_rounds,
        }
        round_log = build_round_log(
            server_round=server_round,
            cids=cids,
            spectral=spectral_context,
            conflict=conflict_context,
            update_stats=update_context,
            graph=graph_context,
            alpha=alpha_context,
            client=client_context,
            config=config_context,
        )
        round_log.update(
            make_round_trace_payload(
                correction_family=self.correction_family,
                control_graph_mode=self.control_graph_mode,
                graph_mode=self.graph_mode,
                alpha_mode=alpha_mode,
                pre_post_round=pre_post["round"],
            )
        )
        self.round_logs.append(round_log)

        metrics = build_fit_metrics(
            spectral=spectral_context,
            conflict=conflict_context,
            alpha_norm=alpha_norm,
            graph_diag_current=graph_diag_current,
            graph_diag=graph_diag,
            filter_diag=filter_diag,
            config=config_context,
            pre_post_round=pre_post["round"],
        )
        return ndarrays_to_parameters(new_global), metrics

SpectralConflictAwareStrategy = GraphFLDiagnosticStrategy


__all__ = ["GraphFLDiagnosticStrategy", "SpectralConflictAwareStrategy"]
