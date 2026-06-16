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

from graphfl_lab.graph.registry import load_graph_plugins
from graphfl_lab.diagnostics.logging import (
    init_artifact_dir,
)
from graphfl_lab.diagnostics.metrics import summarize_pre_post
from graphfl_lab.strategies.graphfl.client_metrics import (
    extract_metric,
    weighted_optional_mean,
)
from graphfl_lab.strategies.graphfl.config_context import build_config_context
from graphfl_lab.strategies.graphfl.config import GraphFLStrategyState
from graphfl_lab.strategies.graphfl.conflict_metrics import (
    compute_conflict_metric_bundle,
)
from graphfl_lab.strategies.graphfl.diagnostic_artifacts import (
    write_round_diagnostic_artifacts,
)
from graphfl_lab.strategies.graphfl.diagnostic_targets import (
    flatten_diagnostic_post_updates,
)
from graphfl_lab.strategies.graphfl.ema import update_client_update_ema
from graphfl_lab.strategies.graphfl.fit_results import collect_client_fit_batch
from graphfl_lab.strategies.graphfl.momentum import apply_server_optimizer
from graphfl_lab.strategies.graphfl.projection import project_with_cached_matrix
from graphfl_lab.strategies.graphfl.round_graph import build_round_graph_state
from graphfl_lab.strategies.graphfl.round_outputs import build_strategy_round_outputs
from graphfl_lab.strategies.graphfl.round_projection import (
    build_projected_graph_space,
    select_graph_source_vectors,
)
from graphfl_lab.strategies.graphfl.round_weights import select_round_weights
from graphfl_lab.strategies.graphfl.spectral_metrics import (
    compute_round_spectral_metrics,
)
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
)
from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregate_target,
    evaluate_aggregation_target,
)
from graphfl_lab.strategies.graphfl.tracing import matrix_log_if_small
from graphfl_lab.strategies.graphfl.update_space import (
    compute_local_updates,
    compute_update_space_arrays,
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
        aggregation_parameters: Optional[Dict[str, Any]] = None,
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
        self.aggregation_parameters = dict(aggregation_parameters or {})
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
        self.graph_filter_strength = max(float(graph_filter_strength), 0.0)
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

    def _graph_vectors(
        self,
        local_weights: List[NDArrays],
        local_updates: List[NDArrays],
        ema_updates: Optional[List[NDArrays]] = None,
    ) -> Tuple[List[np.ndarray], str]:
        return select_graph_source_vectors(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            graph_source=self.graph_source,
            graph_layer_start=self.graph_layer_start,
            graph_layer_end=self.graph_layer_end,
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
                parameters=self.aggregation_parameters,
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
        fit_batch = collect_client_fit_batch(results)
        cids = fit_batch.cids
        local_weights = fit_batch.local_weights
        n_examples_arr = fit_batch.n_examples_arr
        client_metrics = fit_batch.client_metrics
        in_warmup = int(server_round) <= int(self.warmup_rounds)

        # ----------------- update space and projection
        local_updates = compute_local_updates(
            local_weights=local_weights,
            current_global=self._current_global,
        )
        ema_updates, ema_update_source = self._update_client_update_ema(
            local_updates=local_updates,
            cids=cids,
        )
        update_space = compute_update_space_arrays(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
        )
        graph_space = build_projected_graph_space(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            graph_source=self.graph_source,
            graph_layer_start=self.graph_layer_start,
            graph_layer_end=self.graph_layer_end,
            project_fn=self._project,
        )
        graph_source_used = graph_space.graph_source_used
        graph_source_metadata = dict(graph_space.graph_source_metadata)
        graph_source_norms = graph_space.graph_source_norms
        z_mat = graph_space.z_mat
        z_norms = graph_space.z_norms

        # ----------------- client similarity graph
        round_graph = build_round_graph_state(
            z_mat=z_mat,
            cids=cids,
            n_examples_arr=n_examples_arr,
            server_round=int(server_round),
            graph_seed=int(self.graph_seed),
            graph_mode=self.graph_mode,
            knn_k=self.knn_k,
            edge_threshold=self.edge_threshold,
            graph_scale_sigma=self.graph_scale_sigma,
            learned_graph_lambda=self.learned_graph_lambda,
            correction_family=self.correction_family,
            control_graph_mode=self.control_graph_mode,
            graph_source_used=graph_source_used,
            aggregation_target=self.aggregation_target,
            cluster_method=self.cluster_method,
            cluster_k=self.cluster_k,
            cluster_auto_k=self.cluster_auto_k,
            previous_graph_ema=self.state.w_ema,
            use_ema_graph=self.use_ema_graph,
            in_warmup=in_warmup,
            ema_alpha=self.ema_alpha,
        )
        pre_weights = round_graph.pre_weights
        w_curr = round_graph.current_graph
        graph_meta = round_graph.graph_meta
        if graph_source_metadata:
            source_trace = {
                "schema_version": "lifecycle_trace_v1",
                "phase": "client_state",
                "module": str(
                    graph_source_metadata.get(
                        "component_kind", "ClientStateExtractor"
                    )
                ),
                "name": str(
                    graph_source_metadata.get(
                        "component_name", self.graph_source
                    )
                ),
                "round": int(server_round),
                "values": graph_source_metadata,
            }
            graph_meta = {
                **graph_meta,
                "graph_source_metadata": graph_source_metadata,
                "lifecycle_trace": [
                    source_trace,
                    *(graph_meta.get("lifecycle_trace", []) or []),
                ],
            }
        client_cluster_ids = round_graph.client_cluster_ids
        graph_diag_current = round_graph.graph_diag_current
        w_ema = round_graph.used_graph
        graph_used_source = round_graph.graph_used_source
        graph_diag = round_graph.graph_diag
        graph_fallback_used = round_graph.graph_fallback_used

        spectral_metrics = compute_round_spectral_metrics(
            z_mat=z_mat,
            current_graph=w_curr,
            used_graph=w_ema,
            previous_laplacian=self.state.l_prev,
            previous_h_spec_ema=self.state.h_spec_ema,
        )
        l_curr = spectral_metrics.l_curr
        if not in_warmup:
            self.state.h_spec_ema = spectral_metrics.h_spec_ema_candidate

        # ----------------- spectral filtering and conflict score
        conflict_metrics = compute_conflict_metric_bundle(
            z_mat=z_mat,
            l_mat=l_curr,
            filter_strength=self.graph_filter_strength,
            tau_source_name=self.tau_source,
            h_spec=spectral_metrics.h_spec,
            h_spec_normalized=spectral_metrics.h_spec_normalized,
            h_spec_ema=self.state.h_spec_ema,
            h_spec_ema_candidate=spectral_metrics.h_spec_ema_candidate,
            tau_signal_ema=self.state.tau_signal_ema,
            tau_max=self.tau_max,
            tau_gain=self.tau_gain,
            adaptive_tau=self.adaptive_tau,
            fixed_tau=self.fixed_tau,
            e_std_threshold=self.e_std_threshold,
        )
        if conflict_metrics.tau_source.source_used != "h_spec" and not in_warmup:
            self.state.tau_signal_ema = conflict_metrics.tau_source.ema_candidate

        # ----------------- aggregation weights
        weight_selection = select_round_weights(
            n_examples=n_examples_arr,
            conflict_weight=conflict_metrics.conflict_weight,
            diagnostic_only=self.diagnostic_only,
            in_warmup=in_warmup,
            estd_disabled=conflict_metrics.estd_disabled,
            graph_fallback_used=graph_fallback_used,
            conflict_mix=self.conflict_mix,
            min_client_weight=self.min_client_weight,
            correction_family=self.correction_family,
            graph_free_mode=self.graph_free_mode,
            graph_free_gamma=self.graph_free_gamma,
            contribution_cap=self.contribution_cap,
            clip_quantile=self.clip_quantile,
            update_norms=update_space.delta_norms,
        )
        alpha_raw = weight_selection.alpha_raw
        alpha_norm = weight_selection.alpha_norm
        conflict_weight = weight_selection.conflict_weight
        alpha_mode = weight_selection.alpha_mode
        active_client_mask = weight_selection.active_client_mask

        target_override_for_round = "update" if (self.diagnostic_only or in_warmup) else None
        target_evaluation = evaluate_aggregation_target(
            current_global=self._current_global,
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            alpha_norm=alpha_norm,
            l_mat=l_curr,
            config=AggregationTargetConfig(
                target=target_override_for_round or self.aggregation_target,
                filter_strength=self.graph_filter_strength,
                parameters=self.aggregation_parameters,
            ),
        )
        post_flat_updates = target_evaluation.post_flat_updates
        diagnostic_target_used = target_evaluation.target_used
        diagnostic_filter_diag = dict(target_evaluation.metadata)
        aggregation_trace = {
            "schema_version": "lifecycle_trace_v1",
            "phase": "aggregation",
            "module": str(
                diagnostic_filter_diag.get(
                    "component_kind", "AggregationOperator"
                )
            ),
            "name": str(
                diagnostic_filter_diag.get(
                    "component_name", self.aggregation_target
                )
            ),
            "round": int(server_round),
            "values": diagnostic_filter_diag,
        }
        graph_meta = {
            **graph_meta,
            "aggregation_target_metadata": diagnostic_filter_diag,
            "lifecycle_trace": [
                *(graph_meta.get("lifecycle_trace", []) or []),
                aggregation_trace,
            ],
        }
        pre_post = summarize_pre_post(
            flat_updates=update_space.flat_delta_matrix,
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
            write_round_diagnostic_artifacts(
                artifact_dir=self.diagnostics_artifact_dir,
                run_id=self.diagnostics_run_id,
                variant=self.diagnostics_variant,
                seed=int(self.diagnostics_seed),
                server_round=int(server_round),
                accuracy=float(round_accuracy),
                loss=float(round_loss),
                pre_post=pre_post,
                graph_diag=graph_diag,
                wall_time_sec=float(time.perf_counter() - aggregate_started_at),
                graph_method=str(self.graph_method),
                correction_family=str(self.correction_family),
                graph_source_used=str(graph_source_used),
                graph_variant=str(self.graph_mode),
                diagnostic_target_used=str(diagnostic_target_used),
                graph_used_source=str(graph_used_source),
                graph_meta=graph_meta,
                control_graph_mode=str(self.control_graph_mode),
                cluster_method=str(self.cluster_method),
                cluster_k=int(self.cluster_k),
                cluster_auto_k=bool(self.cluster_auto_k),
                cids=cids,
                n_examples_arr=n_examples_arr,
                client_cluster_ids=client_cluster_ids,
                flat_updates=update_space.flat_delta_matrix,
                pre_weights=pre_weights,
                actual_adjacency=w_ema,
                aggregation_target=self.aggregation_target,
                graph_seed=int(self.graph_seed),
                graph_filter_strength=self.graph_filter_strength,
                graph_free_gamma=self.graph_free_gamma,
                loo_enabled=self.loo_enabled,
            )

        # ----------------- aggregate configured target and apply
        candidate_global = target_evaluation.candidate_global
        aggregation_target_used = target_evaluation.target_used
        target_filter_diag = dict(target_evaluation.metadata)
        new_global, server_opt_diag = self._apply_server_optimizer(candidate_global)

        if not in_warmup:
            self.state.w_ema = w_ema
            self.state.l_prev = l_curr

        # ----------------- diagnostics
        w_matrix_log = matrix_log_if_small(
            w_ema,
            max_clients=self.log_w_matrix_max_clients,
        )

        config_context = build_config_context(self)
        round_outputs = build_strategy_round_outputs(
            server_round=server_round,
            cids=cids,
            spectral_metrics=spectral_metrics,
            h_spec_ema=self.state.h_spec_ema,
            in_warmup=in_warmup,
            conflict_metrics=conflict_metrics,
            target_filter_diag=target_filter_diag,
            diagnostic_filter_diag=diagnostic_filter_diag,
            conflict_weight=conflict_weight,
            graph_fallback_used=graph_fallback_used,
            z_norms=z_norms,
            update_space=update_space,
            graph_source_norms=graph_source_norms,
            ema_update_source=ema_update_source,
            graph_source_used=graph_source_used,
            graph_used_source=graph_used_source,
            graph_meta=graph_meta,
            graph_diag_current=graph_diag_current,
            graph_diag=graph_diag,
            w_matrix_log=w_matrix_log,
            alpha_raw=alpha_raw,
            alpha_norm=alpha_norm,
            alpha_mode=alpha_mode,
            active_client_mask=active_client_mask,
            aggregation_target_used=aggregation_target_used,
            diagnostic_target_used=diagnostic_target_used,
            server_opt_diag=server_opt_diag,
            pre_post=pre_post,
            n_examples_arr=n_examples_arr,
            client_train_acc=client_train_acc,
            client_train_loss=client_train_loss,
            config_context=config_context,
            correction_family=self.correction_family,
            control_graph_mode=self.control_graph_mode,
            graph_mode=self.graph_mode,
        )
        self.round_logs.append(round_outputs.round_log)
        return ndarrays_to_parameters(new_global), round_outputs.fit_metrics

SpectralConflictAwareStrategy = GraphFLDiagnosticStrategy


__all__ = ["GraphFLDiagnosticStrategy", "SpectralConflictAwareStrategy"]
