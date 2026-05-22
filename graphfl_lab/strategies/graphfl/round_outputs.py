"""Round log and fit-metric output assembly for GraphFL strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from graphfl_lab.strategies.graphfl.diagnostics import (
    build_fit_metrics,
    build_round_log,
)
from graphfl_lab.strategies.graphfl.round_context import (
    build_alpha_context,
    build_client_context,
    build_conflict_context,
    build_graph_context,
    build_spectral_context,
    build_update_context,
)
from graphfl_lab.strategies.graphfl.tracing import make_round_trace_payload


@dataclass(frozen=True)
class RoundStrategyOutputs:
    round_log: Dict[str, Any]
    fit_metrics: Dict[str, float]


def build_strategy_round_outputs(
    *,
    server_round: int,
    cids: Sequence[str],
    spectral_metrics: Any,
    h_spec_ema: float,
    in_warmup: bool,
    conflict_metrics: Any,
    target_filter_diag: Dict[str, Any],
    diagnostic_filter_diag: Dict[str, Any],
    conflict_weight: np.ndarray,
    graph_fallback_used: bool,
    z_norms: np.ndarray,
    update_space: Any,
    graph_source_norms: np.ndarray,
    ema_update_source: str,
    graph_source_used: str,
    graph_used_source: str,
    graph_meta: Dict[str, Any],
    graph_diag_current: Dict[str, Any],
    graph_diag: Dict[str, Any],
    w_matrix_log: Any,
    alpha_raw: np.ndarray,
    alpha_norm: np.ndarray,
    alpha_mode: str,
    active_client_mask: np.ndarray,
    aggregation_target_used: str,
    diagnostic_target_used: str,
    server_opt_diag: Dict[str, Any],
    pre_post: Dict[str, Any],
    n_examples_arr: np.ndarray,
    client_train_acc: Any,
    client_train_loss: Any,
    config_context: Dict[str, Any],
    correction_family: str,
    control_graph_mode: str,
    graph_mode: str,
) -> RoundStrategyOutputs:
    spectral_context = build_spectral_context(
        spectral_metrics=spectral_metrics,
        h_spec_ema=h_spec_ema,
        in_warmup=in_warmup,
        conflict_metrics=conflict_metrics,
        target_filter_diag=target_filter_diag,
        diagnostic_filter_diag=diagnostic_filter_diag,
    )
    conflict_context = build_conflict_context(
        conflict_metrics=conflict_metrics,
        conflict_weight=conflict_weight,
        graph_fallback_used=graph_fallback_used,
    )
    update_context = build_update_context(
        z_norms=z_norms,
        update_space=update_space,
        graph_source_norms=graph_source_norms,
        ema_update_source=ema_update_source,
    )
    graph_context = build_graph_context(
        graph_source_used=graph_source_used,
        graph_used_source=graph_used_source,
        graph_meta=graph_meta,
        graph_diag_current=graph_diag_current,
        graph_diag=graph_diag,
        w_matrix_log=w_matrix_log,
    )
    alpha_context = build_alpha_context(
        alpha_raw=alpha_raw,
        alpha_norm=alpha_norm,
        alpha_mode=alpha_mode,
        active_client_mask=active_client_mask,
        aggregation_target_used=aggregation_target_used,
        diagnostic_target_used=diagnostic_target_used,
        server_opt_diag=server_opt_diag,
        pre_post=pre_post,
    )
    client_context = build_client_context(
        n_examples_arr=n_examples_arr,
        client_train_acc=client_train_acc,
        client_train_loss=client_train_loss,
    )
    round_log = build_round_log(
        server_round=server_round,
        cids=list(cids),
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
            correction_family=correction_family,
            control_graph_mode=control_graph_mode,
            graph_mode=graph_mode,
            alpha_mode=alpha_mode,
            pre_post_round=pre_post["round"],
        )
    )
    fit_metrics = build_fit_metrics(
        spectral=spectral_context,
        conflict=conflict_context,
        alpha_norm=alpha_norm,
        graph_diag_current=graph_diag_current,
        graph_diag=graph_diag,
        filter_diag=conflict_metrics.filter_diag,
        config=config_context,
        pre_post_round=pre_post["round"],
    )
    return RoundStrategyOutputs(round_log=round_log, fit_metrics=fit_metrics)


__all__ = ["RoundStrategyOutputs", "build_strategy_round_outputs"]
