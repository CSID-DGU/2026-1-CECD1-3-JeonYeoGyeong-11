"""Diagnostic artifact writing orchestration for GraphFL strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from graphfl_lab.diagnostics.logging import (
    append_client_metrics_csv,
    append_counterfactual_metrics_csv,
    append_graph_stats_csv,
    append_module_traces_jsonl,
    append_round_metrics_csv,
)
from graphfl_lab.strategies.graphfl.artifact_rows import (
    build_client_diagnostic_rows,
    build_graph_stats_row,
    build_round_diagnostics_row,
)
from graphfl_lab.strategies.graphfl.counterfactual_artifacts import (
    run_counterfactual_artifacts,
)


def write_round_diagnostic_artifacts(
    *,
    artifact_dir: Path,
    run_id: str,
    variant: str,
    seed: int,
    server_round: int,
    accuracy: float,
    loss: float,
    pre_post: Mapping[str, Any],
    graph_diag: Mapping[str, Any],
    wall_time_sec: float,
    graph_method: str,
    correction_family: str,
    graph_source_used: str,
    graph_variant: str,
    diagnostic_target_used: str,
    graph_used_source: str,
    graph_meta: Mapping[str, Any],
    control_graph_mode: str,
    cluster_method: str,
    cluster_k: int,
    cluster_auto_k: bool,
    cids: Sequence[str],
    n_examples_arr: np.ndarray,
    client_cluster_ids: Sequence[int],
    flat_updates: np.ndarray,
    pre_weights: Sequence[float],
    actual_adjacency: np.ndarray,
    aggregation_target: str,
    graph_seed: int,
    graph_filter_strength: float,
    graph_free_gamma: float,
    loo_enabled: bool,
) -> None:
    graph_kind = str(graph_meta.get("graph_kind", graph_meta.get("kind", "")))
    append_round_metrics_csv(
        artifact_dir / "round_metrics.csv",
        build_round_diagnostics_row(
            run_id=run_id,
            variant=variant,
            seed=int(seed),
            server_round=int(server_round),
            accuracy=float(accuracy),
            loss=float(loss),
            pre_post_round=pre_post["round"],
            graph_diag=graph_diag,
            wall_time_sec=float(wall_time_sec),
            graph_method=str(graph_method),
            correction_family=str(correction_family),
            graph_source=str(graph_source_used),
            graph_variant=str(graph_variant),
            aggregation_target=str(diagnostic_target_used),
            graph_kind=graph_kind,
        ),
    )
    append_graph_stats_csv(
        artifact_dir / "graph_stats.csv",
        build_graph_stats_row(
            run_id=run_id,
            variant=variant,
            seed=int(seed),
            server_round=int(server_round),
            graph_method=str(graph_method),
            correction_family=str(correction_family),
            graph_source=str(graph_source_used),
            graph_variant=str(graph_variant),
            aggregation_target=str(diagnostic_target_used),
            graph_kind=graph_kind,
            graph_used_source=str(graph_used_source),
            graph_diag=graph_diag,
            control_graph_mode=str(control_graph_mode),
            cluster_method=str(cluster_method),
            cluster_k=int(cluster_k),
            cluster_auto_k=bool(cluster_auto_k),
        ),
    )
    append_client_metrics_csv(
        artifact_dir / "client_metrics.csv",
        build_client_diagnostic_rows(
            run_id=run_id,
            variant=variant,
            seed=int(seed),
            server_round=int(server_round),
            cids=cids,
            n_examples_arr=n_examples_arr,
            pre_post=pre_post,
            client_cluster_ids=client_cluster_ids,
        ),
    )
    counterfactual_artifacts = run_counterfactual_artifacts(
        flat_updates=flat_updates,
        weights_pre=pre_weights,
        actual_adjacency=actual_adjacency,
        diagnostic_target_used=str(diagnostic_target_used),
        aggregation_target=aggregation_target,
        diagnostics_seed=int(seed),
        graph_seed=int(graph_seed),
        server_round=int(server_round),
        graph_filter_strength=float(graph_filter_strength),
        graph_free_gamma=float(graph_free_gamma),
        loo_enabled=bool(loo_enabled),
        graph_meta=graph_meta,
        run_id=run_id,
        variant=variant,
        graph_method=str(graph_method),
        graph_variant=str(graph_variant),
    )
    append_counterfactual_metrics_csv(
        artifact_dir / "counterfactual_metrics.csv",
        counterfactual_artifacts.counterfactual_rows,
    )
    append_module_traces_jsonl(
        artifact_dir / "module_traces.jsonl",
        counterfactual_artifacts.module_trace_rows,
    )


__all__ = ["write_round_diagnostic_artifacts"]
