"""Diagnostic artifact row builders for GraphFL strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from graphfl_lab.diagnostics.schema import (
    ClientRoundDiagnostics,
    RoundDiagnostics,
)


def build_round_diagnostics_row(
    *,
    run_id: str,
    variant: str,
    seed: int,
    server_round: int,
    accuracy: float,
    loss: float,
    pre_post_round: Mapping[str, Any],
    graph_diag: Mapping[str, Any],
    wall_time_sec: float,
    graph_method: str,
    correction_family: str,
    graph_source: str,
    graph_variant: str,
    aggregation_target: str,
    graph_kind: str,
) -> Dict[str, object]:
    return RoundDiagnostics(
        run_id=run_id,
        variant=variant,
        seed=int(seed),
        round=int(server_round),
        accuracy=float(accuracy),
        loss=float(loss),
        di_pre=float(pre_post_round["di_pre"]),
        di_post=float(pre_post_round["di_post"]),
        neff_pre=float(pre_post_round["neff_pre"]),
        neff_post=float(pre_post_round["neff_post"]),
        align_mean_pre=float(pre_post_round["align_mean_pre"]),
        align_mean_post=float(pre_post_round["align_mean_post"]),
        loo_mean_pre=float(pre_post_round["loo_mean_pre"]),
        loo_mean_post=float(pre_post_round["loo_mean_post"]),
        graph_density=float(graph_diag["graph_density"]),
        graph_entropy=float(graph_diag["graph_entropy"]),
        alpha_entropy=float(pre_post_round["alpha_entropy"]),
        wall_time_sec=float(wall_time_sec),
        graph_method=str(graph_method),
        correction_family=str(correction_family),
        graph_source=str(graph_source),
        graph_variant=str(graph_variant),
        aggregation_target=str(aggregation_target),
        graph_kind=str(graph_kind),
    ).to_dict()


def build_graph_stats_row(
    *,
    run_id: str,
    variant: str,
    seed: int,
    server_round: int,
    graph_method: str,
    correction_family: str,
    graph_source: str,
    graph_variant: str,
    aggregation_target: str,
    graph_kind: str,
    graph_used_source: str,
    graph_diag: Mapping[str, Any],
    control_graph_mode: str,
    cluster_method: str,
    cluster_k: int,
    cluster_auto_k: bool,
) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "variant": variant,
        "seed": int(seed),
        "round": int(server_round),
        "graph_method": str(graph_method),
        "correction_family": str(correction_family),
        "graph_source": str(graph_source),
        "graph_variant": str(graph_variant),
        "aggregation_target": str(aggregation_target),
        "graph_kind": str(graph_kind),
        "graph_used_source": str(graph_used_source),
        "graph_source_used": str(graph_source),
        "graph_density": float(graph_diag["graph_density"]),
        "graph_entropy": float(graph_diag["graph_entropy"]),
        "graph_num_nodes": int(graph_diag["graph_num_nodes"]),
        "number_of_edges": int(graph_diag["number_of_edges"]),
        "graph_degree_mean": float(graph_diag["graph_degree_mean"]),
        "graph_degree_min": int(graph_diag["graph_degree_min"]),
        "graph_degree_max": int(graph_diag["graph_degree_max"]),
        "graph_empty": bool(graph_diag["graph_empty"]),
        "control_graph_mode": str(control_graph_mode),
        "cluster_method": str(cluster_method),
        "cluster_k_config": int(cluster_k),
        "cluster_auto_k": bool(cluster_auto_k),
    }


def build_client_diagnostic_rows(
    *,
    run_id: str,
    variant: str,
    seed: int,
    server_round: int,
    cids: Sequence[str],
    n_examples_arr: np.ndarray,
    pre_post: Mapping[str, Any],
    client_cluster_ids: Sequence[int],
) -> List[Dict[str, object]]:
    norms_pre = pre_post["norms_pre"]
    norms_post = pre_post["norms_post"]
    rows: List[Dict[str, object]] = []
    for index, cid in enumerate(cids):
        row = ClientRoundDiagnostics(
            run_id=run_id,
            variant=variant,
            seed=int(seed),
            round=int(server_round),
            cid=str(cid),
            num_examples=int(n_examples_arr[index]),
            update_norm_raw=float(norms_pre[index]),
            update_norm_corrected=float(norms_post[index]),
            q_raw=float(pre_post["q_pre"][index]),
            q_corrected=float(pre_post["q_post"][index]),
            alignment_raw=float(pre_post["align_pre"][index]),
            alignment_corrected=float(pre_post["align_post"][index]),
            loo_raw=float(pre_post["loo_pre"][index]),
            loo_corrected=float(pre_post["loo_post"][index]),
            cluster_id=int(client_cluster_ids[index]),
        )
        rows.append(row.to_dict())
    return rows


__all__ = [
    "build_client_diagnostic_rows",
    "build_graph_stats_row",
    "build_round_diagnostics_row",
]
