"""Round graph construction state for GraphFL strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from graphfl_lab.graph.builders import build_relation_graph
from graphfl_lab.graph.diagnostics import compute_graph_diagnostics
from graphfl_lab.strategies.graphfl.graph_metadata import client_cluster_ids_from_meta
from graphfl_lab.strategies.graphfl.graph_state import select_round_graph


@dataclass(frozen=True)
class RoundGraphState:
    pre_weights: np.ndarray
    current_graph: np.ndarray
    graph_meta: Dict[str, Any]
    client_cluster_ids: list[int]
    graph_diag_current: Dict[str, Any]
    used_graph: np.ndarray
    graph_used_source: str
    graph_diag: Dict[str, Any]
    graph_fallback_used: bool


def build_round_graph_state(
    *,
    z_mat: np.ndarray,
    cids: Sequence[str],
    n_examples_arr: np.ndarray,
    server_round: int,
    graph_seed: int,
    graph_mode: str,
    knn_k: int,
    edge_threshold: float,
    graph_scale_sigma: float,
    learned_graph_lambda: float,
    correction_family: str,
    control_graph_mode: str,
    graph_source_used: str,
    aggregation_target: str,
    cluster_method: str,
    cluster_k: int,
    cluster_auto_k: bool,
    previous_graph_ema: np.ndarray | None,
    use_ema_graph: bool,
    in_warmup: bool,
    ema_alpha: float,
) -> RoundGraphState:
    graph_rng = np.random.default_rng(int(graph_seed) * 1009 + int(server_round) * 13)
    pre_weights = n_examples_arr / (float(np.sum(n_examples_arr)) + 1e-12)
    current_graph, graph_meta = build_relation_graph(
        z_mat=z_mat,
        mode=graph_mode,
        knn_k=knn_k,
        edge_threshold=edge_threshold,
        rng=graph_rng,
        graph_scale_sigma=graph_scale_sigma,
        learned_graph_lambda=learned_graph_lambda,
        correction_family=correction_family,
        control_graph_mode=control_graph_mode,
        graph_source=graph_source_used,
        aggregation_target=aggregation_target,
        cluster_method=cluster_method,
        cluster_k=cluster_k,
        cluster_auto_k=cluster_auto_k,
        cluster_seed=int(graph_seed) + int(server_round),
        client_sample_weights=pre_weights,
    )
    graph_diag_current = compute_graph_diagnostics(current_graph)
    used_graph, graph_used_source = select_round_graph(
        current_graph=current_graph,
        previous_graph_ema=previous_graph_ema,
        use_ema_graph=use_ema_graph,
        in_warmup=in_warmup,
        ema_alpha=ema_alpha,
    )
    graph_diag = compute_graph_diagnostics(used_graph)
    return RoundGraphState(
        pre_weights=pre_weights,
        current_graph=current_graph,
        graph_meta=graph_meta,
        client_cluster_ids=client_cluster_ids_from_meta(graph_meta, cids),
        graph_diag_current=graph_diag_current,
        used_graph=used_graph,
        graph_used_source=graph_used_source,
        graph_diag=graph_diag,
        graph_fallback_used=bool(graph_diag["graph_empty"]),
    )


__all__ = ["RoundGraphState", "build_round_graph_state"]
