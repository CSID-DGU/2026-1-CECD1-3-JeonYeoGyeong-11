"""Round aggregation-weight orchestration for GraphFL strategies."""

from __future__ import annotations

import numpy as np

from graphfl_lab.strategies.graphfl.aggregation import (
    AggregationWeightSelection,
    apply_correction_family,
    select_aggregation_weights,
)


def select_round_weights(
    *,
    n_examples: np.ndarray,
    conflict_weight: np.ndarray,
    diagnostic_only: bool,
    in_warmup: bool,
    estd_disabled: bool,
    graph_fallback_used: bool,
    conflict_mix: float,
    min_client_weight: float,
    correction_family: str,
    graph_free_mode: str,
    graph_free_gamma: float,
    contribution_cap: float,
    clip_quantile: float,
    update_norms: np.ndarray,
) -> AggregationWeightSelection:
    selection = select_aggregation_weights(
        n_examples=n_examples,
        conflict_weight=conflict_weight,
        diagnostic_only=diagnostic_only,
        in_warmup=in_warmup,
        estd_disabled=estd_disabled,
        graph_fallback_used=graph_fallback_used,
        conflict_mix=conflict_mix,
        min_client_weight=min_client_weight,
    )
    return apply_correction_family(
        correction_family=correction_family,
        selection=selection,
        n_examples=n_examples,
        graph_free_mode=graph_free_mode,
        graph_free_gamma=graph_free_gamma,
        contribution_cap=contribution_cap,
        clip_quantile=clip_quantile,
        update_norms=update_norms,
    )


__all__ = ["select_round_weights"]
