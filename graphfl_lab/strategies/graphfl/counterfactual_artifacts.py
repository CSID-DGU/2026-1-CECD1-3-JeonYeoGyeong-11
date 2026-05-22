"""Counterfactual artifact orchestration for GraphFL diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from graphfl_lab.lifecycle.counterfactuals import (
    CounterfactualSpec,
    default_counterfactual_specs,
)
from graphfl_lab.lifecycle.diagnostic_runner import (
    CounterfactualDiagnosticRunner,
    MinimalAggregationAdapter,
)
from graphfl_lab.strategies.graphfl.trace_context import with_run_context


@dataclass(frozen=True)
class CounterfactualArtifactRows:
    counterfactual_rows: list[Dict[str, Any]]
    module_trace_rows: list[Dict[str, Any]]


def counterfactual_specs_for_target(
    diagnostic_target_used: str,
    aggregation_target: str,
) -> tuple[CounterfactualSpec, ...]:
    counterfactual_target = str(diagnostic_target_used or aggregation_target)
    return tuple(
        (
            spec
            if spec.name == "graphfree_dominance_reweight"
            else replace(spec, aggregation_target=counterfactual_target)
        )
        for spec in default_counterfactual_specs()
    )


def counterfactual_seed_base(*, diagnostics_seed: int, graph_seed: int) -> int:
    return int(diagnostics_seed) if int(diagnostics_seed) >= 0 else int(graph_seed)


def run_counterfactual_artifacts(
    *,
    flat_updates: np.ndarray,
    weights_pre: Sequence[float],
    actual_adjacency: np.ndarray,
    diagnostic_target_used: str,
    aggregation_target: str,
    diagnostics_seed: int,
    graph_seed: int,
    server_round: int,
    graph_filter_strength: float,
    graph_free_gamma: float,
    loo_enabled: bool,
    graph_meta: Mapping[str, Any],
    run_id: str,
    variant: str,
    graph_method: str,
    graph_variant: str,
) -> CounterfactualArtifactRows:
    seed_base = counterfactual_seed_base(
        diagnostics_seed=diagnostics_seed,
        graph_seed=graph_seed,
    )
    runner = CounterfactualDiagnosticRunner(
        aggregation_adapter=MinimalAggregationAdapter(
            filter_strength=graph_filter_strength,
            dominance_gamma=graph_free_gamma,
        ),
        loo_enabled=loo_enabled,
        rng=np.random.default_rng(seed_base * 4099 + int(server_round)),
    )
    results = runner.run(
        flat_updates=flat_updates,
        weights_pre=weights_pre,
        actual_adjacency=actual_adjacency,
        specs=counterfactual_specs_for_target(
            diagnostic_target_used=diagnostic_target_used,
            aggregation_target=aggregation_target,
        ),
        round_number=int(server_round),
    )

    counterfactual_rows: list[Dict[str, Any]] = []
    module_trace_rows: list[Dict[str, Any]] = []
    for trace in graph_meta.get("lifecycle_trace", []) or []:
        module_trace_rows.append(
            with_run_context(
                dict(trace),
                round_number=int(server_round),
                run_id=run_id,
                variant=variant,
                seed=int(diagnostics_seed),
            )
        )
    for result in results:
        counterfactual_rows.append(
            {
                "run_id": run_id,
                "variant": variant,
                "seed": int(diagnostics_seed),
                "round": int(server_round),
                "graph_method": str(graph_method),
                "graph_variant": str(graph_variant),
                **dict(result.metrics),
            }
        )
        for trace in result.trace_records:
            module_trace_rows.append(
                with_run_context(
                    trace.to_dict(),
                    round_number=int(server_round),
                    run_id=run_id,
                    variant=variant,
                    seed=int(diagnostics_seed),
                )
            )

    return CounterfactualArtifactRows(
        counterfactual_rows=counterfactual_rows,
        module_trace_rows=module_trace_rows,
    )


__all__ = [
    "CounterfactualArtifactRows",
    "counterfactual_seed_base",
    "counterfactual_specs_for_target",
    "run_counterfactual_artifacts",
]
