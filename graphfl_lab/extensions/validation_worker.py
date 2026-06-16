"""Isolated component contract validation worker."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from graphfl_lab.graph import (
    build_relation_graph,
    graph_mode_names,
    graph_source_names,
    load_graph_plugins,
)
from graphfl_lab.graph.sources import (
    GraphSourceConfig,
    resolve_graph_source_result,
)
from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregation_target_names,
    evaluate_aggregation_target,
)


def _sample_arrays():
    updates = [
        [np.asarray([1.0, -0.5, 0.25], dtype=np.float64)],
        [np.asarray([0.8, -0.4, 0.1], dtype=np.float64)],
        [np.asarray([-0.2, 0.9, 0.5], dtype=np.float64)],
        [np.asarray([-0.1, 0.7, 0.6], dtype=np.float64)],
    ]
    current = [np.asarray([2.0, 3.0, 4.0], dtype=np.float64)]
    weights = [[current[0] + client[0]] for client in updates]
    return current, weights, updates


def _checks(
    metadata: dict[str, Any],
    *,
    shape_ok: bool,
    finite_ok: bool,
    required_metadata: tuple[str, ...],
):
    metadata_ok = all(key in metadata for key in required_metadata)
    trace_ok = all(
        key in metadata
        for key in (
            "component_kind",
            "component_name",
            "plugin_module",
            "input_shape",
            "output_shape",
        )
    )
    return [
        {"name": "Registry", "pass": True},
        {"name": "Shape", "pass": bool(shape_ok)},
        {"name": "Finite", "pass": bool(finite_ok)},
        {"name": "Metadata", "pass": bool(metadata_ok)},
        {"name": "Trace/Artifact", "pass": bool(trace_ok)},
    ]


def _validate_source(name: str):
    _, weights, updates = _sample_arrays()
    result = resolve_graph_source_result(
        local_weights=weights,
        local_updates=updates,
        config=GraphSourceConfig(source=name),
    )
    vectors = [np.asarray(vector) for vector in result.vectors]
    width = vectors[0].size if vectors else 0
    metadata = dict(result.metadata or {})
    checks = _checks(
        metadata,
        shape_ok=len(vectors) == len(updates)
        and all(vector.ndim == 1 and vector.size == width for vector in vectors),
        finite_ok=all(np.all(np.isfinite(vector)) for vector in vectors),
        required_metadata=("parameters", "source_used"),
    )
    return checks, metadata


def _validate_builder(name: str):
    z_mat = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float64,
    )
    adjacency, metadata_raw = build_relation_graph(
        z_mat=z_mat,
        mode=name,
        knn_k=2,
        graph_source="update",
        aggregation_target="graph_filtered_update",
    )
    metadata = dict(metadata_raw)
    checks = _checks(
        metadata,
        shape_ok=adjacency.shape == (4, 4)
        and np.allclose(adjacency, adjacency.T)
        and np.allclose(np.diag(adjacency), 0.0),
        finite_ok=bool(np.all(np.isfinite(adjacency)))
        and bool(np.all(adjacency >= 0.0)),
        required_metadata=("parameters", "source_used", "target_used"),
    )
    return checks, metadata


def _validate_aggregation(name: str):
    current, weights, updates = _sample_arrays()
    laplacian = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    evaluation = evaluate_aggregation_target(
        current_global=current,
        local_weights=weights,
        local_updates=updates,
        alpha_norm=np.full(4, 0.25, dtype=np.float64),
        l_mat=laplacian,
        config=AggregationTargetConfig(
            target=name,
            parameters={"beta": 0.5},
        ),
    )
    metadata = dict(evaluation.metadata)
    checks = _checks(
        metadata,
        shape_ok=len(evaluation.candidate_global) == len(current)
        and all(
            out.shape == expected.shape
            for out, expected in zip(evaluation.candidate_global, current)
        )
        and evaluation.post_flat_updates.shape == (4, 3),
        finite_ok=bool(np.all(np.isfinite(evaluation.post_flat_updates)))
        and all(np.all(np.isfinite(layer)) for layer in evaluation.candidate_global),
        required_metadata=("parameters", "target_used"),
    )
    return checks, metadata


def _parse_component(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError("component must use kind:name")
    kind, name = value.split(":", 1)
    kind = kind.strip().lower()
    name = name.strip()
    if kind not in {"source", "builder", "aggregation"} or not name:
        raise ValueError("component must use source|builder|aggregation:name")
    return kind, name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--component", default="")
    args = parser.parse_args(argv)

    before = {
        "source": set(graph_source_names()),
        "builder": set(graph_mode_names()),
        "aggregation": set(aggregation_target_names()),
    }
    loaded = load_graph_plugins(args.plugin)
    after = {
        "source": set(graph_source_names()),
        "builder": set(graph_mode_names()),
        "aggregation": set(aggregation_target_names()),
    }
    if args.component:
        kind, name = _parse_component(args.component)
    else:
        additions = [
            (kind, name)
            for kind in before
            for name in sorted(after[kind] - before[kind])
        ]
        if len(additions) != 1:
            raise ValueError(
                "component is required when the plugin registers zero or multiple components"
            )
        kind, name = additions[0]

    if name not in after[kind]:
        raise ValueError(f"{kind} component {name!r} is not registered")
    validator = {
        "source": _validate_source,
        "builder": _validate_builder,
        "aggregation": _validate_aggregation,
    }[kind]
    checks, metadata = validator(name)
    payload = {
        "schema_version": 1,
        "ok": all(item["pass"] for item in checks),
        "component": {"kind": kind, "name": name},
        "loaded_plugins": loaded,
        "checks": checks,
        "metadata": metadata,
    }
    print(json.dumps(payload, ensure_ascii=True, allow_nan=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
