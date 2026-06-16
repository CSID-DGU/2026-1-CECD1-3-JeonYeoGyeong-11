"""Shared extension preparation for every execution surface."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Mapping

from graphfl_lab.cli.aggregation_targets import BUILTIN_AGGREGATION_TARGETS
from graphfl_lab.graph.presets import apply_graph_preset_to_namespace
from graphfl_lab.graph.registry import load_graph_plugins
from graphfl_lab.strategies.graphfl.targets import (
    aggregation_target_names,
    canonical_aggregation_target,
)


def prepare_graph_extensions(args: Namespace) -> Mapping[str, Any]:
    """Load plugins before resolving designs and validate custom target names."""
    if bool(getattr(args, "_graph_extensions_prepared", False)):
        return getattr(args, "_graph_extension_info", {})

    plugin_spec = str(getattr(args, "graph_plugin", "") or "")
    loaded = load_graph_plugins(plugin_spec)
    preset_info = apply_graph_preset_to_namespace(args)
    target = canonical_aggregation_target(
        str(getattr(args, "aggregation_target", "update"))
    )
    known_targets = set(BUILTIN_AGGREGATION_TARGETS)
    known_targets.update(aggregation_target_names())
    if target not in known_targets:
        known = ", ".join(sorted(known_targets))
        raise ValueError(
            f"Unknown aggregation_target={target!r}. Known targets: {known}"
        )
    setattr(args, "aggregation_target", target)
    info = {
        "loaded_plugins": tuple(loaded),
        "preset": dict(preset_info),
        "aggregation_target": target,
    }
    setattr(args, "_graph_extensions_prepared", True)
    setattr(args, "_graph_extension_info", info)
    return info


__all__ = ["prepare_graph_extensions"]
