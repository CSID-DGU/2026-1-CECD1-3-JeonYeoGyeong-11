"""Compatibility wrappers for graph-FL design presets."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, List, Mapping

from graphfl_lab.designs import design_names, resolve_design
from graphfl_lab.graph.method_specs import (
    get_graph_fl_method_spec,
    graph_fl_method_names,
)
from graphfl_lab.graph.sources.config import normalize_key


_DISABLED_NAMES = {"", "none", "off", "disabled"}


def graph_preset_names() -> List[str]:
    return ["none"] + design_names(include_aliases=True)


def graph_method_names() -> List[str]:
    names = set(design_names(include_aliases=True))
    names.update(graph_fl_method_names())
    return ["none"] + sorted(names)


def resolve_graph_preset_spec(name: str) -> Dict[str, Any]:
    key = normalize_key(name)
    if key in _DISABLED_NAMES:
        return {}
    try:
        return resolve_design(key).to_legacy_args()
    except ValueError as exc:
        known = ", ".join(graph_preset_names())
        raise ValueError(f"Unknown graph_preset={name!r}. Known presets: {known}") from exc


def resolve_graph_method_spec(name: str) -> Dict[str, Any]:
    key = normalize_key(name)
    if key in _DISABLED_NAMES:
        return {}

    try:
        spec = resolve_design(key).to_legacy_args()
        spec.setdefault("graph_method", key)
        return spec
    except ValueError:
        pass

    try:
        method = get_graph_fl_method_spec(key)
    except ValueError as exc:
        known = ", ".join(graph_method_names())
        raise ValueError(f"Unknown graph_method={name!r}. Known methods: {known}") from exc

    if method.support_level == "interface-target":
        raise ValueError(
            f"graph_method={name!r} is an interface target, not a runnable method. "
            "Use a supported method or provide the missing source/aggregation plugin."
        )

    if method.design_name:
        try:
            spec = resolve_design(method.design_name).to_legacy_args()
        except ValueError:
            spec = dict(method.config_overrides)
            spec["graph_design"] = method.design_name
    else:
        spec = dict(method.config_overrides)
    spec["graph_method"] = method.name
    return spec


def apply_graph_preset_to_namespace(args: Namespace) -> Mapping[str, Any]:
    """Apply graph preset or graph method into an argparse namespace once."""
    if bool(getattr(args, "_graph_preset_applied", False)):
        return getattr(
            args,
            "_graph_preset_info",
            {"graph_preset": "none", "graph_method": "none", "applied": {}},
        )

    preset_raw = str(getattr(args, "graph_preset", "none"))
    preset_key = normalize_key(preset_raw)
    method_raw = str(getattr(args, "graph_method", "none"))
    method_key = normalize_key(method_raw)

    source = "none"
    skip_keys: set[str] = set()
    if preset_key not in _DISABLED_NAMES:
        source = "graph_preset"
        spec = resolve_graph_preset_spec(preset_key)
    elif method_key not in _DISABLED_NAMES:
        source = "graph_method"
        spec = resolve_graph_method_spec(method_key)
        skip_keys = set(getattr(args, "_user_arg_dests", ())) - {
            "graph_method",
            "graph_preset",
        }
    else:
        spec = {}

    applied: Dict[str, Any] = {}
    for key, value in spec.items():
        if key in skip_keys:
            continue
        setattr(args, key, value)
        applied[key] = value

    if not hasattr(args, "graph_method"):
        setattr(args, "graph_method", "none")

    info = {
        "source": source,
        "graph_preset": preset_key if preset_key else "none",
        "graph_method": str(getattr(args, "graph_method", "none")),
        "applied": applied,
    }
    setattr(args, "_graph_preset_applied", True)
    setattr(args, "_graph_preset_info", info)
    setattr(args, "graph_preset", info["graph_preset"])
    return info

