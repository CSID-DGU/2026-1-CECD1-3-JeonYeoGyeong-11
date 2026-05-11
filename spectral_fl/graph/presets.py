"""Reusable graph-design presets inspired by prior FL literature.

These presets are lightweight, configurable approximations so experiments can
quickly compose representation + similarity + topology choices.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, List, Mapping

from spectral_fl.graph.sources.config import normalize_key


_PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    # Baselines
    "raw_update_positive_dense": {
        "graph_source": "update",
        "graph_mode": "dense",
        "knn_k": 2,
    },
    "raw_update_positive_knn": {
        "graph_source": "update",
        "graph_mode": "knn",
        "knn_k": 2,
    },
    "signed_conflict_knn": {
        "graph_source": "update",
        "graph_mode": "signed_abs_knn",
        "knn_k": 2,
    },
    # Prior-work-inspired configurations (approximate, not exact reproductions)
    "pfedgraph_like": {
        "graph_source": "weight",
        "graph_mode": "magnitude_knn",
        "knn_k": 2,
    },
    "fedamp_like": {
        "graph_source": "weight",
        "graph_mode": "global_alignment",
        "knn_k": 2,
    },
    "pfedsim_like": {
        "graph_source": "classifier_head_weight",
        "graph_mode": "signed_abs_knn",
        "knn_k": 2,
    },
    "fedaga_like": {
        "graph_source": "ema_update",
        "graph_mode": "magnitude_knn",
        "knn_k": 2,
        "use_ema_graph": True,
        "client_update_ema_alpha": 0.9,
    },
    "gfedfilt_like": {
        "graph_source": "weight",
        "graph_mode": "rbf_knn",
        "knn_k": 2,
        "graph_laplacian_type": "normalized",
        "graph_smoothing_operator": "laplacian",
    },
}


def graph_preset_names() -> List[str]:
    return ["none"] + sorted(_PRESET_SPECS.keys())


def resolve_graph_preset_spec(name: str) -> Dict[str, Any]:
    key = normalize_key(name)
    if key in {"", "none", "off", "disabled"}:
        return {}
    if key not in _PRESET_SPECS:
        known = ", ".join(sorted(_PRESET_SPECS.keys()))
        raise ValueError(f"Unknown graph_preset={name!r}. Known presets: {known}")
    return dict(_PRESET_SPECS[key])


def apply_graph_preset_to_namespace(args: Namespace) -> Mapping[str, Any]:
    """Apply graph preset into an argparse namespace once (idempotent)."""
    if bool(getattr(args, "_graph_preset_applied", False)):
        return getattr(args, "_graph_preset_info", {"graph_preset": "none", "applied": {}})

    preset_raw = str(getattr(args, "graph_preset", "none"))
    preset_key = normalize_key(preset_raw)
    spec = resolve_graph_preset_spec(preset_key)
    applied: Dict[str, Any] = {}
    for key, value in spec.items():
        setattr(args, key, value)
        applied[key] = value

    info = {"graph_preset": preset_key if preset_key else "none", "applied": applied}
    setattr(args, "_graph_preset_applied", True)
    setattr(args, "_graph_preset_info", info)
    setattr(args, "graph_preset", info["graph_preset"])
    return info

