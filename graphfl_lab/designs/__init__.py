"""Composable graph-FL design metadata."""

from .design import ComponentSpec, GraphFLDesign, normalize_design_name
from .prior_work import interface_target_designs, prior_work_design_aliases
from .registry import (
    DesignRegistry,
    design_names,
    design_registry_snapshot,
    register_design,
    register_design_alias,
    resolve_design,
)

__all__ = [
    "ComponentSpec",
    "DesignRegistry",
    "GraphFLDesign",
    "design_names",
    "design_registry_snapshot",
    "interface_target_designs",
    "normalize_design_name",
    "prior_work_design_aliases",
    "register_design",
    "register_design_alias",
    "resolve_design",
]
