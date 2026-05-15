"""Registry for metadata-first graph-FL designs."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .design import GraphFLDesign, normalize_design_name


class DesignRegistry:
    def __init__(self) -> None:
        self._designs: dict[str, GraphFLDesign] = {}
        self._aliases: dict[str, str] = {}

    def register(self, design: GraphFLDesign, *, override: bool = False) -> GraphFLDesign:
        key = normalize_design_name(design.name)
        if key in self._designs and not bool(override):
            raise ValueError(f"GraphFLDesign {design.name!r} is already registered")
        self._designs[key] = design
        return design

    def register_alias(self, alias: str, target: str, *, override: bool = False) -> None:
        alias_key = normalize_design_name(alias)
        target_key = normalize_design_name(target)
        if not alias_key:
            raise ValueError("design alias cannot be empty")
        if alias_key in self._aliases and not bool(override):
            raise ValueError(f"GraphFLDesign alias {alias!r} is already registered")
        self._aliases[alias_key] = target_key

    def resolve(self, name: str) -> GraphFLDesign:
        key = normalize_design_name(name)
        target = self._aliases.get(key, key)
        if target not in self._designs:
            known = ", ".join(self.names(include_aliases=True))
            raise ValueError(f"Unknown GraphFLDesign {name!r}. Known designs: {known}")
        design = self._designs[target]
        if key != target:
            legacy_args = design.to_legacy_args()
            alias_tags = tuple(dict.fromkeys((*design.tags, "compat-alias")))
            return replace(
                design,
                tags=alias_tags,
                client_state=replace(
                    design.client_state,
                    params={**design.client_state.params, **legacy_args},
                ),
            )
        return design

    def names(self, *, include_aliases: bool = False) -> list[str]:
        names = set(self._designs)
        if include_aliases:
            names.update(self._aliases)
        return sorted(names)


_DEFAULT_REGISTRY = DesignRegistry()
_BUILTINS_LOADED = False


def _ensure_builtins_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .presets import builtin_aliases, builtin_designs

    for design in builtin_designs().values():
        _DEFAULT_REGISTRY.register(design, override=True)
    for alias, target in builtin_aliases().items():
        _DEFAULT_REGISTRY.register_alias(alias, target, override=True)
    _BUILTINS_LOADED = True


def register_design(design: GraphFLDesign, *, override: bool = False) -> GraphFLDesign:
    _ensure_builtins_loaded()
    return _DEFAULT_REGISTRY.register(design, override=override)


def register_design_alias(alias: str, target: str, *, override: bool = False) -> None:
    _ensure_builtins_loaded()
    _DEFAULT_REGISTRY.register_alias(alias, target, override=override)


def resolve_design(name: str) -> GraphFLDesign:
    _ensure_builtins_loaded()
    return _DEFAULT_REGISTRY.resolve(name)


def design_names(*, include_aliases: bool = False) -> list[str]:
    _ensure_builtins_loaded()
    return _DEFAULT_REGISTRY.names(include_aliases=include_aliases)


def design_registry_snapshot() -> Mapping[str, GraphFLDesign]:
    _ensure_builtins_loaded()
    return {name: resolve_design(name) for name in design_names()}


__all__ = [
    "DesignRegistry",
    "design_names",
    "design_registry_snapshot",
    "register_design",
    "register_design_alias",
    "resolve_design",
]
