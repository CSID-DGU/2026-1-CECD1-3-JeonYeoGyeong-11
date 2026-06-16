"""Metadata-first graph-FL design objects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


_LEGACY_ARG_KEYS = {
    "aggregation_target",
    "aggregation_params",
    "client_update_ema_alpha",
    "correction_family",
    "graph_free_mode",
    "graph_laplacian_type",
    "graph_method",
    "graph_mode",
    "graph_scale_sigma",
    "graph_smoothing_operator",
    "knn_k",
}

SUPPORT_LEVELS = (
    "core-supported",
    "proxy-supported",
    "interface-target",
    "out-of-scope",
)


def normalize_design_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _validate_support_level(value: str) -> str:
    support_level = str(value).strip()
    if support_level not in SUPPORT_LEVELS:
        raise ValueError(f"support_level must be one of {SUPPORT_LEVELS}, got {value!r}")
    return support_level


def _as_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class ComponentSpec:
    """Declarative metadata for one lifecycle component slot."""

    kind: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    support_level: str = "core-supported"
    is_learned: bool = False
    is_stateful: bool = False
    input_kind: tuple[str, ...] = field(default_factory=tuple)
    output_kind: str = ""
    trace_keys: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        name = normalize_design_name(self.name)
        if not kind:
            raise ValueError("component kind cannot be empty")
        if not name:
            raise ValueError("component name cannot be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "support_level", _validate_support_level(self.support_level))
        object.__setattr__(self, "is_learned", bool(self.is_learned))
        object.__setattr__(self, "is_stateful", bool(self.is_stateful))
        object.__setattr__(self, "input_kind", _as_tuple(self.input_kind))
        object.__setattr__(self, "output_kind", str(self.output_kind))
        object.__setattr__(self, "trace_keys", _as_tuple(self.trace_keys))

    def trace_values(self, *, prefix: str | None = None) -> dict[str, Any]:
        base = "" if prefix is None else f"{prefix}."
        return {
            f"{base}kind": self.kind,
            f"{base}name": self.name,
            f"{base}support_level": self.support_level,
            f"{base}is_learned": self.is_learned,
            f"{base}is_stateful": self.is_stateful,
            f"{base}input_kind": self.input_kind,
            f"{base}output_kind": self.output_kind,
            f"{base}trace_keys": self.trace_keys,
        }


def _component(
    kind: str,
    name: str,
    *,
    support_level: str = "core-supported",
    params: Mapping[str, Any] | None = None,
    is_learned: bool = False,
    is_stateful: bool = False,
    input_kind: tuple[str, ...] = (),
    output_kind: str = "",
    trace_keys: tuple[str, ...] = (),
) -> ComponentSpec:
    return ComponentSpec(
        kind=kind,
        name=name,
        params={} if params is None else params,
        support_level=support_level,
        is_learned=is_learned,
        is_stateful=is_stateful,
        input_kind=input_kind,
        output_kind=output_kind,
        trace_keys=trace_keys,
    )


DEFAULT_DELIVERY = _component(
    "DeliveryPolicy",
    "global_model",
    output_kind="global_weights",
)
DEFAULT_LOCAL_OBJECTIVE = _component(
    "LocalObjectiveHook",
    "standard_local_training",
    output_kind="local_update",
)
DEFAULT_STATE_STORE = _component(
    "StateStore",
    "none",
    output_kind="stateless",
)
DEFAULT_DIAGNOSTICS = _component(
    "DiagnosticProtocol",
    "standard_graph_diagnostics",
    input_kind=("topology",),
    output_kind="diagnostics",
    trace_keys=("graph_density", "graph_entropy", "di_pre", "di_post", "neff_pre", "neff_post"),
)


@dataclass(frozen=True)
class GraphFLDesign:
    """Composable lifecycle-level description of a graph-FL method."""

    name: str
    client_state: ComponentSpec
    relation: ComponentSpec
    topology: ComponentSpec
    aggregation: ComponentSpec
    delivery: ComponentSpec = DEFAULT_DELIVERY
    local_objective: ComponentSpec = DEFAULT_LOCAL_OBJECTIVE
    state_store: ComponentSpec = DEFAULT_STATE_STORE
    diagnostics: ComponentSpec = DEFAULT_DIAGNOSTICS
    support_level: str = "core-supported"
    tags: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    references: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = normalize_design_name(self.name)
        if not name:
            raise ValueError("design name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "support_level", _validate_support_level(self.support_level))
        object.__setattr__(self, "tags", _as_tuple(self.tags))
        object.__setattr__(self, "references", _as_tuple(self.references))
        object.__setattr__(self, "description", str(self.description))

    def components(self) -> dict[str, ComponentSpec]:
        return {
            "client_state": self.client_state,
            "relation": self.relation,
            "topology": self.topology,
            "aggregation": self.aggregation,
            "delivery": self.delivery,
            "local_objective": self.local_objective,
            "state_store": self.state_store,
            "diagnostics": self.diagnostics,
        }

    def to_legacy_args(self) -> dict[str, Any]:
        """Return current strategy/CLI knobs represented by this design."""
        args: dict[str, Any] = {"graph_design": self.name}
        for component in self.components().values():
            for key, value in component.params.items():
                if key == "graph_source":
                    args[key] = value
                elif key in _LEGACY_ARG_KEYS:
                    args[key] = value
                elif key.startswith("legacy."):
                    args[key[len("legacy."):]] = value
        return args

    def trace_metadata(self) -> dict[str, Any]:
        values = {
            "design_name": self.name,
            "support_level": self.support_level,
            "design_tags": self.tags,
        }
        for slot, component in self.components().items():
            values[f"{slot}.name"] = component.name
            values[f"{slot}.kind"] = component.kind
            values[f"{slot}.support_level"] = component.support_level
        return values

    def _with_component(
        self,
        slot: str,
        component: ComponentSpec | None,
        changes: Mapping[str, Any],
    ) -> "GraphFLDesign":
        current = getattr(self, slot)
        replacement = component if component is not None else replace(current, **dict(changes))
        return replace(self, **{slot: replacement})

    def with_client_state(self, component: ComponentSpec | None = None, **changes: Any) -> "GraphFLDesign":
        return self._with_component("client_state", component, changes)

    def with_relation(self, component: ComponentSpec | None = None, **changes: Any) -> "GraphFLDesign":
        return self._with_component("relation", component, changes)

    def with_topology(self, component: ComponentSpec | None = None, **changes: Any) -> "GraphFLDesign":
        return self._with_component("topology", component, changes)

    def with_aggregation(self, component: ComponentSpec | None = None, **changes: Any) -> "GraphFLDesign":
        return self._with_component("aggregation", component, changes)

    def with_diagnostics(self, component: ComponentSpec | None = None, **changes: Any) -> "GraphFLDesign":
        return self._with_component("diagnostics", component, changes)


__all__ = [
    "ComponentSpec",
    "DEFAULT_DELIVERY",
    "DEFAULT_DIAGNOSTICS",
    "DEFAULT_LOCAL_OBJECTIVE",
    "DEFAULT_STATE_STORE",
    "GraphFLDesign",
    "normalize_design_name",
]
