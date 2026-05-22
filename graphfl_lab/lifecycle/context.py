"""Lifecycle context envelopes shared by graph-FL modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence


def _as_tuple(values: Sequence[Any], *, field_name: str) -> tuple[Any, ...]:
    if values is None:
        raise ValueError(f"{field_name} cannot be None")
    return tuple(values)


def _as_int_tuple(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    coerced = tuple(int(value) for value in _as_tuple(values, field_name=field_name))
    if any(value < 0 for value in coerced):
        raise ValueError(f"{field_name} cannot contain negative values")
    return coerced


@dataclass(frozen=True)
class RoundContext:
    """Shared per-round inputs exposed to lifecycle modules."""

    server_round: int
    cids: Sequence[str]
    rng: Any | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    state_store: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        server_round = int(self.server_round)
        if server_round < 0:
            raise ValueError("server_round cannot be negative")
        object.__setattr__(self, "server_round", server_round)
        object.__setattr__(self, "cids", tuple(str(cid) for cid in _as_tuple(self.cids, field_name="cids")))
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class StateExtractionContext:
    """Inputs needed to produce per-client state representations."""

    round_context: RoundContext
    global_weights: Any
    local_weights: Sequence[Any]
    local_updates: Sequence[Any]
    num_examples: Sequence[int]
    client_metrics: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_weights", _as_tuple(self.local_weights, field_name="local_weights"))
        object.__setattr__(self, "local_updates", _as_tuple(self.local_updates, field_name="local_updates"))
        object.__setattr__(self, "num_examples", _as_int_tuple(self.num_examples, field_name="num_examples"))
        object.__setattr__(
            self,
            "client_metrics",
            tuple(dict(metrics) for metrics in self.client_metrics),
        )


@dataclass(frozen=True)
class RelationContext:
    """Inputs needed to estimate relations between client states."""

    round_context: RoundContext
    client_state_output: Any


@dataclass(frozen=True)
class TopologyContext:
    """Inputs needed to transform relations into graph topology."""

    round_context: RoundContext
    relation_output: Any


@dataclass(frozen=True)
class AggregationContext:
    """Inputs needed by aggregation modules after topology construction."""

    round_context: RoundContext
    topology_output: Any
    local_updates: Sequence[Any]
    num_examples: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_updates", _as_tuple(self.local_updates, field_name="local_updates"))
        object.__setattr__(self, "num_examples", _as_int_tuple(self.num_examples, field_name="num_examples"))


def make_round_context(
    *,
    server_round: int,
    cids: Sequence[str],
    rng: Any | None = None,
    config: Mapping[str, Any] | None = None,
    state_store: MutableMapping[str, Any] | None = None,
) -> RoundContext:
    """Build the common context from current strategy round values."""
    return RoundContext(
        server_round=server_round,
        cids=cids,
        rng=rng,
        config={} if config is None else config,
        state_store={} if state_store is None else state_store,
    )


def make_state_extraction_context(
    *,
    server_round: int,
    cids: Sequence[str],
    global_weights: Any,
    local_weights: Sequence[Any],
    local_updates: Sequence[Any],
    num_examples: Sequence[int],
    client_metrics: Sequence[Mapping[str, Any]] = (),
    rng: Any | None = None,
    config: Mapping[str, Any] | None = None,
    state_store: MutableMapping[str, Any] | None = None,
) -> StateExtractionContext:
    """Build a state extraction context from current strategy values."""
    return StateExtractionContext(
        round_context=make_round_context(
            server_round=server_round,
            cids=cids,
            rng=rng,
            config=config,
            state_store=state_store,
        ),
        global_weights=global_weights,
        local_weights=local_weights,
        local_updates=local_updates,
        num_examples=num_examples,
        client_metrics=client_metrics,
    )


__all__ = [
    "AggregationContext",
    "RelationContext",
    "RoundContext",
    "StateExtractionContext",
    "TopologyContext",
    "make_round_context",
    "make_state_extraction_context",
]
