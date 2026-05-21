"""Lifecycle module result and protocol contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .context import AggregationContext, RelationContext, StateExtractionContext, TopologyContext
from .traces import RoundTraceBundle, TraceRecord


MODULE_STATUSES = ("ok", "unsupported", "error")
SUPPORT_LEVELS = (
    "core-supported",
    "proxy-supported",
    "interface-target",
    "out-of-scope",
)


def _validate_label(value: str, *, allowed: tuple[str, ...], field_name: str) -> str:
    label = str(value).strip()
    if label not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")
    return label


def _coerce_trace_records(
    records: Iterable[TraceRecord] | RoundTraceBundle | TraceRecord | None,
) -> tuple[TraceRecord, ...]:
    if records is None:
        return ()
    if isinstance(records, TraceRecord):
        return (records,)
    if isinstance(records, RoundTraceBundle):
        return tuple(records.records)
    coerced = tuple(records)
    for record in coerced:
        if not isinstance(record, TraceRecord):
            raise TypeError("trace_records must contain TraceRecord instances")
    return coerced


@dataclass(frozen=True)
class ModuleResult:
    """Common output envelope returned by lifecycle modules."""

    status: str
    support_level: str
    output: Any = None
    trace_records: Iterable[TraceRecord] | RoundTraceBundle | TraceRecord | None = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _validate_label(self.status, allowed=MODULE_STATUSES, field_name="status"),
        )
        object.__setattr__(
            self,
            "support_level",
            _validate_label(self.support_level, allowed=SUPPORT_LEVELS, field_name="support_level"),
        )
        object.__setattr__(self, "trace_records", _coerce_trace_records(self.trace_records))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.status == "error" and not self.error_type:
            raise ValueError("error results require error_type")

    @classmethod
    def ok(
        cls,
        *,
        output: Any = None,
        support_level: str = "core-supported",
        trace_records: Iterable[TraceRecord] | RoundTraceBundle | TraceRecord | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ModuleResult":
        return cls(
            status="ok",
            support_level=support_level,
            output=output,
            trace_records=trace_records,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def unsupported(
        cls,
        *,
        support_level: str = "interface-target",
        message: str,
        output: Any = None,
        trace_records: Iterable[TraceRecord] | RoundTraceBundle | TraceRecord | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ModuleResult":
        return cls(
            status="unsupported",
            support_level=support_level,
            output=output,
            trace_records=trace_records,
            metadata={} if metadata is None else metadata,
            error_type="UnsupportedModule",
            error_message=str(message),
        )

    @classmethod
    def error(
        cls,
        error: BaseException,
        *,
        support_level: str = "out-of-scope",
        output: Any = None,
        trace_records: Iterable[TraceRecord] | RoundTraceBundle | TraceRecord | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ModuleResult":
        return cls(
            status="error",
            support_level=support_level,
            output=output,
            trace_records=trace_records,
            metadata={} if metadata is None else metadata,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def to_trace_bundle(self) -> RoundTraceBundle:
        bundle = RoundTraceBundle()
        bundle.extend(self.trace_records)
        return bundle


@runtime_checkable
class DeliveryPolicy(Protocol):
    def run(self, context: Any) -> ModuleResult:
        ...


@runtime_checkable
class LocalObjectiveHook(Protocol):
    def run(self, context: Any) -> ModuleResult:
        ...


@runtime_checkable
class ClientStateExtractor(Protocol):
    def run(self, context: StateExtractionContext) -> ModuleResult:
        ...


@runtime_checkable
class RelationEstimator(Protocol):
    def run(self, context: RelationContext) -> ModuleResult:
        ...


@runtime_checkable
class TopologyOperator(Protocol):
    def run(self, context: TopologyContext) -> ModuleResult:
        ...


@runtime_checkable
class AggregationOperator(Protocol):
    def run(self, context: AggregationContext) -> ModuleResult:
        ...


__all__ = [
    "AggregationOperator",
    "ClientStateExtractor",
    "DeliveryPolicy",
    "LocalObjectiveHook",
    "MODULE_STATUSES",
    "ModuleResult",
    "RelationEstimator",
    "SUPPORT_LEVELS",
    "TopologyOperator",
]
