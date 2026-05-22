"""Standard trace records for lifecycle-style graph-FL experiments."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


TRACE_SCHEMA_VERSION = "lifecycle_trace_v1"
_DEFAULT_MAX_ARRAY_VALUES = 16


def _require_label(value: str, field_name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{field_name} cannot be empty")
    return label


def _json_safe_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _array_summary(array: np.ndarray, *, max_array_values: int) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.asarray([])
    summary: dict[str, Any] = {
        "type": "ndarray",
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if finite.size:
        numeric = finite.astype(np.float64, copy=False)
        summary.update(
            {
                "mean": _json_safe_float(float(np.mean(numeric))),
                "std": _json_safe_float(float(np.std(numeric))),
                "min": _json_safe_float(float(np.min(numeric))),
                "max": _json_safe_float(float(np.max(numeric))),
                "norm": _json_safe_float(float(np.linalg.norm(numeric))),
            }
        )
    if arr.size <= int(max_array_values):
        summary["values"] = json_safe(arr.tolist(), max_array_values=max_array_values)
    return summary


def json_safe(value: Any, *, max_array_values: int = _DEFAULT_MAX_ARRAY_VALUES) -> Any:
    """Return a value that can be serialized with strict JSON settings."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return _json_safe_float(value)
    if isinstance(value, np.generic):
        return json_safe(value.item(), max_array_values=max_array_values)
    if isinstance(value, np.ndarray):
        return _array_summary(value, max_array_values=max_array_values)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value), max_array_values=max_array_values)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item, max_array_values=max_array_values)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item, max_array_values=max_array_values) for item in value]
    if isinstance(value, set):
        return [json_safe(item, max_array_values=max_array_values) for item in sorted(value, key=str)]
    return str(value)


@dataclass(frozen=True)
class TraceRecord:
    phase: str
    module: str
    name: str
    values: Mapping[str, Any] = field(default_factory=dict)
    round: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _require_label(self.phase, "phase"))
        object.__setattr__(self, "module", _require_label(self.module, "module"))
        object.__setattr__(self, "name", _require_label(self.name, "name"))
        if self.round is not None and int(self.round) < 0:
            raise ValueError("round cannot be negative")
        if self.round is not None:
            object.__setattr__(self, "round", int(self.round))
        object.__setattr__(self, "values", dict(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "phase": self.phase,
            "module": self.module,
            "name": self.name,
            "round": self.round,
            "values": json_safe(self.values),
        }


@dataclass
class RoundTraceBundle:
    records: list[TraceRecord] = field(default_factory=list)
    schema_version: str = TRACE_SCHEMA_VERSION

    def add(self, record: TraceRecord) -> TraceRecord:
        if not isinstance(record, TraceRecord):
            raise TypeError("RoundTraceBundle.add expects a TraceRecord")
        self.records.append(record)
        return record

    def extend(self, records: Iterable[TraceRecord]) -> None:
        for record in records:
            self.add(record)

    def by_phase(self, phase: str) -> list[TraceRecord]:
        phase_key = _require_label(phase, "phase")
        return [record for record in self.records if record.phase == phase_key]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def to_flat_dict(self, *, prefix: bool = True) -> dict[str, Any]:
        flat: dict[str, Any] = {
            "trace_schema_version": self.schema_version,
            "trace_record_count": int(len(self.records)),
        }
        for index, record in enumerate(self.records):
            label = f"{record.phase}.{record.module}.{record.name}" if prefix else record.name
            base = f"trace.{index}.{label}"
            flat[f"{base}.round"] = record.round
            for key, value in record.values.items():
                flat[f"{base}.{key}"] = json_safe(value)
        return flat
