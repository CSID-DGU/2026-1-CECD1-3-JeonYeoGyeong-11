"""Shared suite statistics and result JSON helpers."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value_f):
            out.append(value_f)
    return out


def safe_mean(values, default=float("nan")):
    finite = _finite_values(values)
    return statistics.mean(finite) if finite else default


def safe_min(values, default=float("nan")):
    finite = _finite_values(values)
    return min(finite) if finite else default


def safe_max(values, default=float("nan")):
    finite = _finite_values(values)
    return max(finite) if finite else default


def safe_pstdev(values, default=0.0):
    finite = _finite_values(values)
    if len(finite) < 2:
        return default
    return statistics.pstdev(finite)


def round_trace_field(trace, key):
    return _finite_values(row.get(key) for row in trace)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def final_acc(result_obj: Dict[str, Any], method: str) -> float:
    acc = result_obj["results"][method]["metrics_distributed"]["accuracy"]
    return float(acc[-1][1]) if acc else float("nan")
