"""Small result-file writers shared by suite implementations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def write_json(path: Path, payload: Any, *, allow_nan: bool = True) -> Path:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=allow_nan)
    return path


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    columns = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
