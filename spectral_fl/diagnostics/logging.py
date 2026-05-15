"""Append-safe CSV artifact writers for diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def init_artifact_dir(base_dir: str | Path) -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


CsvRow = Mapping[str, object]


def _load_existing_csv(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _merged_fieldnames(existing: Iterable[str], rows: Iterable[CsvRow]) -> list[str]:
    fieldnames = list(existing)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _append_rows(path: Path, rows: Iterable[CsvRow]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        fieldnames = _merged_fieldnames([], rows)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    existing_fields, existing_rows = _load_existing_csv(path)
    fieldnames = _merged_fieldnames(existing_fields, rows)
    if fieldnames == existing_fields:
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
        return

    rows_to_write: list[CsvRow] = [*existing_rows, *rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)


def append_round_metrics_csv(path: str | Path, row: CsvRow) -> None:
    _append_rows(Path(path), [row])


def append_client_metrics_csv(path: str | Path, rows: Iterable[CsvRow]) -> None:
    _append_rows(Path(path), rows)


def append_graph_stats_csv(path: str | Path, row: CsvRow) -> None:
    _append_rows(Path(path), [row])


def append_counterfactual_metrics_csv(
    path: str | Path,
    rows: Iterable[CsvRow],
) -> None:
    _append_rows(Path(path), rows)


def append_module_traces_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any] | Any],
) -> None:
    rows = []
    for record in records:
        if hasattr(record, "to_dict"):
            payload = record.to_dict()
        else:
            payload = dict(record)
        rows.append(payload)
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


__all__ = [
    "append_client_metrics_csv",
    "append_counterfactual_metrics_csv",
    "append_graph_stats_csv",
    "append_module_traces_jsonl",
    "append_round_metrics_csv",
    "init_artifact_dir",
]
