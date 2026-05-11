"""Append-safe CSV artifact writers for diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


def init_artifact_dir(base_dir: str | Path) -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_round_metrics_csv(path: str | Path, row: Dict[str, object]) -> None:
    _append_rows(Path(path), [row])


def append_client_metrics_csv(path: str | Path, rows: Iterable[Dict[str, object]]) -> None:
    _append_rows(Path(path), rows)


__all__ = [
    "append_client_metrics_csv",
    "append_round_metrics_csv",
    "init_artifact_dir",
]
