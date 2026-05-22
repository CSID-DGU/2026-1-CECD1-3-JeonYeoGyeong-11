"""Suite artifact filename discovery for vision experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

SUITE_ROWS_JSON_FILENAMES = (
    "vision_suite_rows.json",
    "suite_rows.json",
)
SUITE_SUMMARY_CSV_FILENAMES = (
    "vision_suite_summary.csv",
    "suite_summary.csv",
)
SUITE_SUMMARY_JSON_FILENAMES = (
    "vision_suite_summary.json",
    "suite_summary.json",
)
RESULT_JSON_GLOBS = (("result_vision_*.json", "result_vision_"),)


def resolve_suite_artifact(out_dir: Path, filenames: Iterable[str]) -> Path | None:
    """Return the first existing suite artifact from canonical-to-legacy order."""
    for name in filenames:
        path = out_dir / name
        if path.is_file():
            return path
    return None


def discover_result_json_paths(suite_dir: Path) -> dict[str, Path]:
    """Map result JSON suffix keys to paths under ``result_vision_*.json``."""
    result_paths: dict[str, Path] = {}
    for glob_pattern, prefix in RESULT_JSON_GLOBS:
        for path in sorted(suite_dir.glob(glob_pattern)):
            result_paths[path.name.replace(prefix, "", 1)] = path
    return result_paths


def load_suite_rows_json(out_dir: Path) -> list[dict[str, Any]]:
    """Load suite row metadata from canonical or short legacy JSON artifacts."""
    path = resolve_suite_artifact(out_dir, SUITE_ROWS_JSON_FILENAMES)
    if path is None:
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return rows if isinstance(rows, list) else []


__all__ = [
    "RESULT_JSON_GLOBS",
    "SUITE_ROWS_JSON_FILENAMES",
    "SUITE_SUMMARY_CSV_FILENAMES",
    "SUITE_SUMMARY_JSON_FILENAMES",
    "discover_result_json_paths",
    "load_suite_rows_json",
    "resolve_suite_artifact",
]
