"""Normalized golden-output comparison helpers for cleanup Gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VOLATILE_FIELDS = {
    "timestamp",
    "started_at",
    "completed_at",
    "finished_at",
    "wall_time_sec",
    "total_wall_time_sec",
    "duration_seconds",
    "run_wall_time_sec",
    "seconds_per_round",
    "absolute_path",
    "output_path",
    "canonical_output_path",
    "compatibility_output_path",
    "out_dir",
    "base_dir",
    "diagnostics_dir",
    "plots_dir",
    "reports_dir",
    "snapshots_dir",
    "logs_dir",
    "run_id",
    "host",
    "hostname",
    "python_version",
    "cuda_available",
    "device",
}

REQUIRED_RESULT_SCHEMA_KEYS = (
    "result_schema_version",
    "config_aliases_used",
    "unsupported_components",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_payload(value: Any, volatile_fields: set[str] | None = None) -> Any:
    fields = VOLATILE_FIELDS if volatile_fields is None else volatile_fields
    if isinstance(value, dict):
        return {
            key: normalize_payload(item, fields)
            for key, item in sorted(value.items())
            if key not in fields
        }
    if isinstance(value, list):
        return [normalize_payload(item, fields) for item in value]
    return value


def schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "keys": {key: schema_shape(item) for key, item in sorted(value.items())},
        }
    if isinstance(value, list):
        return {
            "__type__": "list",
            "items": [schema_shape(item) for item in value],
        }
    return type(value).__name__


def _missing_required_keys(payload: Any, label: str) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{label}: top-level payload is not an object"]
    return [
        f"{label}: missing required result schema key {key!r}"
        for key in REQUIRED_RESULT_SCHEMA_KEYS
        if key not in payload
    ]


def compare_payloads(expected: Any, actual: Any) -> list[str]:
    failures: list[str] = []
    failures.extend(_missing_required_keys(expected, "expected"))
    failures.extend(_missing_required_keys(actual, "actual"))
    if schema_shape(expected) != schema_shape(actual):
        failures.append("schema shape differs")
    if normalize_payload(expected) != normalize_payload(actual):
        failures.append("normalized payload differs")
    return failures


def compare_files(expected_path: Path, actual_path: Path) -> list[str]:
    return compare_payloads(load_json(expected_path), load_json(actual_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args(argv)

    failures = compare_files(args.expected, args.actual)
    if failures:
        print(json.dumps({"pass": False, "failed_checks": failures}, indent=2))
        return 1
    print(json.dumps({"pass": True, "failed_checks": []}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
