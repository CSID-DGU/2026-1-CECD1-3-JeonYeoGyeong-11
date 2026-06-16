"""Shared argparse value parsers for thin CLI entrypoints."""

from __future__ import annotations

import argparse
import json


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value_s = str(value).strip().lower()
    if value_s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def json_object(value):
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        if "=" not in text:
            raise argparse.ArgumentTypeError(
                f"Expected a JSON object or key=value pairs: {value!r}"
            ) from exc
        parsed = {}
        for item in text.split(","):
            key, separator, raw_value = item.partition("=")
            key = key.strip()
            if not separator or not key:
                raise argparse.ArgumentTypeError(
                    f"Expected key=value pairs: {value!r}"
                ) from exc
            raw_value = raw_value.strip()
            try:
                parsed[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed[key] = raw_value
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


__all__ = ["json_object", "str2bool"]
