"""Shared argparse value parsers for thin CLI entrypoints."""

from __future__ import annotations

import argparse


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


__all__ = ["str2bool"]
