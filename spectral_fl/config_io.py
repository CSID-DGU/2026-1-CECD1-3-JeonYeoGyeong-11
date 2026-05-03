"""Small JSON config helpers for experiment entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "Optional JSON config file. Values under top-level 'args' become "
            "parser defaults; explicit CLI values override them."
        ),
    )


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    raw_args = data.get("args", data)
    if not isinstance(raw_args, dict):
        raise ValueError(f"Config 'args' must be a JSON object: {config_path}")
    return raw_args


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()
    if not pre_args.config:
        return parser.parse_args()

    valid_dests = {action.dest for action in parser._actions}
    overrides = load_config(pre_args.config)
    normalized: Dict[str, Any] = {}
    unknown = []
    for raw_key, value in overrides.items():
        key = str(raw_key).replace("-", "_")
        if key not in valid_dests:
            unknown.append(str(raw_key))
            continue
        normalized[key] = value

    if unknown:
        parser.error(
            f"Unknown config key(s) in {pre_args.config}: {', '.join(sorted(unknown))}"
        )
    parser.set_defaults(**normalized)
    return parser.parse_args()
