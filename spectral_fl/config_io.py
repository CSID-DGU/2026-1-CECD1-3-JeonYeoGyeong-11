"""Small JSON config helpers for experiment entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ARG_DEST_ALIASES = {
    "spectral_filter_strength": "graph_filter_strength",
}


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


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path, including old ``configs/general`` aliases."""
    config_path = Path(path)
    if config_path.exists():
        return config_path

    parts = config_path.parts
    lowered = [part.lower() for part in parts]
    for index, part in enumerate(lowered[:-1]):
        if part == "configs" and lowered[index + 1] == "general":
            alt_parts = list(parts)
            alt_parts[index + 1] = "vision"
            alt_path = Path(*alt_parts)
            if alt_path.exists():
                return alt_path
            break
    return config_path


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = resolve_config_path(path)
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
    explicit_dests = _explicit_arg_dests(parser, sys.argv[1:])
    if not pre_args.config:
        return _attach_arg_origin_metadata(
            normalize_arg_aliases(parser.parse_args()),
            explicit_dests=explicit_dests,
            config_dests=set(),
        )

    valid_dests = {action.dest for action in parser._actions}
    overrides = load_config(pre_args.config)
    normalized: Dict[str, Any] = {}
    unknown = []
    for raw_key, value in overrides.items():
        key = str(raw_key).replace("-", "_")
        key = ARG_DEST_ALIASES.get(key, key)
        if key not in valid_dests:
            unknown.append(str(raw_key))
            continue
        normalized[key] = value

    if unknown:
        parser.error(
            f"Unknown config key(s) in {pre_args.config}: {', '.join(sorted(unknown))}"
        )
    parser.set_defaults(**normalized)
    return _attach_arg_origin_metadata(
        normalize_arg_aliases(parser.parse_args()),
        explicit_dests=explicit_dests,
        config_dests=set(normalized),
    )


def normalize_arg_aliases(args: argparse.Namespace) -> argparse.Namespace:
    """Populate canonical/compatibility argparse attributes after parsing."""
    if hasattr(args, "graph_filter_strength") and not hasattr(
        args, "spectral_filter_strength"
    ):
        setattr(args, "spectral_filter_strength", args.graph_filter_strength)
    if hasattr(args, "spectral_filter_strength") and not hasattr(
        args, "graph_filter_strength"
    ):
        setattr(args, "graph_filter_strength", args.spectral_filter_strength)
    return args


def public_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Return argparse values without internal parser metadata."""
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def _explicit_arg_dests(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(ARG_DEST_ALIASES.get(dest, dest))
    return explicit


def _attach_arg_origin_metadata(
    args: argparse.Namespace,
    *,
    explicit_dests: set[str],
    config_dests: set[str],
) -> argparse.Namespace:
    user_dests = set(explicit_dests) | set(config_dests)
    setattr(args, "_explicit_arg_dests", frozenset(explicit_dests))
    setattr(args, "_config_arg_dests", frozenset(config_dests))
    setattr(args, "_user_arg_dests", frozenset(user_dests))
    return args
