"""Runtime-backed resolution used by the parser-only CLI."""

from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from graphfl_lab.config_io import public_args_dict
from graphfl_lab.designs import design_names
from graphfl_lab.extensions.runtime import prepare_graph_extensions
from graphfl_lab.graph import graph_mode_names, graph_source_names
from graphfl_lab.strategies.graphfl.targets import aggregation_target_names


RUN_MODES = {
    "suite": "graphfl_lab.cli.vision_suite",
    "ablation": "graphfl_lab.cli.graph_ablation",
    "stress": "graphfl_lab.cli.vision_stress_grid",
    "client-count": "graphfl_lab.cli.vision_client_count_sweep",
}


def parse_with_module(module, argv: list[str]):
    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        return module.parse_args()
    finally:
        sys.argv = old_argv


def resolve_run(mode: str, argv: list[str]) -> tuple[Any, str, list[str]]:
    if mode == "single":
        from graphfl_lab.cli import experiment_dispatcher

        track, remaining = experiment_dispatcher._parse_track(argv)
        selected = track or "cora"
        return experiment_dispatcher.TRACK_MODULES[selected], selected, remaining
    if mode not in RUN_MODES:
        raise ValueError(f"unknown run mode: {mode}")
    module = importlib.import_module(RUN_MODES[mode])
    track = "cora" if mode == "ablation" else "vision"
    return module, track, list(argv)


def _flag_value(argv: list[str], flag: str, default: Any) -> Any:
    try:
        index = argv.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def _variant_component(
    *,
    variant: str,
    method: str,
    cli_args: list[str],
    args: Any,
) -> dict[str, Any]:
    if method != "ours":
        return {
            "variant": variant,
            "method": method,
            "source": "none",
            "builder": "none",
            "aggregation": "none",
            "parameters": {},
        }
    default_k = int(
        getattr(args, "knn_k", 0)
        or next(iter(getattr(args, "knn_ks", [0])), 0)
    )
    return {
        "variant": variant,
        "method": method,
        "source": str(
            _flag_value(
                cli_args,
                "--graph-source",
                getattr(args, "graph_source", "update"),
            )
        ),
        "builder": str(
            _flag_value(
                cli_args,
                "--graph-mode",
                getattr(args, "graph_mode", "dense") or "dense",
            )
        ),
        "aggregation": str(
            _flag_value(
                cli_args,
                "--aggregation-target",
                getattr(args, "aggregation_target", "update"),
            )
        ),
        "parameters": {
            "knn_k": int(_flag_value(cli_args, "--knn-k", default_k)),
            "aggregation": dict(
                getattr(args, "aggregation_params", {}) or {}
            ),
        },
    }


def _variant_components(mode: str, args: Any) -> list[dict[str, Any]]:
    if mode == "ablation":
        from graphfl_lab.experiments.cora.graph_ablation import variant_command

        rows = []
        seed = int(next(iter(getattr(args, "seeds", [0])), 0))
        for variant in getattr(args, "variants", []):
            command, method, _ = variant_command(
                str(variant),
                args,
                seed,
                Path("."),
                "dry_run",
            )
            rows.append(
                _variant_component(
                    variant=str(variant),
                    method=str(method),
                    cli_args=list(command),
                    args=args,
                )
            )
        return rows

    variants = list(getattr(args, "variants", []))
    variant_args = args
    if mode == "stress":
        from graphfl_lab.experiments.vision.stress_grid import (
            expand_variant_templates,
        )

        variants = expand_variant_templates(
            getattr(args, "variant_templates", []),
            getattr(args, "knn_ks", []),
        )
        variant_args = Namespace(**vars(args))
        variant_args.knn_k = int(
            next(iter(getattr(args, "knn_ks", [0])), 0)
        )
    if mode not in {"suite", "stress", "client-count"}:
        return []

    from graphfl_lab.experiments.suites.vision.variants import parse_variant

    rows = []
    for variant in variants:
        method, label, extras = parse_variant(str(variant), variant_args)
        rows.append(
            _variant_component(
                variant=str(label),
                method=str(method),
                cli_args=list(extras),
                args=variant_args,
            )
        )
    return rows


def dry_run_payload(mode: str, argv: list[str]) -> dict[str, Any]:
    module, track, parser_argv = resolve_run(mode, argv)
    args = parse_with_module(module, parser_argv)
    info = prepare_graph_extensions(args)
    variants = _variant_components(mode, args)
    builder = str(getattr(args, "graph_mode", ""))
    if not builder and variants:
        builder = "variant-defined"
    return {
        "dry_run": True,
        "mode": mode,
        "track": track,
        "args": public_args_dict(args),
        "extension_info": info,
        "resolved_components": {
            "source": str(getattr(args, "graph_source", "")),
            "builder": builder,
            "aggregation": str(getattr(args, "aggregation_target", "")),
            "parameters": {
                "knn_k": int(getattr(args, "knn_k", 0)),
                "aggregation": dict(
                    getattr(args, "aggregation_params", {}) or {}
                ),
            },
            "variants": variants,
        },
        "registered": {
            "sources": graph_source_names(),
            "builders": graph_mode_names(),
            "aggregations": aggregation_target_names(),
            "designs": design_names(include_aliases=True),
        },
    }


__all__ = [
    "RUN_MODES",
    "dry_run_payload",
    "parse_with_module",
    "resolve_run",
]
