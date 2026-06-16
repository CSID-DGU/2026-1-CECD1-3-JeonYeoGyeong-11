"""Unified Graph-FL authoring and execution CLI."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from graphfl_lab.cli.authoring import (
    compose_design,
    scaffold_component,
    validate_component,
)
from graphfl_lab.cli.argparse_types import json_object
from graphfl_lab.extensions.run_resolution import (
    RUN_MODES,
    dry_run_payload,
    resolve_run,
)


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _component_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="graphfl component")
    sub = parser.add_subparsers(dest="action", required=True)
    new = sub.add_parser("new")
    new.add_argument("kind", choices=["source", "builder", "aggregation"])
    new.add_argument("name")
    new.add_argument("--workspace", default="")
    new.add_argument("--session-id", default="")
    new.add_argument("--plugin-name", default="poster_plugin")
    validate = sub.add_parser("validate")
    validate.add_argument("plugin")
    validate.add_argument("--component", default="")
    validate.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.action == "new":
        _print_json(
            scaffold_component(
                kind=args.kind,
                name=args.name,
                workspace=args.workspace or None,
                session_id=args.session_id or None,
                plugin_name=args.plugin_name,
            )
        )
        return 0
    payload = validate_component(
        plugin=args.plugin,
        component=args.component or None,
        timeout=args.timeout,
    )
    print("Contract           Result")
    print("-----------------  ------")
    for item in payload["checks"]:
        print(f"{item['name']:<17}  {'PASS' if item['pass'] else 'FAIL'}")
    return 0 if payload["ok"] else 1


def _design_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="graphfl design")
    sub = parser.add_subparsers(dest="action", required=True)
    compose = sub.add_parser("compose")
    compose.add_argument("--plugin", required=True)
    compose.add_argument("--name", required=True)
    compose.add_argument("--source", required=True)
    compose.add_argument("--builder", required=True)
    compose.add_argument("--aggregation", required=True)
    compose.add_argument("--knn-k", type=int, default=2)
    compose.add_argument("--aggregation-params", type=json_object, default={})
    args = parser.parse_args(argv)
    _print_json(
        compose_design(
            plugin=args.plugin,
            name=args.name,
            source=args.source,
            builder=args.builder,
            aggregation=args.aggregation,
            knn_k=args.knn_k,
            aggregation_params=args.aggregation_params,
        )
    )
    return 0


def _invoke_module(module, argv: list[str]):
    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        return module.main()
    finally:
        sys.argv = old_argv


def _run_command(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: graphfl run <single|suite|ablation|stress|client-count> [options]")
        return 0
    mode, remaining = argv[0], argv[1:]
    dry_run = "--dry-run" in remaining
    remaining = [item for item in remaining if item != "--dry-run"]
    module, track, parse_args = resolve_run(mode, remaining)

    if not dry_run:
        if mode == "single":
            from graphfl_lab.cli import experiment_dispatcher

            return int(
                experiment_dispatcher.main(
                    ["--track", track, *parse_args]
                )
                or 0
            )
        return int(_invoke_module(module, parse_args) or 0)

    dry_run_args = (
        ["--track", track, *parse_args]
        if mode == "single"
        else parse_args
    )
    _print_json(dry_run_payload(mode, dry_run_args))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: graphfl <component|design|run> ...\n\n"
            "Author components, compose GraphFLDesign presets, and run experiments."
        )
        return 0
    command, remaining = args[0], args[1:]
    try:
        if command == "component":
            return _component_command(remaining)
        if command == "design":
            return _design_command(remaining)
        if command == "run":
            return _run_command(remaining)
        raise ValueError(f"unknown graphfl command: {command}")
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"graphfl: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
