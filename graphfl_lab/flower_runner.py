"""Local runner for the Flower App experiment entrypoints.

The public, long-term execution target is ``flwr run``.  This helper runs the
same ClientApp/ServerApp components directly for local scripts so the existing
``python run_*.py`` workflow keeps producing result JSON files without using the
deprecated ``start_simulation`` entrypoint.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

from flwr.common import Context, EventType, RecordDict
from flwr.common.constant import RUN_ID_NUM_BYTES
from flwr.common.typing import Run
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.simulation.run_simulation import _run_simulation
from flwr.supercore.constant import NOOP_FEDERATION

from graphfl_lab.config_io import public_args_dict
from graphfl_lab.flower_app import DEFAULT_RUN_CONFIG, client_app, server_app
from graphfl_lab.extensions.runtime import prepare_graph_extensions


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return json.dumps(str(value))


def _absolute_path(value: str) -> str:
    if not value:
        return value
    return str(Path(value).expanduser().resolve())


def _resolved_args_for_run_config(args: Namespace) -> Namespace:
    resolved = Namespace(**vars(args))
    prepare_graph_extensions(resolved)
    return resolved


def args_to_run_config(args: Namespace, track: str) -> Dict[str, Any]:
    resolved_args = _resolved_args_for_run_config(args)
    original_public_keys = set(public_args_dict(args))
    cfg = dict(DEFAULT_RUN_CONFIG)
    cfg["track"] = track
    for key, value in public_args_dict(resolved_args).items():
        if key in {"engine", "config"}:
            continue
        run_key = (
            "graph-filter-strength"
            if key == "spectral_filter_strength"
            else key.replace("_", "-")
        )
        if key not in original_public_keys and run_key not in DEFAULT_RUN_CONFIG:
            continue
        cfg[run_key] = (
            json.dumps(value, sort_keys=True)
            if key == "aggregation_params" and isinstance(value, dict)
            else value
        )

    cfg["data-root"] = _absolute_path(str(cfg["data-root"]))
    cfg["out-dir"] = _absolute_path(str(cfg["out-dir"]))
    if cfg.get("projection-dim") is None:
        cfg["projection-dim"] = 0
    return cfg


def run_config_as_cli_line(run_config: Dict[str, Any]) -> str:
    return " ".join(f"{key}={_toml_value(value)}" for key, value in run_config.items())


def run_app_locally(args: Namespace, track: str) -> None:
    """Run the Flower App components without deprecated ``start_simulation``."""

    run_config = args_to_run_config(args, track=track)
    run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
    run = Run.create_empty(run_id=run_id)
    run.federation = NOOP_FEDERATION
    run.override_config = run_config
    context = Context(
        run_id=run_id,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )
    # Keep local simulation concurrency conservative on Windows to avoid
    # runaway worker spawning and unstable Ray startup under heavy settings.
    backend_config = {
        "client_resources": {"num_cpus": 1.0, "num_gpus": 0.0},
        "init_args": {"include_dashboard": False, "log_to_driver": True},
    }
    _run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=int(run_config["num-clients"]),
        backend_name="ray",
        backend_config=backend_config,
        app_dir=str(Path(__file__).resolve().parents[1]),
        is_app=True,
        run=run,
        server_app_context=context,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
    )


def run_app_via_flwr_cli(args: Namespace, track: str) -> None:
    """Submit the same app through Flower's CLI implementation.

    This path is useful for checking compatibility with the official ``flwr run``
    flow.  The local script default uses ``run_app_locally`` because some Windows
    environments install console-script shims that cannot launch correctly.
    """

    from flwr.cli.run.run import run as flwr_run

    run_config = args_to_run_config(args, track=track)
    run_line = run_config_as_cli_line(run_config)
    federation_line = (
        f"num-supernodes={int(run_config['num-clients'])} "
        "backend=\"ray\" "
        "client-resources-num-cpus=1 "
        "client-resources-num-gpus=0.0 "
        "init-args-logging-level=\"WARNING\" "
        "init-args-log-to-driver=true"
    )
    flwr_run(
        app=str(Path(__file__).resolve().parents[1]),
        superlink=None,
        federation=None,
        run_config_overrides=[run_line],
        federation_config_overrides=[federation_line],
        stream=True,
        output_format="default",
    )


def print_flwr_run_hint(args: Namespace, track: str) -> None:
    run_line = run_config_as_cli_line(args_to_run_config(args, track=track))
    federation_line = (
        f"num-supernodes={int(getattr(args, 'num_clients'))} "
        "backend=\"ray\" client-resources-num-cpus=1 client-resources-num-gpus=0.0"
    )
    print("Equivalent Flower CLI command:")
    print(
        "python -c \"from flwr.cli.app import app; app()\" run . "
        f"--run-config {json.dumps(run_line)} "
        f"--federation-config {json.dumps(federation_line)} "
        "--stream"
    )


def main_dispatch(args: Namespace, track: str) -> None:
    engine = getattr(args, "engine", "app")
    if engine == "app":
        run_app_locally(args, track=track)
        return
    if engine == "flwr-run":
        run_app_via_flwr_cli(args, track=track)
        return
    if engine == "print-flwr-run":
        print_flwr_run_hint(args, track=track)
        return
    raise ValueError(f"Unknown engine: {engine}")


__all__ = [
    "args_to_run_config",
    "main_dispatch",
    "print_flwr_run_hint",
    "run_app_locally",
    "run_app_via_flwr_cli",
]
