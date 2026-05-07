"""Flower App entrypoints for spectral client update graph experiments.

The app keeps the existing result JSON format while moving execution toward
Flower's ClientApp/ServerApp structure.  The command-line scripts pass a flat
``run_config`` into this app; the ServerApp writes the same JSON files consumed
by the analysis scripts.
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.compat import start_grid
from flwr.server.grid import Grid

from spectral_fl.experiments.cora.single_run import (
    attach_round_trace,
    build_meta,
    build_strategy,
    compute_client_class_distribution,
    history_to_dict,
    make_initial_parameters as make_cora_initial_parameters,
    print_final_summary,
)
from spectral_fl.experiments.general.single_run import (
    build_general_meta,
    make_initial_parameters as make_general_initial_parameters,
)
from spectral_fl.app.config import DEFAULT_RUN_CONFIG, args_from_context
from spectral_fl.app.data_cache import client_index, load_cora, load_general
from spectral_fl.clients.cora import FlowerClient
from spectral_fl.clients.vision import VisionFlowerClient
from spectral_fl.data.vision import (
    vision_input_shape,
    vision_num_classes,
)
from spectral_fl.models.vision import build_model


_args_from_context = args_from_context
_client_index = client_index
_load_general = load_general
_load_cora = load_cora


def _run_with_strategy(
    grid: Grid,
    args: Namespace,
    method: str,
    initial_parameters,
) -> Tuple[dict, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    strategy = build_strategy(
        args=args, method=method, initial_parameters=initial_parameters
    )
    started_at = datetime.now()
    start = time.perf_counter()
    history = start_grid(
        grid=grid,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    wall_time_sec = time.perf_counter() - start
    completed_at = datetime.now()
    hist_dict = history_to_dict(history)
    hist_dict["round_trace"] = attach_round_trace(
        method=method,
        history_dict=hist_dict,
        strategy=strategy,
        seed=args.seed,
    )
    hist_dict["timing"] = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_time_sec": float(wall_time_sec),
        "rounds": int(args.rounds),
        "seconds_per_round": float(wall_time_sec / max(int(args.rounds), 1)),
    }
    return hist_dict, strategy


def _run_general_server(grid: Grid, args: Namespace) -> None:
    server_started_at = datetime.now()
    server_start = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    shards, _, _ = _load_general(args)
    num_classes = vision_num_classes(args.dataset)
    in_channels, _, _ = vision_input_shape(args.dataset)

    template = build_model(args.model, num_classes=num_classes, in_channels=in_channels)
    initial_parameters = make_general_initial_parameters(template, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_path = out_dir / f"result_general_{args.method}_seed{args.seed}{tag}.json"
    class_dist = [s.label_hist for s in shards]
    all_results: Dict[str, Any] = {
        "meta": build_general_meta(args, class_dist, out_path),
        "results": {},
    }

    methods = ["fedavg", "ours"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"\n=== Running general FL method: {method} ({args.dataset}) ===")
        hist_dict, _ = _run_with_strategy(
            grid=grid,
            args=args,
            method=method,
            initial_parameters=initial_parameters,
        )
        all_results["results"][method] = hist_dict
        print_final_summary(method, hist_dict)

    total_wall_time_sec = time.perf_counter() - server_start
    all_results["meta"]["timing"] = {
        "started_at": server_started_at.isoformat(),
        "completed_at": datetime.now().isoformat(),
        "total_wall_time_sec": float(total_wall_time_sec),
        "num_methods": len(methods),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


def _run_cora_server(grid: Grid, args: Namespace) -> None:
    server_started_at = datetime.now()
    server_start = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    client_graphs = _load_cora(args)
    in_dim = int(client_graphs[0].data.x.shape[1])
    out_dim = max(int(torch.max(cg.data.y).item() + 1) for cg in client_graphs)
    initial_parameters = make_cora_initial_parameters(
        in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim, seed=args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_path = out_dir / f"result_{args.method}_seed{args.seed}{tag}.json"
    class_dist = compute_client_class_distribution(
        client_graphs=client_graphs, out_dim=out_dim
    )
    all_results: Dict[str, Any] = {
        "meta": build_meta(args, class_dist, out_path),
        "results": {},
    }

    methods = ["fedavg", "ours"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"\n=== Running method: {method} ===")
        hist_dict, _ = _run_with_strategy(
            grid=grid,
            args=args,
            method=method,
            initial_parameters=initial_parameters,
        )
        all_results["results"][method] = hist_dict
        print_final_summary(method, hist_dict)

    total_wall_time_sec = time.perf_counter() - server_start
    all_results["meta"]["timing"] = {
        "started_at": server_started_at.isoformat(),
        "completed_at": datetime.now().isoformat(),
        "total_wall_time_sec": float(total_wall_time_sec),
        "num_methods": len(methods),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {out_path}")


def _make_general_client(context: Context):
    args = _args_from_context(context)
    i = _client_index(context, args.num_clients)
    shards, train_ds, test_ds = _load_general(args)
    num_classes = vision_num_classes(args.dataset)
    in_channels, _, _ = vision_input_shape(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return VisionFlowerClient(
        cid=i,
        train_dataset=train_ds,
        train_indices=shards[i].train_indices,
        test_dataset=test_ds,
        model_name=args.model,
        num_classes=num_classes,
        in_channels=in_channels,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        device=device,
        seed=args.seed,
    ).to_client()


def _make_cora_client(context: Context):
    args = _args_from_context(context)
    i = _client_index(context, args.num_clients)
    client_graphs = _load_cora(args)
    in_dim = int(client_graphs[0].data.x.shape[1])
    out_dim = max(int(torch.max(cg.data.y).item() + 1) for cg in client_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FlowerClient(
        client_graph=client_graphs[i],
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=out_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        local_epochs=args.local_epochs,
        device=device,
        cid=i,
    ).to_client()


def client_fn(context: Context):
    args = _args_from_context(context)
    if args.track == "general-fl":
        return _make_general_client(context)
    if args.track == "cora-fgl":
        return _make_cora_client(context)
    raise ValueError(f"Unknown track: {args.track}")


client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp()


@server_app.main()
def server_main(grid: Grid, context: Context) -> None:
    args = _args_from_context(context)
    if args.track == "general-fl":
        _run_general_server(grid, args)
        return
    if args.track == "cora-fgl":
        _run_cora_server(grid, args)
        return
    raise ValueError(f"Unknown track: {args.track}")
