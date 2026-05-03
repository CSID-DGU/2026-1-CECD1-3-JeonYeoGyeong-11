"""Flower App entrypoints for spectral client update graph experiments.

The app keeps the existing result JSON format while moving execution toward
Flower's ClientApp/ServerApp structure.  The command-line scripts pass a flat
``run_config`` into this app; the ServerApp writes the same JSON files consumed
by the analysis scripts.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.compat import start_grid
from flwr.server.grid import Grid

from run_experiment import (
    attach_round_trace,
    build_meta,
    build_strategy,
    compute_client_class_distribution,
    history_to_dict,
    make_initial_parameters as make_cora_initial_parameters,
    print_final_summary,
)
from run_general_experiment import (
    build_general_meta,
    make_initial_parameters as make_general_initial_parameters,
)
from spectral_fl.client import FlowerClient
from spectral_fl.data import load_cora_clients
from spectral_fl.general_client import VisionFlowerClient
from spectral_fl.general_data import (
    load_vision_clients,
    vision_input_shape,
    vision_num_classes,
)
from spectral_fl.general_models import build_model


DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "track": "general-fl",
    "method": "ours",
    "dataset": "fashionmnist",
    "model": "mlp",
    "num-clients": 5,
    "rounds": 3,
    "local-epochs": 1,
    "batch-size": 64,
    "hidden-dim": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "weight-decay": 5e-4,
    "compression-dim": 256,
    "compression-seed": 0,
    "ema-alpha": 0.8,
    "tau-gain": 2.0,
    "tau-max": 2.0,
    "conflict-mix": 0.2,
    "warmup-rounds": 2,
    "graph-mode": "dense",
    "graph-source": "update",
    "aggregation-target": "update",
    "knn-k": 2,
    "edge-threshold": 0.0,
    "graph-seed": 0,
    "use-ema-graph": True,
    "disable-adaptive-tau": False,
    "fixed-tau": 1.0,
    "diagnostic-only": False,
    "e-std-threshold": 0.0,
    "min-client-weight": 0.0,
    "seed": 42,
    "data-root": "./data/torchvision",
    "out-dir": "./experiments_current/app_smoke",
    "run-tag": "",
    "partition": "dirichlet",
    "dirichlet-alpha": 0.1,
    "train-subset-size": 0,
    "test-subset-size": 0,
    "projection-dim": 0,
}


_GENERAL_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}
_CORA_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_s = str(value).strip().lower()
    if value_s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _cfg(context: Context) -> Dict[str, Any]:
    merged = dict(DEFAULT_RUN_CONFIG)
    merged.update(dict(context.run_config or {}))
    return merged


def _args_from_context(context: Context) -> Namespace:
    cfg = _cfg(context)
    projection_dim_raw = int(cfg.get("projection-dim", 0) or 0)
    projection_dim = projection_dim_raw if projection_dim_raw > 0 else None
    compression_dim = (
        projection_dim if projection_dim is not None else int(cfg["compression-dim"])
    )

    return Namespace(
        track=str(cfg["track"]),
        method=str(cfg["method"]),
        dataset=str(cfg["dataset"]),
        model=str(cfg["model"]),
        num_clients=int(cfg["num-clients"]),
        rounds=int(cfg["rounds"]),
        local_epochs=int(cfg["local-epochs"]),
        batch_size=int(cfg["batch-size"]),
        hidden_dim=int(cfg["hidden-dim"]),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight-decay"]),
        compression_dim=int(compression_dim),
        compression_seed=int(cfg["compression-seed"]),
        ema_alpha=float(cfg["ema-alpha"]),
        tau_gain=float(cfg["tau-gain"]),
        tau_max=float(cfg["tau-max"]),
        conflict_mix=float(cfg["conflict-mix"]),
        warmup_rounds=int(cfg["warmup-rounds"]),
        graph_mode=str(cfg["graph-mode"]),
        graph_source=str(cfg["graph-source"]),
        aggregation_target=str(cfg["aggregation-target"]),
        knn_k=int(cfg["knn-k"]),
        edge_threshold=float(cfg["edge-threshold"]),
        graph_seed=int(cfg["graph-seed"]),
        use_ema_graph=_bool(cfg["use-ema-graph"]),
        disable_adaptive_tau=_bool(cfg["disable-adaptive-tau"]),
        fixed_tau=float(cfg["fixed-tau"]),
        diagnostic_only=_bool(cfg["diagnostic-only"]),
        e_std_threshold=float(cfg["e-std-threshold"]),
        min_client_weight=float(cfg["min-client-weight"]),
        seed=int(cfg["seed"]),
        data_root=str(cfg["data-root"]),
        out_dir=str(cfg["out-dir"]),
        run_tag=str(cfg["run-tag"]),
        partition=str(cfg["partition"]),
        dirichlet_alpha=float(cfg["dirichlet-alpha"]),
        train_subset_size=int(cfg["train-subset-size"]),
        test_subset_size=int(cfg["test-subset-size"]),
        projection_dim=projection_dim,
    )


def _client_index(context: Context, num_clients: int) -> int:
    raw_idx = context.node_config.get("partition-id", context.node_id)
    idx = int(raw_idx)
    if idx < 0 or idx >= int(num_clients):
        raise IndexError(f"Client partition {idx} is outside num_clients={num_clients}")
    return idx


def _general_cache_key(args: Namespace) -> Tuple[Any, ...]:
    return (
        args.dataset,
        str(Path(args.data_root).resolve()),
        int(args.num_clients),
        int(args.seed),
        args.partition,
        float(args.dirichlet_alpha),
        int(args.train_subset_size),
        int(args.test_subset_size),
    )


def _load_general(args: Namespace):
    key = _general_cache_key(args)
    if key not in _GENERAL_CACHE:
        train_subset = (
            int(args.train_subset_size) if int(args.train_subset_size) > 0 else None
        )
        test_subset = (
            int(args.test_subset_size) if int(args.test_subset_size) > 0 else None
        )
        _GENERAL_CACHE[key] = load_vision_clients(
            dataset_name=args.dataset,
            root=args.data_root,
            num_clients=args.num_clients,
            seed=args.seed,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
            train_subset_size=train_subset,
            test_subset_size=test_subset,
        )
    return _GENERAL_CACHE[key]


def _cora_cache_key(args: Namespace) -> Tuple[Any, ...]:
    return (
        str(Path(args.data_root).resolve()),
        int(args.num_clients),
        int(args.seed),
        args.partition,
        float(args.dirichlet_alpha),
    )


def _load_cora(args: Namespace):
    key = _cora_cache_key(args)
    if key not in _CORA_CACHE:
        _CORA_CACHE[key] = load_cora_clients(
            root=args.data_root,
            num_clients=args.num_clients,
            seed=args.seed,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
        )
    return _CORA_CACHE[key]


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
    history = start_grid(
        grid=grid,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    hist_dict = history_to_dict(history)
    hist_dict["round_trace"] = attach_round_trace(
        method=method,
        history_dict=hist_dict,
        strategy=strategy,
        seed=args.seed,
    )
    return hist_dict, strategy


def _run_general_server(grid: Grid, args: Namespace) -> None:
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

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


def _run_cora_server(grid: Grid, args: Namespace) -> None:
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
