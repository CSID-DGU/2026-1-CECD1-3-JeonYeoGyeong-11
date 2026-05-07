"""Dataset cache helpers for the Flower app."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

from flwr.common import Context

from spectral_fl.data import load_cora_clients
from spectral_fl.data.vision import load_vision_clients


_GENERAL_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}
_CORA_CACHE: Dict[Tuple[Any, ...], Any] = {}


def client_index(context: Context, num_clients: int) -> int:
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


def load_general(args: Namespace):
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


def load_cora(args: Namespace):
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
