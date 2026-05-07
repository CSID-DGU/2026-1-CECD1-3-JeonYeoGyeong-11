"""Backward-compatible facade for Cora Flower clients."""

from spectral_fl.clients.cora import (
    FlowerClient,
    evaluate,
    get_parameters,
    seed_everything,
    set_parameters,
    train_one_client,
)

__all__ = [
    "FlowerClient",
    "evaluate",
    "get_parameters",
    "seed_everything",
    "set_parameters",
    "train_one_client",
]
