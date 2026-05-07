"""Backward-compatible facade for vision Flower clients."""

from spectral_fl.clients.vision import (
    VisionFlowerClient,
    get_parameters,
    seed_everything,
    set_parameters,
)

__all__ = [
    "VisionFlowerClient",
    "get_parameters",
    "seed_everything",
    "set_parameters",
]
