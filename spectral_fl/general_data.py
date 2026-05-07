"""Backward-compatible facade for general vision data helpers."""

from spectral_fl.data.vision import (
    VisionClientShard,
    _dirichlet_partition,
    _ensure_min_samples_per_client,
    load_vision_clients,
    vision_input_shape,
    vision_num_classes,
)

__all__ = [
    "VisionClientShard",
    "_dirichlet_partition",
    "_ensure_min_samples_per_client",
    "load_vision_clients",
    "vision_input_shape",
    "vision_num_classes",
]
