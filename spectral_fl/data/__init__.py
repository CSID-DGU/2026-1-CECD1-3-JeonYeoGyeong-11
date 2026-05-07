"""Dataset loading and partition helpers."""

from spectral_fl.data.cora import ClientGraph, load_cora_clients
from spectral_fl.data.vision import (
    VisionClientShard,
    _dirichlet_partition,
    _ensure_min_samples_per_client,
    load_vision_clients,
    vision_input_shape,
    vision_num_classes,
)

__all__ = [
    "ClientGraph",
    "VisionClientShard",
    "_dirichlet_partition",
    "_ensure_min_samples_per_client",
    "load_cora_clients",
    "load_vision_clients",
    "vision_input_shape",
    "vision_num_classes",
]
