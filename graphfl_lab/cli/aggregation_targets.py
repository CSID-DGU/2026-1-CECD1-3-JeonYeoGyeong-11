"""Shared aggregation-target CLI choices for experiment parsers."""

from __future__ import annotations

AGGREGATION_TARGET_CHOICES = (
    "update",
    "weight",
    "graph_filtered_update",
    "graph_filtered_ema_update",
    "graph_filtered_weight",
    # Compatibility aliases retained until Gate 6.
    "spectral_filtered_update",
    "spectral_filtered_ema_update",
    "spectral_filtered_weight",
)

AGGREGATION_TARGET_HELP = (
    "AggregationOperator knob: object averaged with alpha_i to form the next global model. "
    "Prefer graph_filtered_update, graph_filtered_ema_update, or graph_filtered_weight. "
    "spectral_filtered_* spellings remain accepted compatibility aliases."
)

AGGREGATION_TARGET_SUITE_HELP = (
    "Default aggregation target forwarded to the underlying single-run command. "
    "Prefer graph_filtered_* spellings; spectral_filtered_* remains a compatibility alias."
)

__all__ = [
    "AGGREGATION_TARGET_CHOICES",
    "AGGREGATION_TARGET_HELP",
    "AGGREGATION_TARGET_SUITE_HELP",
]
