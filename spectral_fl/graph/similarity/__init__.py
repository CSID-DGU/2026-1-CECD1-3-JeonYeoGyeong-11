"""Pairwise client-signal similarities and distances."""

from spectral_fl.graph.similarity.cosine import (
    cosine_nonnegative,
    dense_absolute_cosine,
    dense_negative_cosine,
    dense_positive_cosine,
    dense_signed_cosine,
)
from spectral_fl.graph.similarity.magnitude import (
    pairwise_sq_dists,
    positive_upper_values,
    resolve_distance_sigma,
)

__all__ = [
    "cosine_nonnegative",
    "dense_absolute_cosine",
    "dense_negative_cosine",
    "dense_positive_cosine",
    "dense_signed_cosine",
    "pairwise_sq_dists",
    "positive_upper_values",
    "resolve_distance_sigma",
]
