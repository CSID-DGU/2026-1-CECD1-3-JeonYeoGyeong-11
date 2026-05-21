"""Cosine similarities for client graph construction."""

from __future__ import annotations

import numpy as np


def cosine_nonnegative(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    cos = float(np.dot(a, b) / denom)
    return max(0.0, cos)


def dense_positive_cosine(z_mat: np.ndarray) -> np.ndarray:
    """Pairwise max(0, cosine) similarity, zero diagonal, symmetric."""
    norms = np.linalg.norm(z_mat, axis=1) + 1e-12
    z_n = z_mat / norms[:, None]
    sim = z_n @ z_n.T
    sim = np.clip(sim, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    sim = 0.5 * (sim + sim.T)
    return sim.astype(np.float64)


def dense_signed_cosine(z_mat: np.ndarray) -> np.ndarray:
    """Pairwise signed cosine similarity, zero diagonal, symmetric."""
    norms = np.linalg.norm(z_mat, axis=1) + 1e-12
    z_n = z_mat / norms[:, None]
    sim = z_n @ z_n.T
    np.fill_diagonal(sim, 0.0)
    sim = 0.5 * (sim + sim.T)
    return sim.astype(np.float64)


def dense_absolute_cosine(z_mat: np.ndarray) -> np.ndarray:
    """Pairwise absolute cosine magnitude, zero diagonal, symmetric."""
    sim = np.abs(dense_signed_cosine(z_mat))
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float64)


def dense_negative_cosine(z_mat: np.ndarray) -> np.ndarray:
    """Pairwise anti-alignment strength max(0, -cosine)."""
    sim = np.clip(-dense_signed_cosine(z_mat), 0.0, None)
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float64)


__all__ = [
    "cosine_nonnegative",
    "dense_absolute_cosine",
    "dense_negative_cosine",
    "dense_positive_cosine",
    "dense_signed_cosine",
]
