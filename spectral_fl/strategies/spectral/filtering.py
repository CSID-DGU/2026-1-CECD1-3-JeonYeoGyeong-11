"""Graph Laplacian filtering for spectral client signals."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def laplacian(w: np.ndarray) -> np.ndarray:
    d = np.diag(np.sum(w, axis=1))
    return d - w


def _spectral_filter_basis(
    l_mat: np.ndarray,
    filter_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Laplacian eigenbasis and low-pass gains.

    ``filter_strength`` is an exponent on the current linear low-pass gain:
    0 disables filtering, 1 preserves the legacy filter, and values above 1
    make the filter more aggressive.
    """
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    lambda_max = float(max(np.max(eigvals), 1e-12))
    base_gains = np.clip(1.0 - (eigvals / lambda_max), 0.0, 1.0)
    strength = max(float(filter_strength), 0.0)
    if strength <= 0.0:
        gains = np.ones_like(base_gains, dtype=np.float64)
    else:
        gains = np.power(base_gains, strength)
    return eigvals, eigvecs, gains.astype(np.float64)


def apply_spectral_filter_with_diagnostics(
    z_mat: np.ndarray,
    l_mat: np.ndarray,
    filter_strength: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Low-pass filter a client signal and report how much was attenuated."""
    eigvals, eigvecs, gains = _spectral_filter_basis(
        l_mat=l_mat,
        filter_strength=filter_strength,
    )
    z_hat = eigvecs.T @ z_mat
    filtered_hat = gains[:, None] * z_hat
    filtered = eigvecs @ filtered_hat

    total_energy = float(np.sum(z_hat * z_hat) + eps)
    output_energy = float(np.sum(filtered_hat * filtered_hat))
    residual_hat = z_hat - filtered_hat
    residual_energy = float(np.sum(residual_hat * residual_hat))
    suppressed_energy = float(
        np.sum((1.0 - gains * gains) * np.sum(z_hat * z_hat, axis=1))
    )

    diagnostics = {
        "spectral_filter_strength": float(filter_strength),
        "spectral_filter_gain_list": [float(x) for x in gains.tolist()],
        "spectral_filter_gain_min": float(np.min(gains)) if gains.size else 0.0,
        "spectral_filter_gain_max": float(np.max(gains)) if gains.size else 0.0,
        "spectral_filter_gain_mean": float(np.mean(gains)) if gains.size else 0.0,
        "spectral_filter_gain_std": float(np.std(gains)) if gains.size else 0.0,
        "spectral_filter_output_energy_ratio": float(output_energy / total_energy),
        "spectral_filter_residual_energy_ratio": float(residual_energy / total_energy),
        "spectral_filter_suppressed_energy_ratio": float(
            suppressed_energy / total_energy
        ),
        "spectral_filter_lambda_max": float(eigvals[-1]) if eigvals.size else 0.0,
    }
    return filtered, diagnostics


def spectral_filter(
    z_mat: np.ndarray,
    l_mat: np.ndarray,
    filter_strength: float = 1.0,
) -> np.ndarray:
    """Low-pass filter Z on the client graph: Z_tilde = U g(Lambda) U^T Z."""
    filtered, _ = apply_spectral_filter_with_diagnostics(
        z_mat=z_mat,
        l_mat=l_mat,
        filter_strength=filter_strength,
    )
    return filtered


def normalized_conflicts(
    z_mat: np.ndarray, z_tilde: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Per-client deviation from graph-smoothed consensus."""
    num = np.linalg.norm(z_mat - z_tilde, axis=1)
    den = np.linalg.norm(z_mat, axis=1) + eps
    return num / den


__all__ = [
    "apply_spectral_filter_with_diagnostics",
    "laplacian",
    "normalized_conflicts",
    "spectral_filter",
]
