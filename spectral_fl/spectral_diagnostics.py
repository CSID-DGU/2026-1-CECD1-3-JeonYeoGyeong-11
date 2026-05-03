"""Spectral diagnostics for client update graph signals."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def laplacian(w: np.ndarray) -> np.ndarray:
    d = np.diag(np.sum(w, axis=1))
    return d - w


def spectral_filter(z_mat: np.ndarray, l_mat: np.ndarray) -> np.ndarray:
    """Low-pass filter Z on the client graph: Z_tilde = U g(Lambda) U^T Z."""
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    lambda_max = float(max(np.max(eigvals), 1e-12))
    gains = 1.0 - (eigvals / lambda_max)
    gains = np.clip(gains, 0.0, 1.0)
    return eigvecs @ np.diag(gains) @ eigvecs.T @ z_mat


def heterogeneity(z_mat: np.ndarray, l_mat: np.ndarray, eps: float = 1e-12) -> float:
    numerator = float(np.trace(z_mat.T @ l_mat @ z_mat))
    denominator = float(np.linalg.norm(z_mat, ord="fro") ** 2 + eps)
    return numerator / denominator


def spectral_energy_diagnostics(
    z_mat: np.ndarray,
    l_mat: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Return graph Fourier decomposition diagnostics for update signal Z.

    Global band energy ratios are exact partitions in the Laplacian eigenbasis.
    Per-client band values are node-domain component norm ratios after
    reconstructing low/mid/high components, so they should be interpreted as
    client-local component strength rather than a strict per-client partition.
    """
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    z_hat = eigvecs.T @ z_mat
    energy = np.sum(z_hat * z_hat, axis=1).astype(np.float64)
    total_energy = float(np.sum(energy) + eps)
    ratios = energy / total_energy

    n_modes = int(eigvals.size)
    band_k = max(1, n_modes // 3) if n_modes > 0 else 0
    low_idx = list(range(0, band_k))
    high_start = max(n_modes - band_k, 0)
    high_idx = list(range(high_start, n_modes))
    mid_idx = list(range(band_k, high_start))

    low_ratio = float(np.sum(ratios[low_idx])) if low_idx else 0.0
    mid_ratio = float(np.sum(ratios[mid_idx])) if mid_idx else 0.0
    high_ratio = float(np.sum(ratios[high_idx])) if high_idx else 0.0
    dc_ratio = float(ratios[0]) if n_modes else 0.0
    highest_ratio = float(ratios[-1]) if n_modes else 0.0
    dominant_mode_idx = int(np.argmax(ratios)) if n_modes else -1

    def reconstruct(mode_indices: list[int]) -> np.ndarray:
        if not mode_indices:
            return np.zeros_like(z_mat, dtype=np.float64)
        u_band = eigvecs[:, mode_indices]
        return u_band @ z_hat[mode_indices, :]

    z_low = reconstruct(low_idx)
    z_mid = reconstruct(mid_idx)
    z_high = reconstruct(high_idx)
    client_norm = np.linalg.norm(z_mat, axis=1) + eps
    low_client_ratio = np.linalg.norm(z_low, axis=1) / client_norm
    mid_client_ratio = np.linalg.norm(z_mid, axis=1) / client_norm
    high_client_ratio = np.linalg.norm(z_high, axis=1) / client_norm

    if n_modes > 1:
        gaps = np.diff(eigvals)
        eigengap_max = float(np.max(gaps))
        eigengap_argmax = int(np.argmax(gaps))
        lambda_2 = float(eigvals[1])
    else:
        eigengap_max = 0.0
        eigengap_argmax = -1
        lambda_2 = 0.0

    entropy = 0.0
    if n_modes > 1:
        entropy = float(-np.sum(ratios * np.log(ratios + eps)) / np.log(n_modes))

    return {
        "laplacian_eigenvalues": [float(x) for x in eigvals.tolist()],
        "spectral_mode_energy_list": [float(x) for x in energy.tolist()],
        "spectral_energy_ratio_list": [float(x) for x in ratios.tolist()],
        "spectral_entropy": float(entropy),
        "frequency_band_indices": {
            "low": [int(x) for x in low_idx],
            "mid": [int(x) for x in mid_idx],
            "high": [int(x) for x in high_idx],
        },
        "low_frequency_energy_ratio": low_ratio,
        "mid_frequency_energy_ratio": mid_ratio,
        "high_frequency_energy_ratio": high_ratio,
        "low_frequency_band_k": int(band_k),
        "mid_frequency_band_k": int(len(mid_idx)),
        "high_frequency_band_k": int(band_k),
        "low_frequency_component_norm_ratio_list": [
            float(x) for x in low_client_ratio.tolist()
        ],
        "mid_frequency_component_norm_ratio_list": [
            float(x) for x in mid_client_ratio.tolist()
        ],
        "high_frequency_component_norm_ratio_list": [
            float(x) for x in high_client_ratio.tolist()
        ],
        "dc_energy_ratio": dc_ratio,
        "highest_frequency_energy_ratio": highest_ratio,
        "dominant_frequency_mode_index": int(dominant_mode_idx),
        "dominant_frequency_mode_lambda": (
            float(eigvals[dominant_mode_idx]) if dominant_mode_idx >= 0 else 0.0
        ),
        "dominant_frequency_energy_ratio": (
            float(ratios[dominant_mode_idx]) if dominant_mode_idx >= 0 else 0.0
        ),
        "high_to_low_energy_ratio": float(high_ratio / (low_ratio + eps)),
        "lambda_max": float(eigvals[-1]) if n_modes else 0.0,
        "lambda_2": lambda_2,
        "eigengap_max": eigengap_max,
        "eigengap_argmax": eigengap_argmax,
    }


def normalized_conflicts(
    z_mat: np.ndarray, z_tilde: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Per-client deviation from graph-smoothed consensus."""
    num = np.linalg.norm(z_mat - z_tilde, axis=1)
    den = np.linalg.norm(z_mat, axis=1) + eps
    return num / den
