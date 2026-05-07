import unittest

import numpy as np

from spectral_fl.spectral_diagnostics import (
    apply_spectral_filter_with_diagnostics,
    heterogeneity,
    laplacian,
    normalized_conflicts,
    spectral_energy_diagnostics,
    spectral_filter,
)
from spectral_fl.update_graph import build_client_graph


class SpectralDiagnosticsTest(unittest.TestCase):
    def test_energy_bands_partition_total_energy(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        graph = build_client_graph(z_mat, mode="uniform")
        diagnostics = spectral_energy_diagnostics(z_mat, laplacian(graph))

        total = (
            diagnostics["low_frequency_energy_ratio"]
            + diagnostics["mid_frequency_energy_ratio"]
            + diagnostics["high_frequency_energy_ratio"]
        )

        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertEqual(len(diagnostics["laplacian_eigenvalues"]), 3)
        self.assertGreaterEqual(diagnostics["spectral_entropy"], 0.0)

    def test_filter_and_conflict_shapes_match_clients(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        graph = build_client_graph(z_mat, mode="dense")
        l_mat = laplacian(graph)
        smoothed = spectral_filter(z_mat, l_mat)
        conflicts = normalized_conflicts(z_mat, smoothed)

        self.assertEqual(smoothed.shape, z_mat.shape)
        self.assertEqual(conflicts.shape, (z_mat.shape[0],))
        self.assertGreaterEqual(heterogeneity(z_mat, l_mat), 0.0)

    def test_filter_strength_controls_attenuation(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=np.float64,
        )
        graph = build_client_graph(z_mat, mode="uniform")
        l_mat = laplacian(graph)

        no_filter, no_diag = apply_spectral_filter_with_diagnostics(
            z_mat, l_mat, filter_strength=0.0
        )
        weak, weak_diag = apply_spectral_filter_with_diagnostics(
            z_mat, l_mat, filter_strength=0.5
        )
        strong, strong_diag = apply_spectral_filter_with_diagnostics(
            z_mat, l_mat, filter_strength=2.0
        )

        self.assertTrue(np.allclose(no_filter, z_mat))
        self.assertAlmostEqual(no_diag["spectral_filter_residual_energy_ratio"], 0.0)
        self.assertGreaterEqual(
            weak_diag["spectral_filter_output_energy_ratio"],
            strong_diag["spectral_filter_output_energy_ratio"],
        )
        self.assertGreaterEqual(
            strong_diag["spectral_filter_residual_energy_ratio"],
            weak_diag["spectral_filter_residual_energy_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
