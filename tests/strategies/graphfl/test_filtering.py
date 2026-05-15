import unittest

import numpy as np

from spectral_fl.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
    laplacian,
    normalized_conflicts,
    spectral_filter,
)


class SpectralFilteringModuleTest(unittest.TestCase):
    def test_zero_strength_is_identity_filter(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        graph = np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64)
        l_mat = laplacian(graph)

        filtered, diagnostics = apply_spectral_filter_with_diagnostics(
            z_mat=z_mat,
            l_mat=l_mat,
            filter_strength=0.0,
        )

        self.assertTrue(np.allclose(filtered, z_mat))
        self.assertAlmostEqual(
            diagnostics["spectral_filter_residual_energy_ratio"],
            0.0,
            places=6,
        )

    def test_filter_output_and_conflicts_keep_client_axis(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        graph = np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64)
        smoothed = spectral_filter(z_mat, laplacian(graph))
        conflicts = normalized_conflicts(z_mat, smoothed)

        self.assertEqual(smoothed.shape, z_mat.shape)
        self.assertEqual(conflicts.shape, (3,))
        self.assertTrue(np.all(conflicts >= 0.0))


if __name__ == "__main__":
    unittest.main()
