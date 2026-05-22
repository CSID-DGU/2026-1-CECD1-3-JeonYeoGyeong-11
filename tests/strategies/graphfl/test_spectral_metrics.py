import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.spectral_metrics import (
    compute_round_spectral_metrics,
)


class GraphFLSpectralMetricsTest(unittest.TestCase):
    def test_compute_round_spectral_metrics_uses_current_graph_without_previous(self):
        z_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        current_graph = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

        metrics = compute_round_spectral_metrics(
            z_mat=z_mat,
            current_graph=current_graph,
            used_graph=current_graph,
            previous_laplacian=None,
            previous_h_spec_ema=2.0,
        )

        self.assertEqual(metrics.metric_graph_source, "current_round_graph")
        np.testing.assert_allclose(metrics.l_curr, [[1.0, -1.0], [-1.0, 1.0]])
        self.assertGreater(metrics.h_spec, 0.0)
        self.assertAlmostEqual(
            metrics.h_spec_ema_candidate,
            0.9 * 2.0 + 0.1 * metrics.h_spec,
        )
        self.assertIn("spectral_entropy", metrics.spectral_diag)
        self.assertIn("lambda_max", metrics.spectral_diag)

    def test_compute_round_spectral_metrics_uses_previous_laplacian_for_metric(self):
        z_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        current_graph = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        previous_laplacian = np.zeros((2, 2), dtype=np.float64)

        metrics = compute_round_spectral_metrics(
            z_mat=z_mat,
            current_graph=current_graph,
            used_graph=current_graph,
            previous_laplacian=previous_laplacian,
            previous_h_spec_ema=0.0,
        )

        self.assertEqual(metrics.metric_graph_source, "previous_round_graph")
        self.assertIs(metrics.l_for_metric, previous_laplacian)
        self.assertEqual(metrics.h_spec, 0.0)


if __name__ == "__main__":
    unittest.main()
