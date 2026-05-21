import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.conflict_metrics import (
    compute_conflict_metric_bundle,
)


class GraphFLConflictMetricsTest(unittest.TestCase):
    def test_compute_conflict_metric_bundle_uses_h_spec_tau_source(self):
        z_mat = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            dtype=np.float64,
        )
        l_mat = np.array(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )

        metrics = compute_conflict_metric_bundle(
            z_mat=z_mat,
            l_mat=l_mat,
            filter_strength=1.0,
            tau_source_name="h_spec",
            h_spec=0.4,
            h_spec_normalized=0.2,
            h_spec_ema=0.3,
            h_spec_ema_candidate=0.31,
            tau_signal_ema=0.9,
            tau_max=2.0,
            tau_gain=1.0,
            adaptive_tau=True,
            fixed_tau=1.0,
            e_std_threshold=0.0,
        )

        self.assertEqual(metrics.tau_source.source_used, "h_spec")
        self.assertAlmostEqual(metrics.tau_source.ema_value, 0.3)
        self.assertEqual(metrics.z_tilde.shape, z_mat.shape)
        self.assertEqual(metrics.e.shape, (3,))
        self.assertFalse(metrics.estd_disabled)
        self.assertIn("spectral_filter_gain_mean", metrics.filter_diag)

    def test_compute_conflict_metric_bundle_updates_non_hspec_tau_candidate(self):
        z_mat = np.array([[1.0], [0.0], [2.0]], dtype=np.float64)
        l_mat = np.array(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )

        metrics = compute_conflict_metric_bundle(
            z_mat=z_mat,
            l_mat=l_mat,
            filter_strength=0.0,
            tau_source_name="e_std",
            h_spec=0.4,
            h_spec_normalized=0.2,
            h_spec_ema=0.3,
            h_spec_ema_candidate=0.31,
            tau_signal_ema=0.5,
            tau_max=2.0,
            tau_gain=1.0,
            adaptive_tau=False,
            fixed_tau=0.75,
            e_std_threshold=10.0,
        )

        self.assertEqual(metrics.tau_source.source_used, "e_std")
        self.assertAlmostEqual(metrics.tau, 0.75)
        self.assertTrue(metrics.estd_disabled)
        self.assertTrue(np.allclose(metrics.conflict_weight, np.ones(3)))
        self.assertNotEqual(
            metrics.tau_source.ema_candidate,
            metrics.tau_source.ema_value,
        )


if __name__ == "__main__":
    unittest.main()
