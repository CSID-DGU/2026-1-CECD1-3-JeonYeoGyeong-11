import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.diagnostic_targets import (
    flatten_diagnostic_post_updates,
)


class GraphFLDiagnosticTargetsTest(unittest.TestCase):
    def setUp(self):
        self.current_global = [np.array([10.0, 20.0])]
        self.local_updates = [
            [np.array([1.0, 2.0])],
            [np.array([3.0, 4.0])],
        ]
        self.local_weights = [
            [np.array([11.0, 22.0])],
            [np.array([13.0, 24.0])],
        ]
        self.ema_updates = [
            [np.array([0.5, 1.5])],
            [np.array([2.5, 3.5])],
        ]
        self.l_mat = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)

    def test_flatten_diagnostic_post_updates_returns_update_delta_matrix(self):
        mat, target, diag = flatten_diagnostic_post_updates(
            current_global=self.current_global,
            local_weights=self.local_weights,
            local_updates=self.local_updates,
            ema_updates=self.ema_updates,
            l_mat=self.l_mat,
            aggregation_target="update",
            filter_strength=1.0,
        )

        self.assertEqual(target, "update_delta")
        self.assertEqual(diag, {})
        np.testing.assert_allclose(mat, [[1.0, 2.0], [3.0, 4.0]])

    def test_flatten_diagnostic_post_updates_returns_weight_delta_matrix(self):
        mat, target, diag = flatten_diagnostic_post_updates(
            current_global=self.current_global,
            local_weights=self.local_weights,
            local_updates=self.local_updates,
            ema_updates=self.ema_updates,
            l_mat=self.l_mat,
            aggregation_target="weight",
            filter_strength=1.0,
        )

        self.assertEqual(target, "local_weight_delta")
        self.assertEqual(diag, {})
        np.testing.assert_allclose(mat, [[1.0, 2.0], [3.0, 4.0]])

    def test_flatten_diagnostic_post_updates_supports_filtered_ema_aliases(self):
        mat, target, diag = flatten_diagnostic_post_updates(
            current_global=self.current_global,
            local_weights=self.local_weights,
            local_updates=self.local_updates,
            ema_updates=self.ema_updates,
            l_mat=self.l_mat,
            aggregation_target="graph_filtered_ema_update",
            filter_strength=0.0,
        )

        self.assertEqual(target, "spectral_filtered_client_ema_update_delta")
        self.assertEqual(diag["graph_filter_strength"], 0.0)
        np.testing.assert_allclose(mat, [[0.5, 1.5], [2.5, 3.5]])

    def test_flatten_diagnostic_post_updates_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "Unknown diagnostic aggregation_target"):
            flatten_diagnostic_post_updates(
                current_global=self.current_global,
                local_weights=self.local_weights,
                local_updates=self.local_updates,
                ema_updates=self.ema_updates,
                l_mat=self.l_mat,
                aggregation_target="server_gcn_exact",
                filter_strength=1.0,
            )


if __name__ == "__main__":
    unittest.main()
