import unittest

import numpy as np

from spectral_fl.graph.builders import build_client_graph
from spectral_fl.graph.controls import (
    build_identity_graph,
    build_random_matched_graph,
    build_shuffled_graph,
)
from spectral_fl.graph.diagnostics import compute_graph_diagnostics


class ControlGraphTest(unittest.TestCase):
    def setUp(self):
        self.z_mat = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        self.base = build_client_graph(self.z_mat, mode="knn", knn_k=1)

    def test_identity_graph_is_all_zero_off_diagonal(self):
        ident = build_identity_graph(self.base.shape[0])
        self.assertTrue(np.allclose(ident, 0.0))

    def test_shuffled_graph_preserves_weight_multiset(self):
        rng = np.random.default_rng(11)
        shuffled = build_shuffled_graph(self.base, rng=rng)
        iu = np.triu_indices(self.base.shape[0], k=1)
        base_vals = np.sort(self.base[iu][self.base[iu] > 0.0])
        shuf_vals = np.sort(shuffled[iu][shuffled[iu] > 0.0])
        self.assertTrue(np.allclose(base_vals, shuf_vals))
        self.assertTrue(np.allclose(shuffled, shuffled.T))

    def test_random_matched_graph_preserves_edge_count(self):
        rng = np.random.default_rng(7)
        rnd = build_random_matched_graph(self.base, rng=rng)
        self.assertEqual(
            compute_graph_diagnostics(self.base)["number_of_edges"],
            compute_graph_diagnostics(rnd)["number_of_edges"],
        )

    def test_graph_diagnostics_include_normalized_entropy(self):
        diag = compute_graph_diagnostics(self.base)
        self.assertIn("graph_entropy", diag)
        self.assertGreaterEqual(float(diag["graph_entropy"]), 0.0)
        self.assertLessEqual(float(diag["graph_entropy"]), 1.0)
        self.assertEqual(int(diag["graph_num_nodes"]), self.base.shape[0])

    def test_builder_dispatch_uses_control_graph_mode(self):
        g = build_client_graph(
            self.z_mat,
            mode="knn",
            knn_k=1,
            correction_family="control_graph",
            control_graph_mode="identity",
        )
        self.assertTrue(np.allclose(g, 0.0))


if __name__ == "__main__":
    unittest.main()
