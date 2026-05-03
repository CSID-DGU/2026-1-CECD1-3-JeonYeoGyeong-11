import unittest

import numpy as np

from spectral_fl.update_graph import build_client_graph, compute_graph_diagnostics


class UpdateGraphTest(unittest.TestCase):
    def test_knn_graph_is_symmetric_without_diagonal(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float64,
        )

        graph = build_client_graph(z_mat, mode="knn", knn_k=1)

        self.assertTrue(np.allclose(graph, graph.T))
        self.assertTrue(np.allclose(np.diag(graph), 0.0))
        self.assertGreater(compute_graph_diagnostics(graph)["number_of_edges"], 0)

    def test_random_graph_matches_knn_edge_count(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(7)

        knn = build_client_graph(z_mat, mode="knn", knn_k=1)
        random_graph = build_client_graph(z_mat, mode="random", knn_k=1, rng=rng)

        self.assertEqual(
            compute_graph_diagnostics(knn)["number_of_edges"],
            compute_graph_diagnostics(random_graph)["number_of_edges"],
        )

    def test_mutual_knn_is_no_denser_than_knn(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )

        knn = build_client_graph(z_mat, mode="knn", knn_k=1)
        mutual = build_client_graph(z_mat, mode="mutual_knn", knn_k=1)

        self.assertTrue(np.allclose(mutual, mutual.T))
        self.assertTrue(np.allclose(np.diag(mutual), 0.0))
        self.assertLessEqual(
            compute_graph_diagnostics(mutual)["number_of_edges"],
            compute_graph_diagnostics(knn)["number_of_edges"],
        )

    def test_magnitude_aware_graph_keeps_nonnegative_weights(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [10.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dense = build_client_graph(z_mat, mode="dense")
        magnitude = build_client_graph(z_mat, mode="magnitude")

        self.assertTrue(np.all(magnitude >= 0.0))
        self.assertTrue(np.allclose(magnitude, magnitude.T))
        self.assertLess(magnitude[0, 1], dense[0, 1])


if __name__ == "__main__":
    unittest.main()
