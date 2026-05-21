import unittest

import numpy as np

from graphfl_lab.graph.builders import build_relation_graph
from graphfl_lab.update_graph import build_client_graph, compute_graph_diagnostics


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

    def test_rbf_graph_prefers_nearby_updates(self):
        z_mat = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        )

        graph = build_client_graph(z_mat, mode="rbf", graph_scale_sigma=1.0)

        self.assertGreater(float(graph[0, 1]), float(graph[0, 2]))
        self.assertTrue(np.allclose(graph, graph.T))
        self.assertTrue(np.allclose(np.diag(graph), 0.0))

    def test_learned_smooth_graph_is_nonnegative_and_normalized_by_rows(self):
        z_mat = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        )

        graph = build_client_graph(z_mat, mode="learned_smooth", learned_graph_lambda=1.0)

        self.assertTrue(np.all(graph >= 0.0))
        self.assertTrue(np.allclose(graph, graph.T))
        self.assertTrue(np.allclose(np.diag(graph), 0.0))
        self.assertGreater(float(graph[0, 1]), float(graph[0, 2]))

    def test_signed_abs_graph_connects_opposite_directions(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

        positive = build_client_graph(z_mat, mode="dense")
        signed_abs = build_client_graph(z_mat, mode="signed_abs")

        self.assertEqual(float(positive[0, 1]), 0.0)
        self.assertAlmostEqual(float(signed_abs[0, 1]), 1.0, places=6)
        self.assertTrue(np.allclose(signed_abs, signed_abs.T))

    def test_negative_graph_keeps_only_anti_alignment(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )

        negative = build_client_graph(z_mat, mode="negative")

        self.assertAlmostEqual(float(negative[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(negative[0, 2]), 0.0, places=6)
        self.assertTrue(np.allclose(np.diag(negative), 0.0))

    def test_pfedgraph_qp_uses_similarity_and_sample_prior(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

        graph, meta = build_relation_graph(
            z_mat,
            mode="pfedgraph_qp",
            learned_graph_lambda=1.0,
            client_sample_weights=[0.7, 0.1, 0.1, 0.1],
        )

        self.assertTrue(np.all(graph >= 0.0))
        self.assertTrue(np.allclose(graph, graph.T))
        self.assertTrue(np.allclose(np.diag(graph), 0.0))
        self.assertGreater(float(graph[0, 1]), float(graph[0, 2]))
        self.assertEqual(
            meta["base_graph_kind"],
            "pfedgraph_qp:symmetric_diagnostic_projection",
        )


if __name__ == "__main__":
    unittest.main()
