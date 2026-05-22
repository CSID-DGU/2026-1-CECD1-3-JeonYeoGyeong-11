import unittest

import numpy as np

from graphfl_lab.graph.builders import build_client_graph, build_relation_graph
from graphfl_lab.graph.clustering import build_block_uniform_graph, cluster_clients


class ClusteringControlTest(unittest.TestCase):
    def test_cluster_clients_kmeans_is_deterministic_for_seed(self):
        z = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        c1 = cluster_clients(z, method="kmeans", k=2, seed=7)
        c2 = cluster_clients(z, method="kmeans", k=2, seed=7)
        self.assertTrue(np.array_equal(c1, c2))

    def test_block_uniform_graph_is_symmetric_zero_diagonal(self):
        cid = np.array([0, 0, 1, 1], dtype=np.int64)
        g = build_block_uniform_graph(cid, intra=0.5, inter=0.1)
        self.assertTrue(np.allclose(g, g.T))
        self.assertTrue(np.allclose(np.diag(g), 0.0))
        self.assertGreater(float(g[0, 1]), float(g[0, 2]))

    def test_builder_supports_clustering_only_family(self):
        z = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        g = build_client_graph(
            z,
            mode="knn",
            knn_k=1,
            correction_family="clustering_only",
            cluster_method="kmeans",
            cluster_k=2,
            cluster_seed=3,
        )
        self.assertTrue(np.allclose(g, g.T))
        self.assertTrue(np.allclose(np.diag(g), 0.0))

    def test_relation_graph_returns_cluster_metadata(self):
        z = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        _, meta = build_relation_graph(
            z,
            mode="knn",
            knn_k=1,
            correction_family="clustering_only",
            cluster_method="kmeans",
            cluster_k=2,
            cluster_seed=3,
        )

        self.assertEqual(meta["graph_kind"], "clustering_only:block_uniform")
        self.assertEqual(len(meta["cluster_ids"]), z.shape[0])
        self.assertTrue(all(isinstance(x, int) for x in meta["cluster_ids"]))


if __name__ == "__main__":
    unittest.main()
