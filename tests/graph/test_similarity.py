import unittest

import numpy as np

from graphfl_lab.graph.similarity import (
    dense_absolute_cosine,
    dense_negative_cosine,
    dense_positive_cosine,
    pairwise_sq_dists,
    resolve_distance_sigma,
)


class GraphSimilarityPackageTest(unittest.TestCase):
    def test_cosine_variants_are_symmetric_with_zero_diagonal(self):
        z_mat = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float64,
        )

        for sim in [
            dense_positive_cosine(z_mat),
            dense_absolute_cosine(z_mat),
            dense_negative_cosine(z_mat),
        ]:
            self.assertTrue(np.allclose(sim, sim.T))
            self.assertTrue(np.allclose(np.diag(sim), 0.0))

        self.assertGreater(dense_negative_cosine(z_mat)[0, 2], 0.0)
        self.assertGreater(dense_absolute_cosine(z_mat)[0, 2], 0.0)
        self.assertEqual(float(dense_positive_cosine(z_mat)[0, 2]), 0.0)

    def test_distance_sigma_uses_positive_upper_distances(self):
        z_mat = np.array([[0.0], [2.0], [4.0]], dtype=np.float64)
        d2 = pairwise_sq_dists(z_mat)

        self.assertTrue(np.allclose(np.diag(d2), 0.0))
        self.assertEqual(resolve_distance_sigma(d2, sigma=3.0), 3.0)
        self.assertAlmostEqual(resolve_distance_sigma(d2, sigma=0.0), 2.0)


if __name__ == "__main__":
    unittest.main()
