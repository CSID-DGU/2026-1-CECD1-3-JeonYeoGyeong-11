import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.projection import project_with_cached_matrix


class GraphFLProjectionTest(unittest.TestCase):
    def test_project_with_cached_matrix_returns_float32_when_small(self):
        vec = np.array([1.0, 2.0], dtype=np.float64)

        projected, matrix = project_with_cached_matrix(
            vec,
            projection_matrix=None,
            compression_dim=4,
            compression_seed=7,
        )

        self.assertIsNone(matrix)
        self.assertEqual(projected.dtype, np.float32)
        np.testing.assert_allclose(projected, [1.0, 2.0])

    def test_project_with_cached_matrix_creates_and_reuses_matrix(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        first, matrix = project_with_cached_matrix(
            vec,
            projection_matrix=None,
            compression_dim=2,
            compression_seed=7,
        )
        second, reused = project_with_cached_matrix(
            vec,
            projection_matrix=matrix,
            compression_dim=2,
            compression_seed=999,
        )

        self.assertIs(reused, matrix)
        self.assertEqual(matrix.shape, (3, 2))
        np.testing.assert_allclose(second, first)


if __name__ == "__main__":
    unittest.main()
