import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.round_projection import (
    build_projected_graph_space,
)


class GraphFLRoundProjectionTest(unittest.TestCase):
    def test_build_projected_graph_space_uses_source_vectors_and_projection(self):
        local_weights = [
            [np.array([2.0, 3.0])],
            [np.array([4.0, 7.0])],
        ]
        local_updates = [
            [np.array([1.0, 1.0])],
            [np.array([2.0, 3.0])],
        ]

        projected = build_projected_graph_space(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=None,
            graph_source="update",
            graph_layer_start=0,
            graph_layer_end=0,
            project_fn=lambda vector: vector * 2.0,
        )

        self.assertEqual(projected.graph_source_used, "update_delta")
        np.testing.assert_allclose(projected.graph_vectors[0], np.array([1.0, 1.0]))
        np.testing.assert_allclose(projected.z_mat[1], np.array([4.0, 6.0]))
        np.testing.assert_allclose(
            projected.graph_source_norms,
            np.array([np.sqrt(2.0), np.sqrt(13.0)]),
        )
        np.testing.assert_allclose(
            projected.z_norms,
            np.array([2.0 * np.sqrt(2.0), 2.0 * np.sqrt(13.0)]),
        )


if __name__ == "__main__":
    unittest.main()
