import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.update_space import (
    compute_local_updates,
    compute_update_space_arrays,
)


class GraphFLUpdateSpaceTest(unittest.TestCase):
    def test_compute_local_updates_subtracts_global_parameters(self):
        local_weights = [
            [np.array([2.0, 4.0]), np.array([5.0])],
            [np.array([3.0, 7.0]), np.array([8.0])],
        ]
        current_global = [np.array([1.0, 2.0]), np.array([3.0])]

        updates = compute_local_updates(
            local_weights=local_weights,
            current_global=current_global,
        )

        np.testing.assert_allclose(updates[0][0], [1.0, 2.0])
        np.testing.assert_allclose(updates[0][1], [2.0])
        np.testing.assert_allclose(updates[1][0], [2.0, 5.0])
        np.testing.assert_allclose(updates[1][1], [5.0])

    def test_compute_update_space_arrays_preserves_flat_delta_matrix_and_norms(self):
        local_weights = [
            [np.array([2.0, 4.0]), np.array([5.0])],
            [np.array([3.0, 7.0]), np.array([8.0])],
        ]
        local_updates = [
            [np.array([1.0, 2.0]), np.array([2.0])],
            [np.array([2.0, 5.0]), np.array([5.0])],
        ]
        ema_updates = [
            [np.array([0.5, 1.5]), np.array([1.0])],
            [np.array([1.0, 4.0]), np.array([4.0])],
        ]

        arrays = compute_update_space_arrays(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
        )

        np.testing.assert_allclose(
            arrays.flat_delta_matrix,
            [[1.0, 2.0, 2.0], [2.0, 5.0, 5.0]],
        )
        np.testing.assert_allclose(
            arrays.delta_norms,
            [3.0, np.sqrt(54.0)],
        )
        np.testing.assert_allclose(
            arrays.ema_delta_norms,
            [np.sqrt(3.5), np.sqrt(33.0)],
        )
        np.testing.assert_allclose(
            arrays.weight_norms,
            [np.sqrt(45.0), np.sqrt(122.0)],
        )


if __name__ == "__main__":
    unittest.main()
