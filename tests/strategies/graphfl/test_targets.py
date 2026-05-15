import unittest

import numpy as np

from spectral_fl.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregate_target,
)


class GraphFLAggregationTargetTest(unittest.TestCase):
    def test_update_target_adds_weighted_delta_to_global(self):
        current = [np.array([1.0, 1.0], dtype=np.float32)]
        updates = [
            [np.array([1.0, 0.0], dtype=np.float32)],
            [np.array([0.0, 1.0], dtype=np.float32)],
        ]
        weights = [
            [current[0] + updates[0][0]],
            [current[0] + updates[1][0]],
        ]

        result, used, diag = aggregate_target(
            current_global=current,
            local_weights=weights,
            local_updates=updates,
            alpha_norm=np.array([0.25, 0.75], dtype=np.float64),
            config=AggregationTargetConfig(target="update"),
        )

        self.assertEqual(used, "update_delta")
        self.assertEqual(diag, {})
        self.assertTrue(np.allclose(result[0], np.array([1.25, 1.75])))

    def test_weight_target_averages_local_weights(self):
        current = [np.array([0.0, 0.0], dtype=np.float32)]
        updates = [
            [np.array([1.0, 1.0], dtype=np.float32)],
            [np.array([3.0, 5.0], dtype=np.float32)],
        ]

        result, used, _ = aggregate_target(
            current_global=current,
            local_weights=updates,
            local_updates=updates,
            alpha_norm=np.array([0.5, 0.5], dtype=np.float64),
            config=AggregationTargetConfig(target="weight"),
        )

        self.assertEqual(used, "local_weight")
        self.assertTrue(np.allclose(result[0], np.array([2.0, 3.0])))

    def test_spectral_filtered_update_requires_laplacian(self):
        current = [np.array([0.0], dtype=np.float32)]
        updates = [[np.array([1.0], dtype=np.float32)]]

        with self.assertRaisesRegex(ValueError, "requires a client Laplacian"):
            aggregate_target(
                current_global=current,
                local_weights=updates,
                local_updates=updates,
                alpha_norm=np.array([1.0], dtype=np.float64),
                config=AggregationTargetConfig(target="spectral_filtered_update"),
            )

    def test_graph_filtered_update_alias_requires_laplacian(self):
        current = [np.array([0.0], dtype=np.float32)]
        updates = [[np.array([1.0], dtype=np.float32)]]

        with self.assertRaisesRegex(ValueError, "requires a client Laplacian"):
            aggregate_target(
                current_global=current,
                local_weights=updates,
                local_updates=updates,
                alpha_norm=np.array([1.0], dtype=np.float64),
                config=AggregationTargetConfig(target="graph_filtered_update"),
            )


if __name__ == "__main__":
    unittest.main()
