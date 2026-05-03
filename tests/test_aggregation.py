import unittest

import numpy as np

from spectral_fl.aggregation import (
    apply_min_client_weight,
    compute_conflict_weights,
    compute_effective_clients,
    compute_entropy,
)


class AggregationTest(unittest.TestCase):
    def test_conflict_weights_penalize_positive_residuals(self):
        e = np.array([0.1, 0.2, 0.9], dtype=np.float64)

        e_z, conflict_weights, _, disabled, e_mean, e_std = compute_conflict_weights(
            e=e,
            tau=1.0,
            e_std_threshold=0.0,
        )

        self.assertFalse(disabled)
        self.assertAlmostEqual(float(np.mean(e_z)), 0.0, places=6)
        self.assertAlmostEqual(e_mean, float(np.mean(e)), places=6)
        self.assertGreater(e_std, 0.0)
        self.assertLess(conflict_weights[2], conflict_weights[0])

    def test_conflict_weights_can_be_disabled_for_small_spread(self):
        e = np.array([0.2, 0.20001, 0.19999], dtype=np.float64)

        _, conflict_weights, _, disabled, _, _ = compute_conflict_weights(
            e=e,
            tau=1.0,
            e_std_threshold=0.1,
        )

        self.assertTrue(disabled)
        self.assertTrue(np.allclose(conflict_weights, np.ones_like(e)))

    def test_weight_utilities_are_normalized(self):
        weights = np.array([0.7, 0.2, 0.1], dtype=np.float64)
        floored = apply_min_client_weight(weights, min_w=0.2)

        self.assertAlmostEqual(float(np.sum(floored)), 1.0, places=6)
        self.assertGreaterEqual(float(np.min(floored)), 0.2)
        self.assertGreater(compute_entropy(floored), 0.0)
        self.assertGreaterEqual(compute_effective_clients(floored), 1.0)


if __name__ == "__main__":
    unittest.main()
