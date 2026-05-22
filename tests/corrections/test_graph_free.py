import unittest

import numpy as np

from graphfl_lab.corrections.graph_free import (
    compute_contribution_cap_weights,
    resolve_graph_free_correction,
)


class GraphFreeCorrectionTest(unittest.TestCase):
    def test_contribution_cap_enforces_upper_bound_when_feasible(self):
        alpha = np.array([0.9, 0.1], dtype=np.float64)
        capped = compute_contribution_cap_weights(alpha, cap=0.6)
        self.assertAlmostEqual(float(np.sum(capped)), 1.0, places=6)
        self.assertLessEqual(float(np.max(capped)), 0.6 + 1e-6)

    def test_resolve_graph_free_correction_returns_mode_used(self):
        alpha, mode = resolve_graph_free_correction(
            alpha=np.array([0.8, 0.2], dtype=np.float64),
            mode="dominance_reweight",
            n_examples=np.array([90.0, 10.0], dtype=np.float64),
            gamma=1.0,
        )
        self.assertEqual(mode, "dominance_reweight")
        self.assertAlmostEqual(float(np.sum(alpha)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
