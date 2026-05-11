import unittest

import numpy as np

from spectral_fl.diagnostics.metrics import (
    compute_dominance_index,
    compute_effective_client_number,
    summarize_pre_post,
)


class DiagnosticMetricsTest(unittest.TestCase):
    def test_dominance_and_effective_clients(self):
        q = np.array([0.7, 0.2, 0.1], dtype=np.float64)
        self.assertAlmostEqual(compute_dominance_index(q), 0.7, places=6)
        self.assertGreater(compute_effective_client_number(q), 1.0)

    def test_summarize_pre_post_outputs_expected_keys(self):
        g = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        summary = summarize_pre_post(
            flat_updates=g,
            weights_pre=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            weights_post=np.array([0.8, 0.1, 0.1], dtype=np.float64),
            loo_enabled=True,
        )
        self.assertIn("round", summary)
        self.assertIn("q_pre", summary)
        self.assertIn("q_post", summary)
        self.assertIn("di_pre", summary["round"])
        self.assertIn("neff_post", summary["round"])
        self.assertEqual(summary["q_pre"].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
