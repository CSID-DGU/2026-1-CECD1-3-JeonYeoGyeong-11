import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.round_weights import select_round_weights


class GraphFLRoundWeightsTest(unittest.TestCase):
    def test_select_round_weights_preserves_conflict_aware_alpha_mode(self):
        selection = select_round_weights(
            n_examples=np.array([10.0, 10.0, 10.0]),
            conflict_weight=np.array([1.0, 0.5, 0.25]),
            diagnostic_only=False,
            in_warmup=False,
            estd_disabled=False,
            graph_fallback_used=False,
            conflict_mix=1.0,
            min_client_weight=0.0,
            correction_family="real_graph",
            graph_free_mode="none",
            graph_free_gamma=1.0,
            contribution_cap=0.0,
            clip_quantile=0.9,
            update_norms=np.array([1.0, 2.0, 3.0]),
        )

        self.assertEqual(selection.alpha_mode, "conflict_aware")
        self.assertLess(selection.alpha_norm[2], selection.alpha_norm[0])
        self.assertTrue(np.allclose(selection.conflict_weight, [1.0, 0.5, 0.25]))

    def test_select_round_weights_preserves_graph_free_alpha_mode_suffix(self):
        selection = select_round_weights(
            n_examples=np.array([90.0, 10.0]),
            conflict_weight=np.ones(2, dtype=np.float64),
            diagnostic_only=False,
            in_warmup=False,
            estd_disabled=False,
            graph_fallback_used=False,
            conflict_mix=0.0,
            min_client_weight=0.0,
            correction_family="graph_free",
            graph_free_mode="contribution_cap",
            graph_free_gamma=1.0,
            contribution_cap=0.6,
            clip_quantile=0.9,
            update_norms=np.array([1.0, 2.0]),
        )

        self.assertIn("graph_free:contribution_cap", selection.alpha_mode)
        self.assertLessEqual(float(np.max(selection.alpha_norm)), 0.6 + 1e-6)
        self.assertAlmostEqual(float(np.sum(selection.alpha_norm)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
