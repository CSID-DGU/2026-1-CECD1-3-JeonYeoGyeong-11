import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.round_graph import build_round_graph_state


class GraphFLRoundGraphTest(unittest.TestCase):
    def test_build_round_graph_state_preserves_sample_weight_normalization(self):
        state = build_round_graph_state(
            z_mat=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float64),
            cids=["0", "1", "2"],
            n_examples_arr=np.array([8.0, 2.0, 0.0], dtype=np.float64),
            server_round=2,
            graph_seed=3,
            graph_mode="dense",
            knn_k=1,
            edge_threshold=0.0,
            graph_scale_sigma=1.0,
            learned_graph_lambda=1.0,
            correction_family="real_graph",
            control_graph_mode="random",
            graph_source_used="update",
            aggregation_target="update",
            cluster_method="none",
            cluster_k=0,
            cluster_auto_k=False,
            previous_graph_ema=None,
            use_ema_graph=True,
            in_warmup=False,
            ema_alpha=0.8,
        )

        np.testing.assert_allclose(state.pre_weights, [0.8, 0.2, 0.0])
        self.assertEqual(state.graph_meta["graph_kind"], "real_graph")
        self.assertEqual(state.graph_used_source, "ema_graph")
        self.assertEqual(state.client_cluster_ids, [-1, -1, -1])
        self.assertFalse(state.graph_fallback_used)

    def test_build_round_graph_state_preserves_seeded_control_graphs(self):
        kwargs = dict(
            z_mat=np.array([[1.0], [0.0], [2.0], [3.0]], dtype=np.float64),
            cids=["0", "1", "2", "3"],
            n_examples_arr=np.ones(4, dtype=np.float64),
            server_round=4,
            graph_seed=11,
            graph_mode="knn",
            knn_k=1,
            edge_threshold=0.0,
            graph_scale_sigma=1.0,
            learned_graph_lambda=1.0,
            correction_family="control_graph",
            control_graph_mode="random",
            graph_source_used="update",
            aggregation_target="update",
            cluster_method="none",
            cluster_k=0,
            cluster_auto_k=False,
            previous_graph_ema=None,
            use_ema_graph=False,
            in_warmup=False,
            ema_alpha=0.8,
        )

        first = build_round_graph_state(**kwargs)
        second = build_round_graph_state(**kwargs)

        np.testing.assert_allclose(first.current_graph, second.current_graph)
        self.assertEqual(first.graph_meta["graph_kind"], "control:random")
        self.assertEqual(first.graph_used_source, "raw_current_graph")


if __name__ == "__main__":
    unittest.main()
