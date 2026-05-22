import unittest
from types import SimpleNamespace

from graphfl_lab.strategies.graphfl.config_context import build_config_context


class GraphFLConfigContextTest(unittest.TestCase):
    def test_build_config_context_projects_explicit_round_config_fields(self):
        strategy = SimpleNamespace(
            adaptive_tau=True,
            aggregation_target="graph_filtered_update",
            client_update_ema_alpha=0.7,
            conflict_mix=0.5,
            diagnostic_only=False,
            diagnostics_enable=True,
            loo_enabled=True,
            edge_threshold=0.1,
            e_std_threshold=0.2,
            fixed_tau=1.0,
            graph_layer_end=2,
            graph_layer_start=1,
            graph_mode="knn",
            graph_method="semantic",
            graph_scale_sigma=1.5,
            graph_source="update",
            correction_family="real_graph",
            control_graph_mode="matched_random",
            cluster_method="label",
            cluster_k=3,
            cluster_auto_k=True,
            graph_free_mode="none",
            graph_free_gamma=1.0,
            clip_quantile=0.9,
            contribution_cap=0.0,
            knn_k=2,
            learned_graph_lambda=1.0,
            min_client_weight=0.01,
            server_learning_rate=0.5,
            server_momentum=0.1,
            graph_filter_strength=0.75,
            tau_source="h_spec",
            use_ema_graph=True,
            warmup_rounds=2,
        )

        context = build_config_context(strategy)

        self.assertEqual(context["aggregation_target"], "graph_filtered_update")
        self.assertEqual(context["correction_family"], "real_graph")
        self.assertEqual(context["control_graph_mode"], "matched_random")
        self.assertEqual(context["graph_filter_strength"], 0.75)
        self.assertEqual(context["graph_filter_strength"], 0.75)
        self.assertEqual(context["warmup_rounds"], 2)


if __name__ == "__main__":
    unittest.main()
