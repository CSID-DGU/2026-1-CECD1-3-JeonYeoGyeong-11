import unittest
from types import SimpleNamespace

import numpy as np

from graphfl_lab.strategies.graphfl.round_context import (
    build_alpha_context,
    build_client_context,
    build_conflict_context,
    build_graph_context,
    build_spectral_context,
    build_update_context,
)


class GraphFLRoundContextTest(unittest.TestCase):
    def test_build_spectral_context_preserves_tau_and_filter_fields(self):
        spectral_metrics = SimpleNamespace(
            h_spec=0.1,
            h_spec_current=0.2,
            h_spec_raw_current=0.3,
            h_spec_normalized=0.4,
            h_spec_current_normalized=0.5,
            h_spec_raw_current_normalized=0.6,
            metric_lambda_max=2.0,
            h_spec_ema_candidate=0.7,
            metric_graph_source="current_round_graph",
            spectral_diag={"spectral_entropy": 0.8},
        )
        tau_source = SimpleNamespace(
            source_used="h_spec",
            source_signal=0.1,
            ema_value=0.2,
            ema_candidate=0.3,
        )
        conflict_metrics = SimpleNamespace(
            tau=1.2,
            tau_source=tau_source,
            filter_diag={"spectral_filter_gain_mean": 0.9},
        )

        context = build_spectral_context(
            spectral_metrics=spectral_metrics,
            h_spec_ema=0.11,
            in_warmup=False,
            conflict_metrics=conflict_metrics,
            target_filter_diag={"target": 1},
            diagnostic_filter_diag={"diag": 2},
        )

        self.assertEqual(context["tau"], 1.2)
        self.assertEqual(context["tau_source_used"], "h_spec")
        self.assertEqual(context["h_spec_ema"], 0.11)
        self.assertEqual(context["spectral_diag"]["spectral_entropy"], 0.8)
        self.assertEqual(context["target_filter_diag"], {"target": 1})

    def test_build_context_helpers_preserve_round_log_inputs(self):
        conflict_metrics = SimpleNamespace(
            e=np.array([0.1, 0.2]),
            e_z=np.array([-1.0, 1.0]),
            raw_cw=np.array([1.0, 0.5]),
            e_mean=0.15,
            e_std_raw=0.05,
            estd_disabled=False,
        )
        update_space = SimpleNamespace(
            delta_norms=np.array([1.0, 2.0]),
            ema_delta_norms=np.array([1.5, 2.5]),
            weight_norms=np.array([3.0, 4.0]),
        )
        pre_post = {
            "round": {"di_pre": 0.1},
            "q_pre": np.array([0.1]),
            "q_post": np.array([0.2]),
            "align_pre": np.array([0.3]),
            "align_post": np.array([0.4]),
            "loo_pre": np.array([0.5]),
            "loo_post": np.array([0.6]),
        }

        conflict = build_conflict_context(
            conflict_metrics=conflict_metrics,
            conflict_weight=np.array([0.9, 0.8]),
            graph_fallback_used=True,
        )
        update = build_update_context(
            z_norms=np.array([5.0]),
            update_space=update_space,
            graph_source_norms=np.array([6.0]),
            ema_update_source="ema_update",
        )
        graph = build_graph_context(
            graph_source_used="update",
            graph_used_source="ema_graph",
            graph_meta={"graph_kind": "real_graph"},
            graph_diag_current={"graph_density": 0.1},
            graph_diag={"graph_density": 0.2},
            w_matrix_log=[[0.0, 1.0]],
        )
        alpha = build_alpha_context(
            alpha_raw=np.array([1.0]),
            alpha_norm=np.array([1.0]),
            alpha_mode="conflict_aware",
            active_client_mask=np.array([True]),
            aggregation_target_used="update_delta",
            diagnostic_target_used="update_delta",
            server_opt_diag={"server_momentum_active": False},
            pre_post=pre_post,
        )
        client = build_client_context(
            n_examples_arr=np.array([10.0]),
            client_train_acc=[0.9],
            client_train_loss=[0.1],
        )

        self.assertTrue(conflict["graph_fallback_used"])
        self.assertEqual(update["ema_update_source"], "ema_update")
        self.assertEqual(graph["graph_used_source"], "ema_graph")
        self.assertEqual(alpha["alpha_mode"], "conflict_aware")
        self.assertEqual(client["client_train_acc"], [0.9])


if __name__ == "__main__":
    unittest.main()
