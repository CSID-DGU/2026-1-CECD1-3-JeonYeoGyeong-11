import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from graphfl_lab.strategies.graphfl.round_outputs import (
    build_strategy_round_outputs,
)


class GraphFLRoundOutputsTest(unittest.TestCase):
    def test_build_strategy_round_outputs_composes_log_trace_and_metrics(self):
        spectral_context = {"h_spec": 0.1, "spectral_diag": {}}
        conflict_context = {"e": np.array([0.2]), "e_mean": 0.2}
        update_context = {"delta_norms": np.array([1.0])}
        graph_context = {"graph_diag": {"graph_density": 1.0}}
        alpha_context = {"alpha_mode": "conflict_aware"}
        client_context = {"client_train_acc": [0.8]}
        round_log = {"round": 3}
        fit_metrics = {"h_spec": 0.1}

        with patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_spectral_context",
            return_value=spectral_context,
        ) as build_spectral, patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_conflict_context",
            return_value=conflict_context,
        ) as build_conflict, patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_update_context",
            return_value=update_context,
        ), patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_graph_context",
            return_value=graph_context,
        ), patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_alpha_context",
            return_value=alpha_context,
        ) as build_alpha, patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_client_context",
            return_value=client_context,
        ), patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_round_log",
            return_value=round_log,
        ) as build_log, patch(
            "graphfl_lab.strategies.graphfl.round_outputs.make_round_trace_payload",
            return_value={"trace_semantic": "real_graph"},
        ) as make_trace, patch(
            "graphfl_lab.strategies.graphfl.round_outputs.build_fit_metrics",
            return_value=fit_metrics,
        ) as build_metrics:
            outputs = build_strategy_round_outputs(
                server_round=3,
                cids=["0"],
                spectral_metrics=SimpleNamespace(h_spec=0.1),
                h_spec_ema=0.2,
                in_warmup=False,
                conflict_metrics=SimpleNamespace(filter_diag={"gain": 1.0}),
                target_filter_diag={"target": 1},
                diagnostic_filter_diag={"diagnostic": 2},
                conflict_weight=np.array([0.9]),
                graph_fallback_used=False,
                z_norms=np.array([1.0]),
                update_space=SimpleNamespace(delta_norms=np.array([1.0])),
                graph_source_norms=np.array([2.0]),
                ema_update_source="ema_update",
                graph_source_used="update",
                graph_used_source="ema_graph",
                graph_meta={"graph_kind": "semantic"},
                graph_diag_current={"graph_density": 0.8},
                graph_diag={"graph_density": 0.9},
                w_matrix_log=[[0.0]],
                alpha_raw=np.array([1.0]),
                alpha_norm=np.array([1.0]),
                alpha_mode="conflict_aware",
                active_client_mask=np.array([True]),
                aggregation_target_used="update_delta",
                diagnostic_target_used="update_delta",
                server_opt_diag={"server_momentum_active": False},
                pre_post={"round": {"di_pre": 0.1}},
                n_examples_arr=np.array([10.0]),
                client_train_acc=[0.8],
                client_train_loss=[0.2],
                config_context={"graph_mode": "knn"},
                correction_family="real_graph",
                control_graph_mode="random",
                graph_mode="knn",
            )

        self.assertEqual(outputs.round_log["trace_semantic"], "real_graph")
        self.assertEqual(outputs.fit_metrics, fit_metrics)
        self.assertEqual(build_spectral.call_args.kwargs["h_spec_ema"], 0.2)
        self.assertEqual(build_conflict.call_args.kwargs["conflict_weight"].tolist(), [0.9])
        self.assertEqual(build_alpha.call_args.kwargs["diagnostic_target_used"], "update_delta")
        self.assertEqual(build_log.call_args.kwargs["cids"], ["0"])
        self.assertEqual(make_trace.call_args.kwargs["pre_post_round"], {"di_pre": 0.1})
        self.assertEqual(build_metrics.call_args.kwargs["filter_diag"], {"gain": 1.0})


if __name__ == "__main__":
    unittest.main()
