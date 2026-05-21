import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from graphfl_lab.strategies.graphfl.diagnostic_artifacts import (
    write_round_diagnostic_artifacts,
)


class GraphFLDiagnosticArtifactsTest(unittest.TestCase):
    def test_write_round_diagnostic_artifacts_preserves_artifact_contract(self):
        pre_post = {
            "round": {
                "di_pre": 0.1,
                "di_post": 0.2,
                "neff_pre": 2.0,
                "neff_post": 1.7,
                "align_mean_pre": 0.3,
                "align_mean_post": 0.4,
                "loo_mean_pre": 0.5,
                "loo_mean_post": 0.6,
                "alpha_entropy": 0.7,
            },
            "norms_pre": np.array([1.0, 2.0]),
            "norms_post": np.array([1.1, 2.1]),
            "q_pre": np.array([0.2, 0.3]),
            "q_post": np.array([0.4, 0.5]),
            "align_pre": np.array([0.6, 0.7]),
            "align_post": np.array([0.8, 0.9]),
            "loo_pre": np.array([1.0, 1.1]),
            "loo_post": np.array([1.2, 1.3]),
        }
        graph_diag = {
            "graph_density": 0.9,
            "graph_entropy": 0.8,
            "graph_num_nodes": 2,
            "number_of_edges": 1,
            "graph_degree_mean": 1.0,
            "graph_degree_min": 1,
            "graph_degree_max": 1,
            "graph_empty": False,
        }
        counterfactual_rows = [{"counterfactual": "actual"}]
        trace_rows = [{"phase": "graph"}]

        with patch(
            "graphfl_lab.strategies.graphfl.diagnostic_artifacts.append_round_metrics_csv"
        ) as append_round, patch(
            "graphfl_lab.strategies.graphfl.diagnostic_artifacts.append_graph_stats_csv"
        ) as append_graph, patch(
            "graphfl_lab.strategies.graphfl.diagnostic_artifacts.append_client_metrics_csv"
        ) as append_client, patch(
            "graphfl_lab.strategies.graphfl.diagnostic_artifacts.append_counterfactual_metrics_csv"
        ) as append_counterfactual, patch(
            "graphfl_lab.strategies.graphfl.diagnostic_artifacts.append_module_traces_jsonl"
        ) as append_traces, patch(
            "graphfl_lab.strategies.graphfl.diagnostic_artifacts.run_counterfactual_artifacts",
            return_value=SimpleNamespace(
                counterfactual_rows=counterfactual_rows,
                module_trace_rows=trace_rows,
            ),
        ) as run_counterfactual:
            write_round_diagnostic_artifacts(
                artifact_dir=Path("artifacts"),
                run_id="run-a",
                variant="ours",
                seed=7,
                server_round=3,
                accuracy=0.8,
                loss=0.2,
                pre_post=pre_post,
                graph_diag=graph_diag,
                wall_time_sec=1.25,
                graph_method="relation",
                correction_family="real_graph",
                graph_source_used="update",
                graph_variant="knn",
                diagnostic_target_used="update_delta",
                graph_used_source="ema_graph",
                graph_meta={"graph_kind": "semantic"},
                control_graph_mode="random",
                cluster_method="label",
                cluster_k=2,
                cluster_auto_k=True,
                cids=["0", "1"],
                n_examples_arr=np.array([5.0, 6.0]),
                client_cluster_ids=[4, -1],
                flat_updates=np.array([[1.0, 0.0], [0.0, 1.0]]),
                pre_weights=np.array([5.0, 6.0]),
                actual_adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
                aggregation_target="update",
                graph_seed=11,
                graph_filter_strength=0.5,
                graph_free_gamma=1.2,
                loo_enabled=True,
            )

        self.assertEqual(append_round.call_args.args[0], Path("artifacts") / "round_metrics.csv")
        round_row = append_round.call_args.args[1]
        self.assertEqual(round_row["aggregation_target"], "update_delta")
        self.assertEqual(round_row["graph_kind"], "semantic")

        graph_row = append_graph.call_args.args[1]
        self.assertEqual(graph_row["graph_used_source"], "ema_graph")
        self.assertEqual(graph_row["control_graph_mode"], "random")

        client_rows = append_client.call_args.args[1]
        self.assertEqual(client_rows[1]["cluster_id"], -1)
        self.assertEqual(client_rows[1]["loo_corrected"], 1.3)

        self.assertEqual(
            run_counterfactual.call_args.kwargs["diagnostic_target_used"],
            "update_delta",
        )
        self.assertEqual(
            run_counterfactual.call_args.kwargs["graph_filter_strength"],
            0.5,
        )
        self.assertEqual(append_counterfactual.call_args.args[1], counterfactual_rows)
        self.assertEqual(append_traces.call_args.args[1], trace_rows)


if __name__ == "__main__":
    unittest.main()
