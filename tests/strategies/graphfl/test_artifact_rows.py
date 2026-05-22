import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.artifact_rows import (
    build_client_diagnostic_rows,
    build_graph_stats_row,
    build_round_diagnostics_row,
)


class GraphFLArtifactRowsTest(unittest.TestCase):
    def test_build_round_diagnostics_row_preserves_full_metric_contract(self):
        row = build_round_diagnostics_row(
            run_id="run-a",
            variant="ours",
            seed=7,
            server_round=3,
            accuracy=0.8,
            loss=0.2,
            pre_post_round={
                "di_pre": 0.1,
                "di_post": 0.2,
                "neff_pre": 2.0,
                "neff_post": 1.5,
                "align_mean_pre": 0.3,
                "align_mean_post": 0.4,
                "loo_mean_pre": 0.5,
                "loo_mean_post": 0.6,
                "alpha_entropy": 0.7,
            },
            graph_diag={"graph_density": 0.9, "graph_entropy": 0.8},
            wall_time_sec=1.25,
            graph_method="relation",
            correction_family="real_graph",
            graph_source="update",
            graph_variant="knn",
            aggregation_target="update_delta",
            graph_kind="semantic",
        )

        self.assertEqual(row["round"], 3)
        self.assertEqual(row["di_pre"], 0.1)
        self.assertEqual(row["neff_post"], 1.5)
        self.assertEqual(row["loo_mean_post"], 0.6)
        self.assertEqual(row["graph_kind"], "semantic")

    def test_build_graph_stats_row_preserves_graph_metadata_fields(self):
        row = build_graph_stats_row(
            run_id="run-a",
            variant="ours",
            seed=7,
            server_round=3,
            graph_method="relation",
            correction_family="real_graph",
            graph_source="update",
            graph_variant="knn",
            aggregation_target="update_delta",
            graph_kind="semantic",
            graph_used_source="ema_graph",
            graph_diag={
                "graph_density": 0.9,
                "graph_entropy": 0.8,
                "graph_num_nodes": 4,
                "number_of_edges": 3,
                "graph_degree_mean": 1.5,
                "graph_degree_min": 1,
                "graph_degree_max": 2,
                "graph_empty": False,
            },
            control_graph_mode="random",
            cluster_method="label",
            cluster_k=2,
            cluster_auto_k=True,
        )

        self.assertEqual(row["graph_source_used"], "update")
        self.assertEqual(row["graph_used_source"], "ema_graph")
        self.assertEqual(row["number_of_edges"], 3)
        self.assertTrue(row["cluster_auto_k"])

    def test_build_client_diagnostic_rows_preserves_per_client_metrics(self):
        rows = build_client_diagnostic_rows(
            run_id="run-a",
            variant="ours",
            seed=7,
            server_round=3,
            cids=["0", "1"],
            n_examples_arr=np.array([5.0, 6.0]),
            pre_post={
                "norms_pre": np.array([1.0, 2.0]),
                "norms_post": np.array([1.5, 2.5]),
                "q_pre": np.array([0.1, 0.2]),
                "q_post": np.array([0.3, 0.4]),
                "align_pre": np.array([0.5, 0.6]),
                "align_post": np.array([0.7, 0.8]),
                "loo_pre": np.array([0.9, 1.0]),
                "loo_post": np.array([1.1, 1.2]),
            },
            client_cluster_ids=[4, -1],
        )

        self.assertEqual(rows[0]["cid"], "0")
        self.assertEqual(rows[0]["num_examples"], 5)
        self.assertEqual(rows[1]["cluster_id"], -1)
        self.assertEqual(rows[1]["loo_corrected"], 1.2)


if __name__ == "__main__":
    unittest.main()
