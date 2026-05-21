import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.counterfactual_artifacts import (
    counterfactual_seed_base,
    counterfactual_specs_for_target,
    run_counterfactual_artifacts,
)


class GraphFLCounterfactualArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.flat_updates = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        self.weights = np.array([10.0, 10.0, 2.0, 2.0], dtype=np.float64)
        self.adjacency = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_counterfactual_specs_for_target_retargets_graph_variants_only(self):
        specs = counterfactual_specs_for_target(
            diagnostic_target_used="spectral_filtered_update_delta",
            aggregation_target="update",
        )

        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(
            by_name["actual"].aggregation_target,
            "spectral_filtered_update_delta",
        )
        self.assertEqual(
            by_name["matched_random"].aggregation_target,
            "spectral_filtered_update_delta",
        )
        self.assertEqual(
            by_name["graphfree_dominance_reweight"].aggregation_target,
            "graphfree_dominance_reweight",
        )

    def test_counterfactual_seed_base_prefers_explicit_diagnostics_seed(self):
        self.assertEqual(
            counterfactual_seed_base(diagnostics_seed=12, graph_seed=99),
            12,
        )
        self.assertEqual(
            counterfactual_seed_base(diagnostics_seed=-1, graph_seed=99),
            99,
        )

    def test_run_counterfactual_artifacts_emits_rows_and_trace_context(self):
        artifacts = run_counterfactual_artifacts(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
            diagnostic_target_used="update_delta",
            aggregation_target="update",
            diagnostics_seed=7,
            graph_seed=3,
            server_round=2,
            graph_filter_strength=1.0,
            graph_free_gamma=1.0,
            loo_enabled=True,
            graph_meta={
                "lifecycle_trace": [
                    {"phase": "relation", "values": {"status": "ok"}}
                ]
            },
            run_id="run-a",
            variant="ours",
            graph_method="relation",
            graph_variant="knn",
        )

        names = {row["counterfactual"] for row in artifacts.counterfactual_rows}
        self.assertIn("actual", names)
        self.assertIn("matched_random", names)
        self.assertIn("graphfree_dominance_reweight", names)
        self.assertTrue(
            all(row["run_id"] == "run-a" for row in artifacts.counterfactual_rows)
        )
        self.assertEqual(artifacts.module_trace_rows[0]["values"]["run_id"], "run-a")
        self.assertTrue(
            any(
                row["values"].get("metrics", {}).get("counterfactual") == "actual"
                for row in artifacts.module_trace_rows
            )
        )


if __name__ == "__main__":
    unittest.main()
