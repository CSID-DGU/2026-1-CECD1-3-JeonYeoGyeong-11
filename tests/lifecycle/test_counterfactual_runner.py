import unittest

import numpy as np

from graphfl_lab.diagnostics.metrics import summarize_pre_post
from graphfl_lab.lifecycle.counterfactuals import CounterfactualSpec, default_counterfactual_specs
from graphfl_lab.lifecycle.diagnostic_runner import (
    CounterfactualDiagnosticRunner,
    MinimalAggregationAdapter,
)


class CounterfactualRunnerTest(unittest.TestCase):
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

    def test_actual_path_metrics_match_existing_summary_for_global_update(self):
        spec = CounterfactualSpec(
            name="actual",
            topology_mode="actual",
            aggregation_target="global_update",
        )
        runner = CounterfactualDiagnosticRunner(loo_enabled=True, rng=np.random.default_rng(4))

        result = runner.run(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
            specs=[spec],
            round_number=3,
        )[0]
        expected = summarize_pre_post(
            flat_updates=self.flat_updates,
            flat_updates_post=self.flat_updates,
            weights_pre=self.weights,
            weights_post=self.weights,
            loo_enabled=True,
        )["round"]

        self.assertEqual(result.name, "actual")
        self.assertAlmostEqual(result.metrics["di_pre"], expected["di_pre"], places=6)
        self.assertAlmostEqual(result.metrics["di_post"], expected["di_post"], places=6)
        self.assertEqual(result.trace_records[0].phase, "counterfactual")

    def test_default_counterfactual_set_emits_required_variants(self):
        runner = CounterfactualDiagnosticRunner(rng=np.random.default_rng(7))

        results = runner.run(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
        )

        names = {result.name for result in results}
        expected = {spec.name for spec in default_counterfactual_specs()}
        self.assertTrue(expected.issubset(names))
        self.assertIn("matched_random", names)
        self.assertIn("graphfree_dominance_reweight", names)
        for result in results:
            self.assertEqual(result.metrics["status"], "ok")
            self.assertIn("graph_density", result.metrics)
            self.assertIn("alpha_matrix_entropy", result.metrics)

    def test_shadow_paths_do_not_mutate_inputs(self):
        updates_before = self.flat_updates.copy()
        adjacency_before = self.adjacency.copy()
        runner = CounterfactualDiagnosticRunner(rng=np.random.default_rng(1))

        runner.run(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
        )

        self.assertTrue(np.allclose(self.flat_updates, updates_before))
        self.assertTrue(np.allclose(self.adjacency, adjacency_before))

    def test_graphfree_dominance_reweight_changes_weights_without_graph_mixing(self):
        adapter = MinimalAggregationAdapter(dominance_gamma=1.0)
        runner = CounterfactualDiagnosticRunner(aggregation_adapter=adapter)
        spec = CounterfactualSpec(
            name="graphfree_dominance_reweight",
            topology_mode="identity",
            aggregation_target="graphfree_dominance_reweight",
            graph_free_mode="dominance_reweight",
        )

        result = runner.run(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
            specs=[spec],
        )[0]

        self.assertFalse(np.allclose(result.weights_post, self.weights / np.sum(self.weights)))
        self.assertTrue(np.allclose(result.post_flat_updates, self.flat_updates))
        self.assertEqual(result.metrics["aggregation_target_used"], "dominance_reweight")

    def test_filtered_delta_aliases_are_executable_for_warmup_free_runs(self):
        runner = CounterfactualDiagnosticRunner(rng=np.random.default_rng(3))
        specs = [
            CounterfactualSpec(
                name="filtered_update_delta",
                aggregation_target="spectral_filtered_update_delta",
            ),
            CounterfactualSpec(
                name="filtered_weight_delta",
                aggregation_target="spectral_filtered_local_weight_delta",
            ),
            CounterfactualSpec(
                name="filtered_ema_delta",
                aggregation_target="spectral_filtered_client_ema_update_delta",
            ),
        ]

        results = runner.run(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
            specs=specs,
        )

        self.assertEqual([result.metrics["status"] for result in results], ["ok", "ok", "ok"])
        self.assertEqual(
            results[0].metrics["aggregation_target_used"],
            "graph_filtered_update_delta",
        )

    def test_unsupported_or_invalid_variant_emits_error_record(self):
        runner = CounterfactualDiagnosticRunner()
        spec = CounterfactualSpec(name="bad_target", aggregation_target="server_gcn_exact")

        result = runner.run(
            flat_updates=self.flat_updates,
            weights_pre=self.weights,
            actual_adjacency=self.adjacency,
            specs=[spec],
        )[0]

        self.assertEqual(result.metrics["status"], "error")
        self.assertEqual(result.trace_records[0].values["status"], "error")


if __name__ == "__main__":
    unittest.main()
