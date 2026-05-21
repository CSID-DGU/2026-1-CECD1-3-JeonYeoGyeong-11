import unittest

import numpy as np

from graphfl_lab.lifecycle.aggregation import AggregationResult, GraphAggregationOperator
from graphfl_lab.lifecycle.context import AggregationContext, RoundContext
from graphfl_lab.lifecycle.topology import TopologyOutput


class AggregationOperatorTest(unittest.TestCase):
    def _context(self, *, config=None, cluster_ids=None):
        adjacency = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        return AggregationContext(
            round_context=RoundContext(
                server_round=1,
                cids=["a", "b", "c"],
                config={} if config is None else config,
            ),
            topology_output=TopologyOutput(
                adjacency=adjacency,
                graph_kind="knn",
                cluster_ids=cluster_ids,
            ),
            local_updates=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            num_examples=[2, 1, 1],
        )

    def test_global_update_aggregation_executes(self):
        result = GraphAggregationOperator("global_update").run(self._context())

        self.assertEqual(result.status, "ok")
        self.assertIsInstance(result.output, AggregationResult)
        self.assertEqual(result.output.aggregation_target, "update")
        self.assertTrue(np.allclose(result.output.alpha, np.array([0.5, 0.25, 0.25])))
        self.assertIn("alpha_entropy", result.trace_records[0].values)

    def test_spectral_filtered_update_executes(self):
        result = GraphAggregationOperator("spectral_filtered_update").run(self._context())

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output.aggregation_target, "spectral_filtered_update")
        self.assertEqual(result.output.global_update.shape, (2,))

    def test_weight_and_spectral_filtered_weight_execute_when_weights_are_available(self):
        config = {
            "local_weights": np.array(
                [
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [3.0, 2.0],
                ],
                dtype=np.float64,
            )
        }

        weight = GraphAggregationOperator("weight").run(self._context(config=config))
        filtered = GraphAggregationOperator("spectral_filtered_weight").run(self._context(config=config))

        self.assertEqual(weight.output.aggregation_target, "weight")
        self.assertEqual(filtered.output.aggregation_target, "spectral_filtered_weight")
        self.assertIsNotNone(weight.output.global_weights)
        self.assertIsNotNone(filtered.output.global_weights)

    def test_graphfree_controls_execute(self):
        for target in [
            "graphfree_norm_clip",
            "graphfree_contribution_cap",
            "graphfree_dominance_reweight",
        ]:
            with self.subTest(target=target):
                result = GraphAggregationOperator(target).run(self._context())
                self.assertEqual(result.status, "ok")
                self.assertIsNotNone(result.output.alpha)
                self.assertAlmostEqual(float(np.sum(result.output.alpha)), 1.0, places=6)

    def test_alpha_matrix_is_distinct_from_global_alpha(self):
        result = GraphAggregationOperator("directed_neighbor_weight").run(self._context())

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.support_level, "proxy-supported")
        self.assertIsNotNone(result.output.alpha)
        self.assertIsNotNone(result.output.alpha_matrix)
        self.assertEqual(result.output.alpha_matrix.shape, (3, 3))

    def test_interface_target_fails_loudly(self):
        result = GraphAggregationOperator("personalized_weight").run(self._context())

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.support_level, "interface-target")
        self.assertEqual(result.trace_records[0].values["status"], "unsupported")

    def test_cluster_wise_update_is_proxy_supported(self):
        result = GraphAggregationOperator("cluster_wise_update").run(
            self._context(cluster_ids=np.array([0, 0, 1], dtype=np.int64))
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.support_level, "proxy-supported")
        self.assertTrue(np.array_equal(result.output.cluster_ids, np.array([0, 0, 1])))


if __name__ == "__main__":
    unittest.main()
