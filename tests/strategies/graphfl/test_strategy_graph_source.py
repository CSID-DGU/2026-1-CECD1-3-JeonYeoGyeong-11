import unittest
from types import SimpleNamespace

import numpy as np

from graphfl_lab.strategies.baselines import sort_fit_results_by_cid
from graphfl_lab.strategies.graphfl.strategy import GraphFLDiagnosticStrategy


class StrategyGraphSourceTest(unittest.TestCase):
    def test_fit_results_are_sorted_by_numeric_client_id(self):
        results = [
            (SimpleNamespace(cid="proxy-ignored"), SimpleNamespace(metrics={"cid": 10})),
            (SimpleNamespace(cid="2"), SimpleNamespace(metrics={})),
            (SimpleNamespace(cid="proxy-ignored"), SimpleNamespace(metrics={"cid": 1})),
        ]

        ordered = sort_fit_results_by_cid(results)

        got = [
            str(fit_res.metrics.get("cid", getattr(proxy, "cid", "?")))
            for proxy, fit_res in ordered
        ]
        self.assertEqual(got, ["1", "2", "10"])

    def test_layer_slice_update_uses_tail_tensors(self):
        strategy = GraphFLDiagnosticStrategy(
            graph_source="layer_slice_update",
            graph_layer_start=-2,
        )
        local_updates = [
            [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
                np.array([4.0, 5.0], dtype=np.float32),
                np.array([6.0], dtype=np.float32),
            ]
        ]

        vectors, source_used = strategy._graph_vectors(
            local_weights=local_updates,
            local_updates=local_updates,
        )

        self.assertEqual(source_used, "update_delta_layers_2:4")
        self.assertTrue(np.allclose(vectors[0], np.array([4.0, 5.0, 6.0])))

    def test_layerwise_slice_update_normalizes_selected_tensors(self):
        strategy = GraphFLDiagnosticStrategy(
            graph_source="layerwise_slice_update",
            graph_layer_start=-2,
        )
        local_updates = [
            [
                np.array([10.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
                np.array([5.0], dtype=np.float32),
            ]
        ]

        vectors, source_used = strategy._graph_vectors(
            local_weights=local_updates,
            local_updates=local_updates,
        )

        self.assertEqual(source_used, "layerwise_normalized_update_delta_layers_1:3")
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0, places=6)

    def test_classifier_head_update_uses_final_weight_bias_pair(self):
        strategy = GraphFLDiagnosticStrategy(graph_source="classifier_head_update")
        local_updates = [
            [
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
                np.array([5.0], dtype=np.float32),
            ]
        ]

        vectors, source_used = strategy._graph_vectors(
            local_weights=local_updates,
            local_updates=local_updates,
        )

        self.assertEqual(source_used, "classifier_head_update_delta_layers_2:4")
        self.assertTrue(np.allclose(vectors[0], np.array([3.0, 4.0, 5.0])))

    def test_ema_update_graph_source_uses_smoothed_signal(self):
        strategy = GraphFLDiagnosticStrategy(graph_source="ema_update")
        current = [[np.array([1.0, 2.0], dtype=np.float32)]]
        ema = [[np.array([10.0, 20.0], dtype=np.float32)]]

        vectors, source_used = strategy._graph_vectors(
            local_weights=current,
            local_updates=current,
            ema_updates=ema,
        )

        self.assertEqual(source_used, "client_ema_update_delta")
        self.assertTrue(np.allclose(vectors[0], np.array([10.0, 20.0])))

    def test_normalized_ema_update_graph_source_uses_smoothed_direction(self):
        strategy = GraphFLDiagnosticStrategy(graph_source="normalized_ema_update")
        current = [[np.array([1.0, 2.0], dtype=np.float32)]]
        ema = [[np.array([3.0, 4.0], dtype=np.float32)]]

        vectors, source_used = strategy._graph_vectors(
            local_weights=current,
            local_updates=current,
            ema_updates=ema,
        )

        self.assertEqual(source_used, "normalized_client_ema_update_delta")
        self.assertTrue(np.allclose(vectors[0], np.array([0.6, 0.8])))


if __name__ == "__main__":
    unittest.main()
