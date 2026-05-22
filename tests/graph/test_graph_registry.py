import unittest

import numpy as np

from graphfl_lab.graph.builders import build_relation_graph
from graphfl_lab.graph.registry import (
    GraphBuildContext,
    register_graph_builder,
    require_graph_context,
    unregister_graph_builder,
)


class GraphRegistryTest(unittest.TestCase):
    def tearDown(self):
        unregister_graph_builder("unit_test_star")
        unregister_graph_builder("unit_test_source_bound")

    def test_registered_builder_can_be_used_as_graph_mode(self):
        @register_graph_builder("unit_test_star", override=True)
        def _star_graph(context: GraphBuildContext):
            n = context.z_mat.shape[0]
            adj = np.zeros((n, n), dtype=np.float64)
            adj[0, 1:] = float(context.graph_scale_sigma)
            adj[1:, 0] = float(context.graph_scale_sigma)
            return adj, {"graph_kind": "plugin:star"}

        z_mat = np.eye(4, dtype=np.float64)
        adj, meta = build_relation_graph(
            z_mat=z_mat,
            mode="unit_test_star",
            graph_scale_sigma=0.25,
            graph_source="classifier_head_update",
            aggregation_target="graph_filtered_update",
        )

        self.assertTrue(np.allclose(adj[0, 1:], 0.25))
        self.assertTrue(np.allclose(adj[1:, 0], 0.25))
        self.assertEqual(meta["base_graph_builder"], "registered")
        self.assertEqual(meta["base_graph_kind"], "plugin:star")
        self.assertEqual(meta["graph_kind"], "real_graph")
        self.assertEqual(meta["graph_source"], "classifier_head_update")
        self.assertEqual(meta["aggregation_target"], "graph_filtered_update")

    def test_registered_builder_can_restrict_source_and_target(self):
        @register_graph_builder("unit_test_source_bound", override=True)
        def _source_bound_graph(context: GraphBuildContext):
            require_graph_context(
                context,
                graph_sources=("classifier_head_update",),
                aggregation_targets=("spectral_filtered_update",),
            )
            return np.ones((context.z_mat.shape[0], context.z_mat.shape[0]))

        z_mat = np.eye(3, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "requires graph_source"):
            build_relation_graph(
                z_mat=z_mat,
                mode="unit_test_source_bound",
                graph_source="weight",
                aggregation_target="graph_filtered_update",
            )


if __name__ == "__main__":
    unittest.main()
