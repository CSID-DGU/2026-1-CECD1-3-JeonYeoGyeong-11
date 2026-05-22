import unittest

import numpy as np

from graphfl_lab.graph.builders import GraphBuildContext, build_relation_graph
from graphfl_lab.graph.registry import register_graph_builder, require_graph_context
from graphfl_lab.graph.sources import graph_vectors_for_graphfl


class GraphSourceRegistryTest(unittest.TestCase):
    def test_graph_vectors_for_graphfl_is_canonical(self):
        self.assertTrue(callable(graph_vectors_for_graphfl))

    def test_registered_builder_can_build_star_graph(self):
        @register_graph_builder("unit_test_star", override=True)
        def _star_graph(context):
            n = context.z_mat.shape[0]
            adj = np.zeros((n, n), dtype=np.float64)
            adj[0, 1:] = 0.25
            adj[1:, 0] = 0.25
            return adj

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
        self.assertEqual(meta["base_graph_mode"], "unit_test_star")
        self.assertEqual(meta["graph_kind"], "real_graph")
        self.assertEqual(meta["graph_source"], "classifier_head_update")
        self.assertEqual(meta["aggregation_target"], "graph_filtered_update")

    def test_registered_builder_can_restrict_source_and_target(self):
        @register_graph_builder("unit_test_source_bound", override=True)
        def _source_bound_graph(context: GraphBuildContext):
            require_graph_context(
                context,
                graph_sources=("classifier_head_update",),
                aggregation_targets=("graph_filtered_update",),
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
