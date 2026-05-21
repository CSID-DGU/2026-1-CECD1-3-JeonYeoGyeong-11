import unittest

import numpy as np

from graphfl_lab.graph.sources import (
    GraphSourceConfig,
    GraphSourceContext,
    GraphSourceResult,
    graph_vectors_for_spectral,
    register_graph_source,
    unregister_graph_source,
)


class GraphSourceRegistryTest(unittest.TestCase):
    def tearDown(self):
        unregister_graph_source("unit_test_update_norms")

    def test_registered_source_can_supply_custom_client_representations(self):
        @register_graph_source("unit_test_update_norms", override=True)
        def _update_norms(context: GraphSourceContext):
            vectors = [
                np.array([float(np.linalg.norm(arrays[0]))], dtype=np.float64)
                for arrays in context.local_updates
            ]
            return GraphSourceResult(
                vectors=vectors,
                source_used="unit_test_update_norms",
                metadata={"source_family": "test"},
            )

        local_updates = [
            [np.array([3.0, 4.0], dtype=np.float32)],
            [np.array([0.0, 2.0], dtype=np.float32)],
        ]
        vectors, source_used = graph_vectors_for_spectral(
            local_weights=local_updates,
            local_updates=local_updates,
            config=GraphSourceConfig(source="unit_test_update_norms"),
        )

        self.assertEqual(source_used, "unit_test_update_norms")
        self.assertTrue(np.allclose(vectors[0], np.array([5.0])))
        self.assertTrue(np.allclose(vectors[1], np.array([2.0])))


if __name__ == "__main__":
    unittest.main()
