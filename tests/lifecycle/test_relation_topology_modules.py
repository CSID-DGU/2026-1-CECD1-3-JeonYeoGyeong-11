import unittest

import numpy as np

from graphfl_lab.lifecycle.client_state import ClientStateOutput, ClientStatePayload
from graphfl_lab.lifecycle.context import RelationContext, RoundContext, TopologyContext
from graphfl_lab.lifecycle.relation import (
    GraphRelationEstimator,
    UnsupportedRelationEstimator,
    estimate_relation_from_vectors,
)
from graphfl_lab.lifecycle.topology import (
    ClusterBlockTopologyOperator,
    GraphTopologyOperator,
    UnsupportedTopologyOperator,
    build_topology_from_relation,
)


class RelationTopologyModuleTest(unittest.TestCase):
    def _state_output(self):
        return ClientStateOutput(
            state_kind="updates",
            payload=ClientStatePayload(
                vectors=(
                    np.array([1.0, 0.0]),
                    np.array([0.9, 0.1]),
                    np.array([0.0, 1.0]),
                )
            ),
            source_used="unit_test",
        )

    def test_relation_and_topology_modules_emit_separate_outputs_and_traces(self):
        round_context = RoundContext(server_round=4, cids=["a", "b", "c"])
        relation_result = GraphRelationEstimator("positive_cosine").run(
            RelationContext(round_context=round_context, client_state_output=self._state_output())
        )
        topology_result = GraphTopologyOperator("knn", knn_k=1).run(
            TopologyContext(round_context=round_context, relation_output=relation_result.output)
        )

        self.assertEqual(relation_result.status, "ok")
        self.assertEqual(relation_result.output.relation_kind, "cosine")
        self.assertEqual(relation_result.trace_records[0].phase, "relation")
        self.assertEqual(topology_result.status, "ok")
        self.assertEqual(topology_result.output.graph_kind, "knn")
        self.assertEqual(topology_result.trace_records[0].phase, "topology")
        self.assertTrue(np.allclose(topology_result.output.adjacency, topology_result.output.adjacency.T))

    def test_rbf_relation_and_dense_topology_prefers_nearby_clients(self):
        relation = estimate_relation_from_vectors(
            np.array([[0.0], [0.1], [3.0]], dtype=np.float64),
            relation_kind="rbf",
            graph_scale_sigma=1.0,
        )
        topology = build_topology_from_relation(relation, mode="rbf")

        self.assertEqual(relation.relation_kind, "rbf")
        self.assertGreater(float(topology.adjacency[0, 1]), float(topology.adjacency[0, 2]))
        self.assertEqual(topology.graph_kind, "dense")

    def test_pfedgraph_relation_records_sample_prior_and_topology_projection(self):
        relation = estimate_relation_from_vectors(
            np.array(
                [
                    [1.0, 0.0],
                    [0.95, 0.05],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            relation_kind="pfedgraph_qp",
            client_sample_weights=[0.7, 0.1, 0.1, 0.1],
        )
        topology = build_topology_from_relation(relation, mode="pfedgraph_qp")

        self.assertTrue(relation.is_directed)
        self.assertTrue(relation.relation_meta["prior_used"])
        self.assertEqual(topology.graph_kind, "pfedgraph_qp:symmetric_diagnostic_projection")
        self.assertTrue(np.allclose(topology.adjacency, topology.adjacency.T))

    def test_cluster_block_topology_is_proxy_supported(self):
        round_context = RoundContext(server_round=1, cids=["a", "b", "c"])
        relation = estimate_relation_from_vectors(np.eye(3), relation_kind="cosine")

        result = ClusterBlockTopologyOperator([0, 0, 1], intra=0.5).run(
            TopologyContext(round_context=round_context, relation_output=relation)
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.support_level, "proxy-supported")
        self.assertGreater(float(result.output.adjacency[0, 1]), float(result.output.adjacency[0, 2]))

    def test_unsupported_relation_and_topology_are_explicit(self):
        round_context = RoundContext(server_round=1, cids=["a"])
        relation_context = RelationContext(round_context=round_context, client_state_output=self._state_output())
        unsupported_relation = UnsupportedRelationEstimator("learned_attention").run(relation_context)

        relation = estimate_relation_from_vectors(np.eye(1), relation_kind="cosine")
        unsupported_topology = UnsupportedTopologyOperator("mask_aware_graph").run(
            TopologyContext(round_context=round_context, relation_output=relation)
        )

        self.assertEqual(unsupported_relation.status, "unsupported")
        self.assertEqual(unsupported_relation.support_level, "interface-target")
        self.assertEqual(unsupported_topology.status, "unsupported")
        self.assertEqual(unsupported_topology.trace_records[0].values["component_kind"], "TopologyOperator")


if __name__ == "__main__":
    unittest.main()
