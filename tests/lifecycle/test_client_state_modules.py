import unittest

import numpy as np

from spectral_fl.lifecycle.client_state import (
    ClientStateOutput,
    ClientStatePayload,
    GraphSourceClientStateExtractor,
    UnsupportedClientStateExtractor,
)
from spectral_fl.lifecycle.context import make_state_extraction_context


class ClientStateModuleTest(unittest.TestCase):
    def test_graph_source_extractor_returns_vector_envelope_and_trace(self):
        context = make_state_extraction_context(
            server_round=2,
            cids=["c0", "c1"],
            global_weights=[np.array([0.0, 0.0])],
            local_weights=[
                [np.array([1.0, 1.0])],
                [np.array([2.0, 1.0])],
            ],
            local_updates=[
                [np.array([1.0, 0.0])],
                [np.array([0.0, 2.0])],
            ],
            num_examples=[5, 7],
            client_metrics=[{"loss": 0.3}, {"loss": 0.4}],
        )

        result = GraphSourceClientStateExtractor("update").run(context)

        self.assertEqual(result.status, "ok")
        self.assertIsInstance(result.output, ClientStateOutput)
        self.assertEqual(result.output.state_kind, "updates")
        self.assertEqual(result.output.source_used, "update_delta")
        self.assertTrue(np.allclose(result.output.vector_matrix(), np.array([[1.0, 0.0], [0.0, 2.0]])))
        self.assertEqual(result.output.per_client_meta[1]["num_examples"], 7)
        self.assertEqual(result.trace_records[0].phase, "client_state")
        self.assertEqual(result.trace_records[0].values["source_used"], "update_delta")

    def test_state_output_can_represent_non_flat_payloads(self):
        output = ClientStateOutput(
            state_kind="hybrid",
            payload=ClientStatePayload(
                tensors=({"head": np.array([1.0])},),
                scalar_features=np.array([[0.5]]),
            ),
            source_used="unit_test",
        )

        self.assertEqual(output.state_kind, "hybrid")
        with self.assertRaisesRegex(ValueError, "does not expose flat vectors"):
            output.vector_matrix()

    def test_unsupported_state_extractor_is_explicit(self):
        context = make_state_extraction_context(
            server_round=1,
            cids=["c0"],
            global_weights=[np.array([0.0])],
            local_weights=[[np.array([1.0])]],
            local_updates=[[np.array([1.0])]],
            num_examples=[1],
        )

        result = UnsupportedClientStateExtractor("functional_embedding").run(context)

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.support_level, "interface-target")
        self.assertEqual(result.trace_records[0].values["component_kind"], "ClientStateExtractor")


if __name__ == "__main__":
    unittest.main()
