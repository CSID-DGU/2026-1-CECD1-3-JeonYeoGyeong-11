import json
import unittest

from spectral_fl.lifecycle.traces import TRACE_SCHEMA_VERSION
from spectral_fl.strategies.graphfl.tracing import make_round_trace_payload


class SpectralTracingTest(unittest.TestCase):
    def test_round_trace_payload_contains_lifecycle_schema_marker(self):
        payload = make_round_trace_payload(
            correction_family="real_graph",
            control_graph_mode="random",
            graph_mode="knn",
            alpha_mode="softmax",
            pre_post_round={
                "round": 2,
                "di_pre": 0.1,
                "di_post": 0.2,
                "neff_pre": 3.0,
                "neff_post": 4.0,
                "align_mean_pre": 0.5,
                "align_mean_post": 0.6,
                "loo_mean_pre": 0.7,
                "loo_mean_post": 0.8,
            },
        )

        self.assertEqual(payload["trace_schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(payload["di_pre"], 0.1)
        self.assertEqual(len(payload["lifecycle_trace"]), 1)
        record = payload["lifecycle_trace"][0]
        self.assertEqual(record["phase"], "diagnostic")
        self.assertEqual(record["module"], "pre_post")
        self.assertEqual(record["round"], 2)
        self.assertEqual(record["values"]["neff_post"], 4.0)
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
