import json
import math
import unittest

import numpy as np

from spectral_fl.lifecycle.traces import (
    TRACE_SCHEMA_VERSION,
    RoundTraceBundle,
    TraceRecord,
    json_safe,
)


class TraceRecordTest(unittest.TestCase):
    def test_trace_record_is_strict_json_serializable(self):
        record = TraceRecord(
            phase="relation",
            module="cosine",
            name="raw_similarity",
            round=3,
            values={
                "scale": np.float32(0.5),
                "matrix": np.array([[1.0, 2.0], [np.nan, np.inf]]),
                "labels": {"b", "a"},
                "nested": {"losses": [1.0, math.inf]},
            },
        )

        payload = record.to_dict()

        self.assertEqual(payload["schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(payload["values"]["scale"], 0.5)
        self.assertEqual(payload["values"]["matrix"]["shape"], [2, 2])
        self.assertEqual(payload["values"]["matrix"]["values"][1], [None, None])
        self.assertEqual(payload["values"]["labels"], ["a", "b"])
        self.assertIsNone(payload["values"]["nested"]["losses"][1])
        json.dumps(payload, allow_nan=False)

    def test_invalid_labels_and_round_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            TraceRecord(phase="", module="m", name="n")
        with self.assertRaisesRegex(ValueError, "round"):
            TraceRecord(phase="relation", module="m", name="n", round=-1)

    def test_design_space_metadata_stays_inside_values(self):
        record = TraceRecord(
            phase="topology",
            module="knn",
            name="graph",
            values={
                "variant": "actual",
                "support_level": "core-supported",
                "status": "ok",
                "design_name": "head_knn_filtered_update",
                "component_kind": "TopologyOperator",
                "component_name": "knn",
                "input_kind": ("cosine_relation",),
                "output_kind": "adjacency",
                "is_learned": False,
                "is_stateful": False,
                "is_directed": False,
                "is_symmetric": True,
                "is_weighted": True,
                "is_dynamic": False,
                "is_layerwise": False,
                "diagnostics": {"graph_density": np.float64(0.5)},
            },
        )

        payload = record.to_dict()

        self.assertNotIn("design_name", payload)
        self.assertEqual(payload["values"]["design_name"], "head_knn_filtered_update")
        self.assertEqual(payload["values"]["input_kind"], ["cosine_relation"])
        self.assertEqual(payload["values"]["diagnostics"]["graph_density"], 0.5)
        json.dumps(payload, allow_nan=False)


class RoundTraceBundleTest(unittest.TestCase):
    def test_bundle_filters_and_flattens_records(self):
        bundle = RoundTraceBundle()
        relation = TraceRecord(
            phase="relation",
            module="cosine",
            name="raw_similarity",
            round=1,
            values={"edge_mean": 0.25},
        )
        topology = TraceRecord(
            phase="topology",
            module="knn",
            name="adjacency",
            round=1,
            values={"density": np.float64(0.5)},
        )

        self.assertIs(bundle.add(relation), relation)
        bundle.extend([topology])

        self.assertEqual(bundle.by_phase("relation"), [relation])
        flat = bundle.to_flat_dict()
        self.assertEqual(flat["trace_schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(flat["trace_record_count"], 2)
        self.assertEqual(flat["trace.0.relation.cosine.raw_similarity.edge_mean"], 0.25)
        self.assertEqual(flat["trace.1.topology.knn.adjacency.density"], 0.5)
        json.dumps(bundle.to_dicts(), allow_nan=False)

    def test_bundle_rejects_non_trace_record(self):
        bundle = RoundTraceBundle()
        with self.assertRaisesRegex(TypeError, "TraceRecord"):
            bundle.add(object())


class JsonSafeTest(unittest.TestCase):
    def test_large_array_uses_summary_without_values(self):
        payload = json_safe(np.arange(20, dtype=np.float64), max_array_values=4)

        self.assertEqual(payload["type"], "ndarray")
        self.assertEqual(payload["shape"], [20])
        self.assertNotIn("values", payload)
        self.assertEqual(payload["min"], 0.0)
        self.assertEqual(payload["max"], 19.0)
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
