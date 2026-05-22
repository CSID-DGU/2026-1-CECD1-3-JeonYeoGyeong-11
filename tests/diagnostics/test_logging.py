import csv
import json
import tempfile
import unittest
from pathlib import Path

from graphfl_lab.diagnostics.logging import (
    append_counterfactual_metrics_csv,
    append_module_traces_jsonl,
)
from graphfl_lab.lifecycle.traces import TraceRecord


class DiagnosticsLoggingTest(unittest.TestCase):
    def test_counterfactual_metrics_writer_appends_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "counterfactual_metrics.csv"

            append_counterfactual_metrics_csv(
                path,
                [
                    {
                        "run_id": "run",
                        "round": 1,
                        "counterfactual": "actual",
                        "status": "ok",
                    },
                    {
                        "run_id": "run",
                        "round": 1,
                        "counterfactual": "matched_random",
                        "status": "ok",
                    },
                ],
            )

            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(
                [row["counterfactual"] for row in rows],
                ["actual", "matched_random"],
            )

    def test_csv_writer_expands_schema_when_later_rows_have_more_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "counterfactual_metrics.csv"

            append_counterfactual_metrics_csv(
                path,
                [
                    {
                        "run_id": "run",
                        "round": 1,
                        "counterfactual": "actual",
                        "status": "error",
                        "error_message": "not supported yet",
                    }
                ],
            )
            append_counterfactual_metrics_csv(
                path,
                [
                    {
                        "run_id": "run",
                        "round": 1,
                        "counterfactual": "graphfree_dominance_reweight",
                        "status": "ok",
                        "di_pre": 0.5,
                        "graph_density": 0.0,
                    }
                ],
            )

            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["error_message"], "not supported yet")
            self.assertEqual(rows[1]["di_pre"], "0.5")
            self.assertIn("graph_density", rows[1])

    def test_module_trace_writer_preserves_trace_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "module_traces.jsonl"
            trace = TraceRecord(
                phase="counterfactual",
                module="diagnostic_runner",
                name="actual",
                round=2,
                values={"status": "ok"},
            )

            append_module_traces_jsonl(path, [trace])

            with path.open(encoding="utf-8") as f:
                payload = json.loads(f.readline())
            self.assertEqual(payload["phase"], "counterfactual")
            self.assertEqual(payload["module"], "diagnostic_runner")
            self.assertEqual(payload["values"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
