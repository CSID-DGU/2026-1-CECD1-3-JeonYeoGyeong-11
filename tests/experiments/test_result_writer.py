import csv
import json
import tempfile
import unittest
from pathlib import Path

from graphfl_lab.experiments.suites.result_writer import write_csv_rows, write_json


class ResultWriterTest(unittest.TestCase):
    def test_write_json_uses_indented_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"

            returned = write_json(path, {"b": 2, "a": 1})

            self.assertEqual(returned, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": 2})
            self.assertIn("\n  ", path.read_text(encoding="utf-8"))

    def test_write_csv_rows_preserves_field_order(self):
        rows = [{"b": 2, "a": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"

            returned = write_csv_rows(path, rows, fieldnames=["a", "b"])

            self.assertEqual(returned, path)
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["a", "b"])
                self.assertEqual(list(reader), [{"a": "1", "b": "2"}])


if __name__ == "__main__":
    unittest.main()
