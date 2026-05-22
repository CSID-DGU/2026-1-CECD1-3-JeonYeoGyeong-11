import json
import unittest
from pathlib import Path

from graphfl_lab.experiments.suites.vision.artifacts import (
    discover_result_json_paths,
    load_suite_rows_json,
    resolve_suite_artifact,
    SUITE_ROWS_JSON_FILENAMES,
    SUITE_SUMMARY_CSV_FILENAMES,
)


class SuiteArtifactsTest(unittest.TestCase):
    def test_resolve_suite_artifact_prefers_canonical_filename(self):
        with self.subTest("rows_json"):
            out_dir = Path(self.id().replace(".", "_"))
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                (out_dir / "general_suite_rows.json").write_text("[]", encoding="utf-8")
                (out_dir / "vision_suite_rows.json").write_text(
                    json.dumps([{"variant": "ours_knn_k1"}]),
                    encoding="utf-8",
                )
                path = resolve_suite_artifact(out_dir, SUITE_ROWS_JSON_FILENAMES)
                self.assertEqual(path.name, "vision_suite_rows.json")
            finally:
                for child in out_dir.iterdir():
                    child.unlink()
                out_dir.rmdir()

        with self.subTest("summary_csv"):
            out_dir = Path(self.id().replace(".", "_") + "_csv")
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                (out_dir / "general_suite_summary.csv").write_text("variant\nfedavg\n", encoding="utf-8")
                (out_dir / "vision_suite_summary.csv").write_text("variant\nours\n", encoding="utf-8")
                path = resolve_suite_artifact(out_dir, SUITE_SUMMARY_CSV_FILENAMES)
                self.assertEqual(path.name, "vision_suite_summary.csv")
            finally:
                for child in out_dir.iterdir():
                    child.unlink()
                out_dir.rmdir()

    def test_discover_result_json_paths_prefers_vision_alias(self):
        out_dir = Path(self.id().replace(".", "_") + "_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            general = out_dir / "result_general_ours_seed1_tag.json"
            vision = out_dir / "result_vision_ours_seed1_tag.json"
            general.write_text('{"meta": {"seed": 1}}', encoding="utf-8")
            vision.write_text('{"meta": {"seed": 1, "canonical": true}}', encoding="utf-8")
            paths = discover_result_json_paths(out_dir)
            self.assertEqual(paths["ours_seed1_tag.json"], vision)
        finally:
            for child in out_dir.iterdir():
                child.unlink()
            out_dir.rmdir()

    def test_load_suite_rows_json_reads_canonical_rows(self):
        out_dir = Path(self.id().replace(".", "_") + "_rows")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            (out_dir / "vision_suite_rows.json").write_text(
                json.dumps([{"variant": "fedavg"}]),
                encoding="utf-8",
            )
            rows = load_suite_rows_json(out_dir)
            self.assertEqual(rows, [{"variant": "fedavg"}])
        finally:
            for child in out_dir.iterdir():
                child.unlink()
            out_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
