import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "scripts" / "dev" / "golden.py"


def load_golden_module():
    spec = importlib.util.spec_from_file_location("dev_golden", GOLDEN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def payload(**overrides):
    data = {
        "result_schema_version": "1",
        "config_aliases_used": [],
        "unsupported_components": [],
        "run_id": "a",
        "timestamp": "2026-01-01T00:00:00Z",
        "summary": [{"variant": "fedavg", "accuracy": 0.5}],
    }
    data.update(overrides)
    return data


class GoldenComparisonTest(unittest.TestCase):
    def test_normalized_compare_ignores_volatile_fields(self):
        golden = load_golden_module()
        expected = payload(run_id="a", timestamp="one")
        actual = payload(run_id="b", timestamp="two")

        self.assertEqual(golden.compare_payloads(expected, actual), [])

    def test_compare_fails_on_missing_required_schema_key(self):
        golden = load_golden_module()
        actual = payload()
        actual.pop("unsupported_components")

        failures = golden.compare_payloads(payload(), actual)

        self.assertTrue(any("unsupported_components" in item for item in failures))

    def test_compare_fails_on_schema_shape_change(self):
        golden = load_golden_module()
        actual = payload(extra_field=True)

        failures = golden.compare_payloads(payload(), actual)

        self.assertIn("schema shape differs", failures)

    def test_compare_fails_on_normalized_value_change(self):
        golden = load_golden_module()
        actual = payload(summary=[{"variant": "fedavg", "accuracy": 0.7}])

        failures = golden.compare_payloads(payload(), actual)

        self.assertIn("normalized payload differs", failures)

    def test_compare_files(self):
        golden = load_golden_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "expected.json"
            actual = root / "actual.json"
            expected.write_text(
                '{"result_schema_version":"1","config_aliases_used":[],"unsupported_components":[],"run_id":"a"}',
                encoding="utf-8",
            )
            actual.write_text(
                '{"result_schema_version":"1","config_aliases_used":[],"unsupported_components":[],"run_id":"b"}',
                encoding="utf-8",
            )

            self.assertEqual(golden.compare_files(expected, actual), [])


if __name__ == "__main__":
    unittest.main()
