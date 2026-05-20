import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = ROOT / "scripts" / "dev" / "run.py"


def load_run_module():
    spec = importlib.util.spec_from_file_location("dev_run", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GateCheckEntrypointTest(unittest.TestCase):
    def test_gate0_fails_when_required_files_are_missing(self):
        module = load_run_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

            report = module.run_gate_check("0", root)

        self.assertFalse(report["pass"])
        self.assertTrue(report["failed_checks"])
        self.assertIn("gate", report)
        self.assertIn("verified_at", report)
        self.assertIn("commit_sha", report)

    def test_current_gate0_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("0", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "0")

    def test_future_gate_checks_fail_closed_until_implemented(self):
        module = load_run_module()

        report = module.run_gate_check("4a", ROOT)

        self.assertFalse(report["pass"])
        self.assertIn("not implemented yet", report["failed_checks"][0])

    def test_gate1_fails_without_inventory_and_tag(self):
        module = load_run_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

            report = module.run_gate_check("1", root)

        self.assertFalse(report["pass"])
        self.assertTrue(any("rename-inventory" in item for item in report["failed_checks"]))
        self.assertTrue(any("pre-graphfl-rename" in item for item in report["failed_checks"]))

    def test_current_gate1_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("1", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "1")

    def test_gate2_fails_without_schema_files(self):
        module = load_run_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

            report = module.run_gate_check("2", root)

        self.assertFalse(report["pass"])
        self.assertTrue(any("result_schema.py" in item for item in report["failed_checks"]))

    def test_current_gate2_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("2", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "2")

    def test_gate3a_fails_without_alias_files(self):
        module = load_run_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

            report = module.run_gate_check("3a", root)

        self.assertFalse(report["pass"])
        self.assertTrue(any("graphfl_lab" in item for item in report["failed_checks"]))

    def test_current_gate3a_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("3a", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "3a")

    def test_full_gate3_fails_closed_until_import_batches_complete(self):
        module = load_run_module()

        report = module.run_gate_check("3", ROOT)

        self.assertFalse(report["pass"])
        self.assertIn("not complete", report["failed_checks"][0])


if __name__ == "__main__":
    unittest.main()
