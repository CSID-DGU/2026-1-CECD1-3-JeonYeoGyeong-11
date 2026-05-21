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

        report = module.run_gate_check("5a", ROOT)

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

    def test_gate3_fails_without_real_move_files(self):
        module = load_run_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

            report = module.run_gate_check("3", root)

        self.assertFalse(report["pass"])
        self.assertTrue(any("graphfl_lab" in item for item in report["failed_checks"]))

    def test_gate3a_reports_superseded(self):
        module = load_run_module()

        report = module.run_gate_check("3a", ROOT)

        self.assertFalse(report["pass"])
        self.assertIn("superseded", report["failed_checks"][0])

    def test_gate3b_fails_on_legacy_import_identity(self):
        module = load_run_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            package = root / "graphfl_lab"
            package.mkdir()
            (package / "__init__.py").write_text(
                "Canonical package root\n",
                encoding="utf-8",
            )
            legacy = root / "spectral_fl"
            legacy.mkdir()
            (legacy / "__init__.py").write_text(
                "DeprecationWarning\nGRAPHFL_LAB_SILENCE_DEPRECATION\ngraphfl_lab\n",
                encoding="utf-8",
            )
            (root / "tests" / "core").mkdir(parents=True)
            (root / "tests" / "core" / "test_package_alias.py").write_text(
                "\n".join(
                    [
                        "test_graphfl_lab_imports_flower_app",
                        "test_spectral_fl_warns_by_default",
                        "test_spectral_fl_warning_can_be_silenced",
                        "test_sys_modules_alias_roots_exist",
                        "test_pickle_round_trip_for_canonical_import",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "bad.py").write_text("from spectral_fl.foo import bar\n", encoding="utf-8")

            original_tracked_files = module._tracked_files
            module._tracked_files = lambda _root: [
                "graphfl_lab/__init__.py",
                "spectral_fl/__init__.py",
                "tests/core/test_package_alias.py",
                "bad.py",
            ]
            try:
                report = module.run_gate_check("3b", root)
            finally:
                module._tracked_files = original_tracked_files

        self.assertFalse(report["pass"])
        self.assertTrue(any("forbidden legacy import token" in item for item in report["failed_checks"]))

    def test_current_gate3b_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("3b", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "3b")

    def test_current_gate3_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("3", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "3")

    def test_current_gate4a_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("4a", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "4a")

    def test_current_gate4b_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("4b", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "4b")

    def test_gate4c_prep_exists_but_requires_remote_green(self):
        module = load_run_module()

        report = module.run_gate_check("4c", ROOT)

        self.assertFalse(report["pass"])
        self.assertTrue(any("manual-nightly green" in item for item in report["failed_checks"]))

    def test_current_gate5a_prep_contract_passes(self):
        module = load_run_module()

        report = module.run_gate_check("5a-prep", ROOT)

        self.assertTrue(report["pass"], report["failed_checks"])
        self.assertEqual(report["gate"], "5a-prep")


if __name__ == "__main__":
    unittest.main()
