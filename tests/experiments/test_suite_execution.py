import tempfile
import unittest
from pathlib import Path
from unittest import mock

from graphfl_lab.experiments.suites.execution import execute_or_reuse_result, run_cmd


class SuiteExecutionTest(unittest.TestCase):
    def test_execute_reuses_existing_result_without_running_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            result_path.write_text("{}", encoding="utf-8")

            with mock.patch("graphfl_lab.experiments.suites.execution.run_cmd") as runner:
                reused, elapsed = execute_or_reuse_result(
                    ["cmd"],
                    result_path,
                    True,
                    cwd=root,
                )

        self.assertTrue(reused)
        self.assertIsNone(elapsed)
        runner.assert_not_called()

    def test_execute_runs_when_reuse_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("graphfl_lab.experiments.suites.execution.run_cmd") as runner:
                reused, elapsed = execute_or_reuse_result(
                    ["cmd"],
                    root / "missing.json",
                    False,
                    cwd=root,
                )

        self.assertFalse(reused)
        self.assertIsInstance(elapsed, float)
        runner.assert_called_once_with(["cmd"], cwd=root)

    def test_run_cmd_uses_cwd_and_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("subprocess.run") as proc:
                run_cmd(["cmd", "arg"], cwd=root)

        proc.assert_called_once_with(["cmd", "arg"], cwd=str(root), check=True)


if __name__ == "__main__":
    unittest.main()
