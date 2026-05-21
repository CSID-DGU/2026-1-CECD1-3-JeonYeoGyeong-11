import sys
import unittest
import warnings
from unittest import mock

from graphfl_lab.cli import experiment_dispatcher


class ExperimentDispatcherTest(unittest.TestCase):
    def test_missing_track_defaults_to_cora_with_deprecation_warning(self):
        calls = []

        def fake_main():
            calls.append(("cora", sys.argv[1:]))

        with mock.patch.object(experiment_dispatcher.cora_experiment, "main", fake_main):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DeprecationWarning)
                experiment_dispatcher.main(["--rounds", "1"])

        self.assertEqual(calls, [("cora", ["--rounds", "1"])])
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_track_cora_dispatches_without_track_argument(self):
        calls = []

        def fake_main():
            calls.append(("cora", sys.argv[1:]))

        with mock.patch.object(experiment_dispatcher.cora_experiment, "main", fake_main):
            experiment_dispatcher.main(["--track", "cora", "--rounds", "1"])

        self.assertEqual(calls, [("cora", ["--rounds", "1"])])

    def test_track_vision_dispatches_without_track_argument(self):
        calls = []

        def fake_main():
            calls.append(("vision", sys.argv[1:]))

        with mock.patch.object(experiment_dispatcher.vision_experiment, "main", fake_main):
            experiment_dispatcher.main(["--track", "vision", "--dataset", "mnist"])

        self.assertEqual(calls, [("vision", ["--dataset", "mnist"])])

    def test_sys_argv_is_restored_after_dispatch(self):
        before = list(sys.argv)

        with mock.patch.object(experiment_dispatcher.cora_experiment, "main", lambda: None):
            experiment_dispatcher.main(["--track", "cora"])

        self.assertEqual(sys.argv, before)


if __name__ == "__main__":
    unittest.main()
