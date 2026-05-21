import math
import unittest
from argparse import Namespace
from datetime import datetime

from graphfl_lab.experiments.suites.vision.metadata import (
    build_suite_meta,
    record_preloaded_fedavg_meta,
    record_suite_timing,
    record_training_data_note,
)


def base_args():
    return Namespace(
        train_subset_size=0,
        test_subset_size=0,
        dataset="mnist",
        _config_aliases_used=(),
    )


class VisionSuiteMetadataTest(unittest.TestCase):
    def test_build_suite_meta_documents_full_diagnostic_set(self):
        meta = build_suite_meta(
            base_args(),
            "suite",
            datetime(2026, 5, 21, 3, 0, 0),
        )

        self.assertEqual(meta["track"], "vision-fl")
        self.assertEqual(meta["suite_tag"], "suite")
        self.assertIn("mean_di_pre/post", meta["trace_aggregate_semantics"])
        self.assertIn("mean_neff_pre/post", meta["trace_aggregate_semantics"])
        self.assertIn("mean_alignment_pre/post", meta["trace_aggregate_semantics"])
        self.assertIn("mean_loo_pre/post", meta["trace_aggregate_semantics"])

    def test_record_training_data_note_tracks_full_and_subset_modes(self):
        meta = {}
        record_training_data_note(meta, base_args())
        self.assertEqual(meta["training_data_note"], "full_dataset_splits")

        args = base_args()
        args.train_subset_size = 10
        args.test_subset_size = 5
        record_training_data_note(meta, args)
        self.assertEqual(meta["training_data_note"], "subset_train_10_test_5")

    def test_record_preloaded_and_timing_metadata(self):
        meta = {}
        record_preloaded_fedavg_meta(meta, "runs/fedavg", {2: 0.5, 1: 0.4})
        self.assertEqual(meta["preloaded_fedavg_dir"], "runs/fedavg")
        self.assertEqual(meta["preloaded_fedavg_seeds"], [1, 2])

        record_suite_timing(
            meta,
            datetime(2026, 5, 21, 3, 0, 0),
            0.0,
            [
                {"run_wall_time_sec": 1.5},
                {"run_wall_time_sec": "bad"},
                {"run_wall_time_sec": float("nan")},
            ],
        )
        self.assertEqual(meta["timing"]["started_at"], "2026-05-21T03:00:00")
        self.assertGreaterEqual(meta["timing"]["suite_wall_time_sec"], 0.0)
        self.assertAlmostEqual(meta["timing"]["sum_recorded_run_wall_time_sec"], 1.5)
        self.assertFalse(math.isnan(meta["timing"]["sum_recorded_run_wall_time_sec"]))


if __name__ == "__main__":
    unittest.main()
