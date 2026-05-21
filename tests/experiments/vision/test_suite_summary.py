import unittest
from argparse import Namespace

from graphfl_lab.experiments.suites.vision.summary import build_summary_rows


def base_args():
    return Namespace(
        dataset="mnist",
        partition="dirichlet",
        dirichlet_alpha=0.3,
        num_clients=4,
        seeds=[1, 2],
    )


class VisionSuiteSummaryTest(unittest.TestCase):
    def test_build_summary_rows_aggregates_all_diagnostic_fields(self):
        rows = [
            {
                "variant": "fedavg",
                "seed": 1,
                "fedavg_acc": 0.50,
                "ours_acc": float("nan"),
                "delta": 0.0,
                "run_wall_time_sec": 10.0,
                "result_method_wall_time_sec": 9.0,
                "seconds_per_round": 1.0,
                "timing_source": "suite_observed",
            },
            {
                "variant": "ours_knn_k2",
                "seed": 1,
                "fedavg_acc": 0.50,
                "ours_acc": 0.60,
                "delta": 0.10,
                "run_wall_time_sec": 12.0,
                "result_method_wall_time_sec": 11.0,
                "seconds_per_round": 1.2,
                "timing_source": "suite_observed",
                "graph_mode": "knn",
                "mean_h_spec": 0.2,
                "mean_di_pre": 0.7,
                "mean_di_post": 0.3,
            },
            {
                "variant": "ours_knn_k2",
                "seed": 2,
                "fedavg_acc": 0.55,
                "ours_acc": 0.57,
                "delta": 0.02,
                "run_wall_time_sec": 14.0,
                "result_method_wall_time_sec": 13.0,
                "seconds_per_round": 1.4,
                "timing_source": "suite_observed",
                "graph_mode": "knn",
                "mean_h_spec": 0.4,
                "mean_di_pre": 0.5,
                "mean_di_post": 0.2,
            },
        ]

        summary = build_summary_rows(rows, base_args())

        self.assertEqual(summary[0]["variant"], "ours_knn_k2")
        self.assertEqual(summary[0]["dataset"], "mnist")
        self.assertEqual(summary[0]["partition"], "dirichlet")
        self.assertEqual(summary[0]["num_clients"], 4)
        self.assertAlmostEqual(summary[0]["mean_acc"], 0.585)
        self.assertAlmostEqual(summary[0]["mean_delta"], 0.06)
        self.assertAlmostEqual(summary[0]["median_delta"], 0.06)
        self.assertEqual(summary[0]["number_of_positive_seeds"], 2)
        self.assertEqual(summary[0]["graph_mode"], "knn")
        self.assertAlmostEqual(summary[0]["mean_H_spec"], 0.3)
        self.assertAlmostEqual(summary[0]["mean_di_pre"], 0.6)
        self.assertAlmostEqual(summary[0]["mean_di_post"], 0.25)
        self.assertAlmostEqual(summary[0]["seed1_delta"], 0.10)
        self.assertAlmostEqual(summary[0]["seed2_wall_time_sec"], 14.0)
        self.assertEqual(summary[1]["variant"], "fedavg")
        self.assertEqual(summary[1]["mean_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
