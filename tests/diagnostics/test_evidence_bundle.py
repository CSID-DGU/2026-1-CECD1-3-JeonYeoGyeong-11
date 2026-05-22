import unittest

from graphfl_lab.diagnostics.evidence_bundle import (
    validate_single_run_evidence_bundle,
    validate_suite_summary_evidence_bundle,
)
from graphfl_lab.diagnostics.result_schema import with_result_schema


def series(value):
    return [[1, value]]


def single_run_payload(**metric_overrides):
    metrics = {
        "alignment_mean_pre": series(0.1),
        "alignment_mean_post": series(0.2),
        "di_pre": series(0.7),
        "di_post": series(0.5),
        "neff_pre": series(2.0),
        "neff_post": series(3.0),
        "loo_mean_pre": series(0.4),
        "loo_mean_post": series(0.3),
        "graph_density": series(0.5),
        "high_frequency_energy_ratio": series(0.25),
    }
    metrics.update(metric_overrides)
    return with_result_schema(
        {
            "meta": {},
            "results": {
                "ours": {
                    "losses_distributed": series(1.0),
                    "metrics_distributed": {"accuracy": series(0.5)},
                    "metrics_distributed_fit": metrics,
                }
            },
        }
    )


def suite_payload(**row_overrides):
    row = {
        "variant": "ours_real_graph",
        "mean_delta": 0.1,
        "min_delta": 0.0,
        "win_rate": 1.0,
        "mean_alignment_pre": 0.1,
        "mean_alignment_post": 0.2,
        "mean_di_pre": 0.7,
        "mean_di_post": 0.5,
        "mean_neff_pre": 2.0,
        "mean_neff_post": 3.0,
        "mean_loo_pre": 0.4,
        "mean_loo_post": 0.3,
        "mean_graph_density": 0.5,
        "mean_high_frequency_energy_ratio": 0.25,
    }
    row.update(row_overrides)
    return with_result_schema({"meta": {}, "summary": [row]})


class EvidenceBundleTest(unittest.TestCase):
    def test_single_run_evidence_bundle_passes_with_decision_mechanism_and_secondary_metrics(self):
        self.assertEqual(validate_single_run_evidence_bundle(single_run_payload()), [])

    def test_single_run_evidence_bundle_fails_when_mechanism_metric_is_missing(self):
        payload = single_run_payload()
        payload["results"]["ours"]["metrics_distributed_fit"].pop("di_pre")

        failures = validate_single_run_evidence_bundle(payload)

        self.assertTrue(any("di_pre" in failure for failure in failures))

    def test_single_run_evidence_bundle_requires_graph_method_payload(self):
        payload = with_result_schema(
            {
                "meta": {},
                "results": {
                    "fedavg": {
                        "losses_distributed": series(1.0),
                        "metrics_distributed": {"accuracy": series(0.5)},
                    }
                },
            }
        )

        failures = validate_single_run_evidence_bundle(payload)

        self.assertIn("results: no graph method payload with diagnostics", failures)

    def test_suite_summary_evidence_bundle_passes_with_summary_metrics(self):
        self.assertEqual(validate_suite_summary_evidence_bundle(suite_payload()), [])

    def test_suite_summary_evidence_bundle_fails_when_secondary_metric_is_missing(self):
        payload = suite_payload()
        payload["summary"][0].pop("mean_loo_pre")

        failures = validate_suite_summary_evidence_bundle(payload)

        self.assertTrue(any("mean_loo_pre" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
