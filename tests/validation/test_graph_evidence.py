import csv
import json
import tempfile
import unittest
from pathlib import Path

from graphfl_lab.validation.graph_evidence import (
    PAPER_KERNEL_NOTE,
    composability_rows,
    design_space_rows,
    design_space_summary_rows,
    extension_contract_rows,
    external_mechanism_alignment_rows,
    generate_evidence_pack,
    graph_parity_rows,
    metric_validity_rows,
    real_diagnostic_consistency_rows,
    scenario_manifest,
)


class GraphEvidenceValidationTest(unittest.TestCase):
    def test_graph_parity_rows_cover_internal_reference_drift(self):
        rows = graph_parity_rows(profile="smoke")

        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(row["claim_level"] == "implementation-regression" for row in rows))
        self.assertTrue(all(float(row["max_abs_diff"]) <= 1e-9 for row in rows))
        self.assertTrue(all(float(row["edge_f1"]) == 1.0 for row in rows))
        self.assertTrue(all(row["verdict"] == "pass" for row in rows))

    def test_external_alignment_schema_and_pfedgraph_directionality(self):
        rows = external_mechanism_alignment_rows()
        by_component = {row["component"]: row for row in rows if row["method"] == "pFedGraph"}

        self.assertIn("directed collaboration kernel", by_component)
        self.assertIn("symmetric diagnostic projection", by_component)
        self.assertEqual(
            by_component["directed collaboration kernel"]["graph_directionality"],
            "directed",
        )
        self.assertEqual(
            by_component["symmetric diagnostic projection"]["directionality_loss"],
            "reported",
        )
        for row in rows:
            self.assertIn(row["reference_type"], {"paper-kernel", "proxy-reference"})
            self.assertTrue(row["source_url"])
            self.assertTrue(row["commit_or_version"])
            self.assertTrue(row["matched_component"])
            self.assertTrue(row["unmatched_gap"])
            self.assertEqual(row["verdict"], "pass")

    def test_manifest_declares_metric_directions_before_results(self):
        manifest = scenario_manifest(profile="smoke")

        self.assertEqual(manifest["performance_relevance"], "future-work")
        self.assertGreaterEqual(len(manifest["scenarios"]), 3)
        for scenario in manifest["scenarios"]:
            self.assertTrue(scenario["rationale"])
            self.assertTrue(scenario["good_graph_definition"])
            self.assertTrue(scenario["expected_metric_direction"])
            self.assertIn("pass_rule", scenario)

    def test_metric_validity_uses_manifest_expected_directions(self):
        manifest = scenario_manifest(profile="smoke")
        rows = metric_validity_rows(manifest)

        self.assertTrue(rows)
        framework_rows = [
            row
            for row in rows
            if row["metric_family"] == "framework_diagnostic" and row["applies"]
        ]
        self.assertTrue(framework_rows)
        self.assertTrue(all(row["threshold_kind"] == "operational sanity gate" for row in rows))
        self.assertTrue(all(row["verdict"] == "pass" for row in framework_rows))

    def test_sample_prior_reference_metric_is_non_blocking_validation_metric(self):
        manifest = scenario_manifest(profile="poster")
        rows = metric_validity_rows(manifest)
        directed_rows = [
            row
            for row in rows
            if row["scenario"] == "sample_prior_collaboration"
            and row["metric"] == "directed_row_similarity_to_ground_truth"
        ]

        self.assertTrue(directed_rows)
        self.assertTrue(all(row["metric_family"] == "validation_metric" for row in directed_rows))

    def test_composability_classifies_supported_and_explicit_unsupported(self):
        rows = composability_rows(profile="smoke")
        statuses = {row["status"] for row in rows}

        self.assertIn("supported-pass", statuses)
        self.assertIn("unsupported-explicit", statuses)
        self.assertNotIn("needs-review", statuses)
        self.assertTrue(all(row["verdict"] == "pass" for row in rows))

    def test_design_space_enumerates_claimable_cartesian_product(self):
        rows = design_space_rows(profile="smoke")
        summary = design_space_summary_rows(rows)
        total = next(row for row in summary if row["axis"] == "cartesian_product")

        self.assertEqual(len(rows), 72)
        self.assertEqual(total["supported_pass"], 72)
        self.assertEqual(total["calculation_checks_passed"], 72)
        self.assertEqual(total["needs_review"], 0)
        self.assertTrue(all(row["calculation_checks_passed"] for row in rows))
        self.assertTrue(all(row["source_vector_contract_ok"] for row in rows))
        self.assertTrue(all(row["adjacency_contract_ok"] for row in rows))
        self.assertTrue(all(row["aggregation_contract_ok"] for row in rows))
        self.assertTrue(all(row["control_semantics_ok"] for row in rows))
        self.assertTrue(all(row["graph_emitted"] for row in rows))
        self.assertTrue(all(row["diagnostics_emitted"] for row in rows))
        self.assertTrue(all(row["artifact_row_emitted"] for row in rows))

    def test_extension_contracts_cover_source_builder_and_design(self):
        rows = extension_contract_rows()
        kinds = {row["extension_kind"] for row in rows}

        self.assertIn("graph_source", kinds)
        self.assertIn("graph_builder", kinds)
        self.assertIn("design_preset", kinds)
        self.assertIn("aggregation_target", kinds)
        self.assertTrue(all(row["verdict"] == "pass" for row in rows))

    def test_real_diagnostic_zero_delta_is_flagged_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics = Path(tmp) / "diagnostics"
            diagnostics.mkdir()
            path = diagnostics / "round_metrics.csv"
            path.write_text(
                "\n".join(
                    [
                        "run_id,variant,server_round,di_pre,di_post,neff_pre,neff_post,align_mean_pre,align_mean_post,loo_mean_pre,loo_mean_post",
                        "ours_real_graph_k2_seed42,ours_real_graph_k2,1,0.1,0.1,2.0,2.0,0.3,0.3,0.4,0.4",
                        "ours_real_graph_k2_seed42,ours_real_graph_k2,2,0.2,0.2,3.0,3.0,0.5,0.5,0.6,0.6",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = real_diagnostic_consistency_rows(tmp)

        self.assertEqual(rows[0]["measurement_status"], "measured_zero_delta")
        self.assertEqual(rows[0]["verdict"], "needs-review")

    def test_real_diagnostic_identity_control_zero_delta_is_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics = Path(tmp) / "diagnostics"
            diagnostics.mkdir()
            path = diagnostics / "round_metrics.csv"
            path.write_text(
                "\n".join(
                    [
                        "run_id,variant,server_round,di_pre,di_post,neff_pre,neff_post,align_mean_pre,align_mean_post,loo_mean_pre,loo_mean_post",
                        "ours_identity_control_k2_seed42,ours_identity_control_k2,1,0.1,0.1,2.0,2.0,0.3,0.3,0.4,0.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = real_diagnostic_consistency_rows(tmp)

        self.assertEqual(rows[0]["measurement_status"], "measured_zero_delta_expected_control")
        self.assertEqual(rows[0]["verdict"], "pass")

    def test_generate_evidence_pack_writes_poster_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = generate_evidence_pack(
                Path(tmp) / "evidence",
                profile="smoke",
                include_external=True,
            )

            self.assertTrue(pack.verdict["pass"])
            expected = {
                "poster_tables",
                "claim_boundaries",
                "scenario_manifest",
                "graph_parity_summary",
                "external_mechanism_alignment",
                "metric_validity_summary",
                "composability_matrix",
                "design_space_matrix",
                "design_space_summary",
                "design_space_boundaries",
                "extension_contract_summary",
                "validation_verdict",
            }
            self.assertTrue(expected.issubset(pack.files))
            for key in expected:
                self.assertTrue(Path(pack.files[key]).is_file(), key)
            self.assertTrue((pack.out_dir / "figures" / "composability_matrix_heatmap.svg").is_file())

            verdict = json.loads((pack.out_dir / "validation_verdict.json").read_text(encoding="utf-8"))
            self.assertEqual(verdict["checks"]["performance_relevance"], "future-work")
            self.assertTrue(verdict["manifest_sha256"])

            with (pack.out_dir / "external_mechanism_alignment.csv").open(
                encoding="utf-8",
                newline="",
            ) as f:
                external_rows = list(csv.DictReader(f))
            self.assertTrue(external_rows)

            claim_text = (pack.out_dir / "claim_boundaries.md").read_text(encoding="utf-8")
            self.assertIn(PAPER_KERNEL_NOTE, claim_text)

            poster_text = (pack.out_dir / "poster_tables.md").read_text(encoding="utf-8")
            self.assertIn("min_pass_rate", poster_text)
            self.assertIn("family", poster_text)
            self.assertIn("Full Design-Space Coverage", poster_text)


if __name__ == "__main__":
    unittest.main()
