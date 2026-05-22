"""Developer gate-check entrypoint for staged cleanup work."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPORT_PATH = Path("docs/maintenance/last_gate_check.json")
NIGHTLY_REPORT_PATH = Path("docs/maintenance/last_nightly_run.json")

GATE0_REQUIRED_FILES = (
    "docs/maintenance/cleanup-status.md",
    "docs/maintenance/rename-inventory.md",
    "docs/maintenance/line-budget-allowlist.txt",
    "MIGRATION.md",
    "docs/removed-materials.md",
    "scripts/dev/run.py",
    "tests/golden/README.md",
)

GATE0_REQUIRED_TEXT = {
    "docs/maintenance/cleanup-status.md": (
        "Existing Unstaged Docs",
        "Existing Plan Mapping",
        "scripts/dev/run.py gate-check <gate>",
        "Gate 4c",
        "workflow_dispatch",
        "real move verification",
        "single Gate branch",
        "docs/framework/claim.md",
        "docs/framework/graph_fl_experimental_design.md",
        "docs/framework/graph_fl_experimental_design_appendix.md",
        "docs/research/prior-work-review.md",
    ),
    "docs/maintenance/rename-inventory.md": (
        "tests/structure/test_boundaries.py",
        "pickle/checkpoint",
        "spectral_fl",
        "graphfl_lab",
    ),
    "docs/maintenance/line-budget-allowlist.txt": (
        "git diff --numstat",
        "added - removed must be <= 0",
    ),
    "MIGRATION.md": (
        "For Users",
        "For Maintainers",
        "pre-graphfl-rename",
    ),
    "docs/removed-materials.md": (
        "pre-graphfl-rename",
        "Tombstones",
    ),
    "tests/golden/README.md": (
        "Volatile Fields",
        "timestamp",
        "run_id",
        "Schema comparison is exact",
    ),
}

GATE1_REQUIRED_TEXT = {
    "docs/maintenance/rename-inventory.md": (
        "Gate 1 Pattern Summary",
        "Serialized Asset Inventory",
        "data/Cora/processed/data.pt",
        "tests/structure/test_boundaries.py",
        "spectral_fl",
        "result_general_",
        "spectral_filter_strength",
    ),
    "docs/maintenance/line-budget-allowlist.txt": (
        "Protected Paths",
        "graphfl_lab/experiments/suites/vision/variants.py",
        "graphfl_lab/strategies/graphfl/strategy.py",
        "graphfl_lab/experiments/vision/suite.py",
        "added - removed <= 0",
    ),
    "docs/removed-materials.md": (
        "pre-graphfl-rename",
        "e647da931bb3a78cc228ac2ad31103537b5ed640",
    ),
}

GATE2_REQUIRED_TEXT = {
    "graphfl_lab/diagnostics/result_schema.py": (
        "RESULT_SCHEMA_VERSION",
        "LEGACY_RESULT_SCHEMA_VERSION",
        "with_result_schema",
        "config_aliases_used",
        "unsupported_components",
    ),
    "graphfl_lab/config_io.py": (
        "_config_aliases_used",
        "ARG_DEST_ALIASES",
        "configs/general->configs/vision",
    ),
    "graphfl_lab/flower_app.py": (
        "with_result_schema",
        "config_aliases_from_args",
        "unsupported_components_from_args",
    ),
    "graphfl_lab/experiments/vision/suite.py": (
        "with_result_schema",
        "config_aliases_from_args",
        "unsupported_components_from_args",
    ),
    "graphfl_lab/experiments/cora/graph_ablation.py": (
        "with_result_schema",
        "config_aliases_from_args",
        "unsupported_components_from_args",
    ),
    "tests/diagnostics/test_result_schema.py": (
        "test_with_result_schema_adds_required_fields",
        "test_missing_version_reads_as_v0",
        "test_config_aliases_are_recorded",
    ),
}

GATE3_REQUIRED_TEXT = {
    "graphfl_lab/__init__.py": (
        "Canonical package root",
    ),
    "pyproject.toml": (
        "graphfl_lab*",
        "graphfl_lab.flower_app:server_app",
        "graphfl_lab.flower_app:client_app",
    ),
    "docs/removed-materials.md": (
        "spectral_fl/__init__.py",
        "Gate 6 batch 6",
    ),
    "tests/core/test_package_alias.py": (
        "test_graphfl_lab_imports_flower_app",
        "test_spectral_fl_shim_removed",
        "test_pickle_round_trip_for_canonical_import",
    ),
}

GATE3B_FORBIDDEN_IMPORT_ALLOWLIST = {
    "graphfl_lab/__init__.py",
    "scripts/dev/migrate_serialized_objects.py",
    "scripts/dev/run.py",
    "tests/dev/test_run_gate_check.py",
}

GATE3B_FORBIDDEN_IMPORTS = (
    "from spectral_fl",
    "import spectral_fl",
    "spectral_fl.",
)

GATE4A_REQUIRED_TEXT = {
    "graphfl_lab/cli/experiment_dispatcher.py": (
        "--track",
        "vision",
        "cora",
        "DeprecationWarning",
        "TRACK_MODULES",
    ),
    "run_experiment.py": (
        "experiment_dispatcher",
        "main = _impl.main",
    ),
    "tests/cli/test_experiment_dispatcher.py": (
        "test_missing_track_defaults_to_cora_with_deprecation_warning",
        "test_track_cora_dispatches_without_track_argument",
        "test_track_vision_dispatches_without_track_argument",
    ),
}

GATE4B_REQUIRED_TEXT = {
    "graphfl_lab/cli/experiment_dispatcher.py": (
        "main_for_track",
        "vision_main",
        "cora_main",
    ),
    "run_vision_experiment.py": (
        "experiment_dispatcher",
        "main = _dispatcher.vision_main",
    ),
    "tests/cli/test_experiment_dispatcher.py": (
        "test_named_track_helpers_use_unified_dispatch",
    ),
}

GATE4C_REQUIRED_TEXT = {
    ".github/workflows/ci.yml": (
        "graphfl_lab",
        "python -m unittest discover -s tests",
    ),
    ".github/workflows/nightly.yml": (
        "schedule:",
        "workflow_dispatch:",
        "graphfl_lab",
        "python -m unittest discover -s tests",
    ),
    "scripts/dev/golden.py": (
        "VOLATILE_FIELDS",
        "REQUIRED_RESULT_SCHEMA_KEYS",
        "normalize_payload",
        "compare_payloads",
    ),
    "tests/dev/test_golden.py": (
        "test_normalized_compare_ignores_volatile_fields",
        "test_compare_fails_on_schema_shape_change",
        "test_compare_fails_on_normalized_value_change",
    ),
    "tests/golden/README.md": (
        "Gate 4c captures smoke outputs",
        "Volatile Fields",
        "Schema comparison is exact",
    ),
}

GATE5A_PREP_REQUIRED_TEXT = {
    "graphfl_lab/experiments/suites/result_writer.py": (
        "write_json",
        "write_csv_rows",
    ),
    "graphfl_lab/experiments/vision/suite.py": (
        "write_suite_summary_artifacts(",
        "summary_json, rows_path, csv_path = write_suite_summary_artifacts",
    ),
    "graphfl_lab/experiments/cora/graph_ablation.py": (
        "write_json(summary_json, suite_summary)",
        "write_json(rows_path, rows)",
        "write_csv_rows(csv_path, summary_rows",
    ),
    "graphfl_lab/experiments/vision/stress_grid.py": (
        "write_json(root / \"stress_grid_auto_review.json\"",
        "write_json(root / \"stress_grid_manifest.json\"",
        "write_json(root / \"stress_grid_summary.json\"",
    ),
    "graphfl_lab/experiments/vision/client_count_sweep.py": (
        "write_json(root / \"client_count_sweep_summary.json\"",
    ),
    "graphfl_lab/experiments/suites/vision/reporting.py": (
        "def write_suite_summary_artifacts(",
        'summary_json = out_dir / "vision_suite_summary.json"',
        "write_json(summary_json, suite_summary)",
        "write_csv_rows(",
    ),
    "tests/experiments/test_result_writer.py": (
        "test_write_json_uses_indented_payload",
        "test_write_csv_rows_preserves_field_order",
    ),
    "docs/maintenance/cleanup-status.md": (
        "Gate 5a-prep",
        "do not claim full Gate 5a completion",
    ),
}

GATE5B_PREP_REQUIRED_TEXT = {
    "graphfl_lab/experiments/suites/execution.py": (
        "run_cmd",
        "execute_or_reuse_result",
        "reuse_existing",
    ),
    "graphfl_lab/experiments/vision/suite.py": (
        "from graphfl_lab.experiments.suites.execution import execute_or_reuse_result",
        "from graphfl_lab.experiments.suites.vision.features import",
        "cwd=PROJECT_ROOT",
    ),
    "graphfl_lab/experiments/suites/vision/features.py": (
        "collect_run_features",
        "collect_timing_features",
        "load_preloaded_fedavg_accs",
        "rank_key",
    ),
    "graphfl_lab/experiments/suites/vision/summary.py": (
        "build_summary_rows",
        "mean_di_pre",
        "seed{seed}_delta",
    ),
    "graphfl_lab/experiments/suites/vision/metadata.py": (
        "build_suite_meta",
        "record_suite_timing",
        "mean_di_pre/post",
    ),
    "graphfl_lab/experiments/vision/client_count_sweep.py": (
        "from graphfl_lab.experiments.suites.execution import run_cmd",
        "run_cmd(cmd, cwd=PROJECT_ROOT)",
    ),
    "graphfl_lab/experiments/vision/stress_grid.py": (
        "from graphfl_lab.experiments.suites.execution import run_cmd",
        "run_cmd(cmd, cwd=PROJECT_ROOT)",
    ),
    "tests/experiments/test_suite_execution.py": (
        "test_execute_reuses_existing_result_without_running_command",
        "test_execute_runs_when_reuse_disabled",
        "test_run_cmd_uses_cwd_and_check",
    ),
    "tests/experiments/vision/test_suite_features.py": (
        "test_collect_run_features_exports_diagnostic_means_and_aliases",
        "test_load_preloaded_fedavg_accs_reads_latest_supported_names",
    ),
    "tests/experiments/vision/test_suite_summary.py": (
        "test_build_summary_rows_aggregates_all_diagnostic_fields",
    ),
    "tests/experiments/vision/test_suite_metadata.py": (
        "test_build_suite_meta_documents_full_diagnostic_set",
        "test_record_preloaded_and_timing_metadata",
    ),
    "docs/maintenance/cleanup-status.md": (
        "Gate 5b-prep",
        "do not claim full Gate 5b completion",
    ),
}

GATE5C_PREP_REQUIRED_TEXT = {
    "graphfl_lab/experiments/suites/vision/variant_helpers.py": (
        "token_float",
        "diagnostic_graph_args",
        "result_path_for_variant",
    ),
    "graphfl_lab/experiments/suites/vision/variant_commands.py": (
        "build_base_cmd",
        "_user_arg_dests",
        "--train-subset-size",
    ),
    "graphfl_lab/experiments/suites/vision/variant_core.py": (
        "parse_core_graph_variant",
        "ours_default_graph",
        "ours_graph_mode_",
    ),
    "graphfl_lab/experiments/suites/vision/variant_families.py": (
        "parse_baseline_variant",
        "--fedopt-eta",
        "fedsim_rbf_knn_k",
    ),
    "graphfl_lab/experiments/suites/vision/variant_diagnostics.py": (
        "parse_diagnostic_variant",
        "ours_real_graph",
        "ours_graphfree_",
    ),
    "graphfl_lab/experiments/suites/vision/variant_legacy.py": (
        "parse_legacy_residual_variant",
        "ours_legacy_residual_reweight",
        "legacy_residual_reweight_args",
    ),
    "graphfl_lab/experiments/suites/vision/variant_sources.py": (
        "parse_source_variant",
        "classifier_head_weight",
        "layer_slice_update",
    ),
    "graphfl_lab/experiments/suites/vision/variant_suffixes.py": (
        "parse_suffix_variant",
        "graph_filter_only",
        "parser(base, args)",
    ),
    "graphfl_lab/experiments/suites/vision/variant_targets.py": (
        "parse_target_variant",
        "graph_filtered_update",
        '"spectral_filtered"',
    ),
    "graphfl_lab/experiments/suites/vision/variants.py": (
        "from graphfl_lab.experiments.suites.vision.variant_commands import build_base_cmd",
        "from graphfl_lab.experiments.suites.vision.variant_core import parse_core_graph_variant",
        "from graphfl_lab.experiments.suites.vision.variant_diagnostics import parse_diagnostic_variant",
        "from graphfl_lab.experiments.suites.vision.variant_families import parse_baseline_variant",
        "from graphfl_lab.experiments.suites.vision.variant_helpers import",
        "from graphfl_lab.experiments.suites.vision.variant_legacy import",
        "from graphfl_lab.experiments.suites.vision.variant_sources import parse_source_variant",
        "from graphfl_lab.experiments.suites.vision.variant_suffixes import parse_suffix_variant",
        "from graphfl_lab.experiments.suites.vision.variant_targets import parse_target_variant",
        "result_path_for_variant(out_dir, method, seed, run_tag)",
    ),
    "tests/experiments/vision/test_variant_helpers.py": (
        "test_result_path_for_variant_uses_canonical_vision_filename",
        "test_diagnostic_graph_args_adds_cluster_auto_k_for_cluster_only",
    ),
    "tests/experiments/vision/test_variant_commands.py": (
        "test_build_base_cmd_preserves_core_command_contract",
        "test_build_base_cmd_forwards_only_explicit_user_graph_args",
    ),
    "tests/experiments/vision/test_variant_core.py": (
        "test_parse_core_graph_variant_handles_default_and_basic_graphs",
        "test_parse_core_graph_variant_returns_none_for_source_specific_family",
    ),
    "tests/experiments/vision/test_variant_families.py": (
        "test_parse_baseline_variant_handles_fedopt_suffixes",
        "test_parse_baseline_variant_returns_none_for_ours",
    ),
    "tests/experiments/vision/test_variant_diagnostics.py": (
        "test_parse_diagnostic_variant_handles_real_and_control_graphs",
        "test_parse_diagnostic_variant_returns_none_for_other_families",
    ),
    "tests/experiments/vision/test_variant_legacy.py": (
        "test_parse_legacy_residual_variant_handles_old_and_compat_tokens",
        "test_parse_legacy_residual_variant_returns_none_for_current_tokens",
    ),
    "tests/experiments/vision/test_variant_sources.py": (
        "test_parse_source_variant_handles_weight_and_head_graphs",
        "test_parse_source_variant_returns_none_for_target_only_family",
    ),
    "tests/experiments/vision/test_variant_suffixes.py": (
        "test_parse_suffix_variant_handles_tau_lowpass_and_server_momentum",
        "test_parse_suffix_variant_handles_filter_only_and_rejects_baselines",
    ),
    "tests/experiments/vision/test_variant_targets.py": (
        "test_parse_target_variant_handles_graph_filtered_family",
        "test_parse_target_variant_preserves_legacy_spectral_family",
    ),
    "docs/maintenance/cleanup-status.md": (
        "Gate 5c-prep",
        "leave `parse_variant` branch order unchanged",
    ),
}

GATE5D_PREP_REQUIRED_TEXT = {
    "graphfl_lab/strategies/graphfl/artifact_rows.py": (
        "build_round_diagnostics_row",
        "build_graph_stats_row",
        "build_client_diagnostic_rows",
    ),
    "graphfl_lab/strategies/graphfl/ema.py": (
        "update_client_update_ema",
        "initialized_current_update",
        "ema_update",
    ),
    "graphfl_lab/strategies/graphfl/client_metrics.py": (
        "extract_metric",
        "weighted_optional_mean",
        "return out if any_found else None",
    ),
    "graphfl_lab/strategies/graphfl/config_context.py": (
        "build_config_context",
        "spectral_filter_strength",
        "warmup_rounds",
    ),
    "graphfl_lab/strategies/graphfl/conflict_metrics.py": (
        "ConflictMetrics",
        "compute_conflict_metric_bundle",
        "tau_source_name",
    ),
    "graphfl_lab/strategies/graphfl/counterfactual_artifacts.py": (
        "run_counterfactual_artifacts",
        "counterfactual_specs_for_target",
        "graphfree_dominance_reweight",
        "from graphfl_lab.strategies.graphfl.trace_context import with_run_context",
    ),
    "graphfl_lab/strategies/graphfl/diagnostic_artifacts.py": (
        "write_round_diagnostic_artifacts",
        "append_round_metrics_csv",
        "append_counterfactual_metrics_csv",
        "run_counterfactual_artifacts",
    ),
    "graphfl_lab/strategies/graphfl/diagnostic_targets.py": (
        "flatten_diagnostic_post_updates",
        "graph_filtered_client_ema_update_delta",
        "Unknown diagnostic aggregation_target",
    ),
    "graphfl_lab/strategies/graphfl/fit_results.py": (
        "ClientFitBatch",
        "collect_client_fit_batch",
        "sort_fit_results_by_cid",
    ),
    "graphfl_lab/strategies/graphfl/graph_metadata.py": (
        "client_cluster_ids_from_meta",
        "cluster_ids",
        "return [-1 for _ in cids]",
    ),
    "graphfl_lab/strategies/graphfl/graph_state.py": (
        "select_round_graph",
        "warmup_current_graph",
        "raw_current_graph",
    ),
    "graphfl_lab/strategies/graphfl/projection.py": (
        "project_with_cached_matrix",
        "make_gaussian_projection",
        "compression_seed",
    ),
    "graphfl_lab/strategies/graphfl/round_context.py": (
        "build_spectral_context",
        "build_conflict_context",
        "build_alpha_context",
    ),
    "graphfl_lab/strategies/graphfl/round_graph.py": (
        "RoundGraphState",
        "build_round_graph_state",
        "from graphfl_lab.strategies.graphfl.graph_metadata import client_cluster_ids_from_meta",
        "from graphfl_lab.strategies.graphfl.graph_state import select_round_graph",
        "int(graph_seed) * 1009 + int(server_round) * 13",
    ),
    "graphfl_lab/strategies/graphfl/round_outputs.py": (
        "RoundStrategyOutputs",
        "build_strategy_round_outputs",
        "make_round_trace_payload",
        "build_fit_metrics",
    ),
    "graphfl_lab/strategies/graphfl/round_projection.py": (
        "ProjectedGraphSpace",
        "build_projected_graph_space",
        "select_graph_source_vectors",
        "graph_vectors_for_graphfl",
        "project_fn",
    ),
    "graphfl_lab/strategies/graphfl/round_weights.py": (
        "select_round_weights",
        "select_aggregation_weights",
        "apply_correction_family",
    ),
    "graphfl_lab/strategies/graphfl/spectral_metrics.py": (
        "RoundSpectralMetrics",
        "compute_round_spectral_metrics",
        "previous_round_graph",
    ),
    "graphfl_lab/strategies/graphfl/trace_context.py": (
        "with_run_context",
        "values.setdefault(\"run_id\"",
        "round_number",
    ),
    "graphfl_lab/strategies/graphfl/update_space.py": (
        "UpdateSpaceArrays",
        "compute_local_updates",
        "compute_update_space_arrays",
    ),
    "graphfl_lab/strategies/graphfl/strategy.py": (
        "from graphfl_lab.strategies.graphfl.client_metrics import",
        "from graphfl_lab.strategies.graphfl.config_context import build_config_context",
        "from graphfl_lab.strategies.graphfl.conflict_metrics import",
        "from graphfl_lab.strategies.graphfl.diagnostic_artifacts import",
        "from graphfl_lab.strategies.graphfl.diagnostic_targets import",
        "from graphfl_lab.strategies.graphfl.ema import update_client_update_ema",
        "from graphfl_lab.strategies.graphfl.projection import project_with_cached_matrix",
        "from graphfl_lab.strategies.graphfl.round_graph import build_round_graph_state",
        "from graphfl_lab.strategies.graphfl.round_outputs import build_strategy_round_outputs",
        "from graphfl_lab.strategies.graphfl.round_projection import",
        "from graphfl_lab.strategies.graphfl.round_weights import select_round_weights",
        "from graphfl_lab.strategies.graphfl.spectral_metrics import",
        "from graphfl_lab.strategies.graphfl.update_space import",
        "write_round_diagnostic_artifacts(",
        "round_outputs = build_strategy_round_outputs(",
        "graph_space = build_projected_graph_space(",
        "config_context = build_config_context(self)",
        "conflict_metrics = compute_conflict_metric_bundle(",
        "local_updates = compute_local_updates(",
        "update_space = compute_update_space_arrays(",
        "spectral_metrics = compute_round_spectral_metrics(",
        "extract_metric(client_metrics",
        "flatten_diagnostic_post_updates(",
        "fit_batch = collect_client_fit_batch(results)",
        "weighted_optional_mean(client_train_acc",
        "update_client_update_ema(",
        "project_with_cached_matrix(",
        "round_graph = build_round_graph_state(",
        "weight_selection = select_round_weights(",
    ),
    "tests/strategies/graphfl/test_ema.py": (
        "test_update_client_update_ema_initializes_and_copies_updates",
        "test_update_client_update_ema_blends_existing_updates",
    ),
    "tests/strategies/graphfl/test_artifact_rows.py": (
        "test_build_round_diagnostics_row_preserves_full_metric_contract",
        "test_build_client_diagnostic_rows_preserves_per_client_metrics",
    ),
    "tests/strategies/graphfl/test_client_metrics.py": (
        "test_extract_metric_uses_first_available_alias_per_client",
        "test_weighted_optional_mean_skips_missing_values",
    ),
    "tests/strategies/graphfl/test_config_context.py": (
        "test_build_config_context_projects_explicit_round_config_fields",
        "spectral_filter_strength",
    ),
    "tests/strategies/graphfl/test_conflict_metrics.py": (
        "test_compute_conflict_metric_bundle_uses_h_spec_tau_source",
        "test_compute_conflict_metric_bundle_updates_non_hspec_tau_candidate",
    ),
    "tests/strategies/graphfl/test_counterfactual_artifacts.py": (
        "test_counterfactual_specs_for_target_retargets_graph_variants_only",
        "test_run_counterfactual_artifacts_emits_rows_and_trace_context",
    ),
    "tests/strategies/graphfl/test_diagnostic_targets.py": (
        "test_flatten_diagnostic_post_updates_returns_update_delta_matrix",
        "test_flatten_diagnostic_post_updates_supports_filtered_ema_aliases",
    ),
    "tests/strategies/graphfl/test_fit_results.py": (
        "test_collect_client_fit_batch_sorts_by_numeric_cid_and_converts_arrays",
        "test_collect_client_fit_batch_uses_proxy_cid_when_metric_missing",
    ),
    "tests/strategies/graphfl/test_graph_metadata.py": (
        "test_client_cluster_ids_from_meta_converts_matching_list",
        "test_client_cluster_ids_from_meta_uses_fallback_when_length_mismatch",
    ),
    "tests/strategies/graphfl/test_graph_state.py": (
        "test_select_round_graph_preserves_ema_label_when_previous_missing",
        "test_select_round_graph_blends_previous_and_current_graph",
    ),
    "tests/strategies/graphfl/test_projection.py": (
        "test_project_with_cached_matrix_returns_float32_when_small",
        "test_project_with_cached_matrix_creates_and_reuses_matrix",
    ),
    "tests/strategies/graphfl/test_round_context.py": (
        "test_build_spectral_context_preserves_tau_and_filter_fields",
        "test_build_context_helpers_preserve_round_log_inputs",
    ),
    "tests/strategies/graphfl/test_round_weights.py": (
        "test_select_round_weights_preserves_conflict_aware_alpha_mode",
        "test_select_round_weights_preserves_graph_free_alpha_mode_suffix",
    ),
    "tests/strategies/graphfl/test_round_graph.py": (
        "test_build_round_graph_state_preserves_sample_weight_normalization",
        "test_build_round_graph_state_preserves_seeded_control_graphs",
    ),
    "tests/strategies/graphfl/test_spectral_metrics.py": (
        "test_compute_round_spectral_metrics_uses_current_graph_without_previous",
        "test_compute_round_spectral_metrics_uses_previous_laplacian_for_metric",
    ),
    "tests/strategies/graphfl/test_trace_context.py": (
        "test_with_run_context_adds_missing_round_and_values",
        "test_with_run_context_preserves_existing_round_and_values",
    ),
    "tests/strategies/graphfl/test_update_space.py": (
        "test_compute_local_updates_subtracts_global_parameters",
        "test_compute_update_space_arrays_preserves_flat_delta_matrix_and_norms",
    ),
    "docs/maintenance/cleanup-status.md": (
        "Gate 5d-prep",
        "keep CSV/JSONL append calls in `GraphFLDiagnosticStrategy`",
        "keep CSV append calls in `GraphFLDiagnosticStrategy`",
        "keep the explicit field list tested",
        "keep state assignment for H_spec EMA in `GraphFLDiagnosticStrategy`",
        "keep tau-signal EMA state assignment in `GraphFLDiagnosticStrategy`",
        "preserve flattened delta matrix used by diagnostics and counterfactuals",
        "preserve graph seed formula and sample-weight normalization",
        "keep `build_round_log` and `build_fit_metrics` as the output contract",
        "preserve existing alpha_mode composition",
        "preserve stable CID ordering from baselines ordering",
        "preserve `-1` fallback for missing or mismatched cluster lists",
        "preserve existing `ema_graph` label when EMA is enabled outside warmup",
        "preserve default round/run_id/variant/seed insertion",
        "keep aggregate_fit call sites unchanged apart from function names",
        "keep `_diagnostic_post_flat_updates` as a wrapper around current strategy state",
        "keep state assignment in `GraphFLDiagnosticStrategy`",
        "keep projection matrix storage on `GraphFLDiagnosticStrategy`",
    ),
}


def repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


def commit_sha(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _load_nightly_evidence(root: Path) -> dict[str, object] | None:
    path = root / NIGHTLY_REPORT_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _gate4c_nightly_evidence_checks(root: Path) -> list[str]:
    evidence = _load_nightly_evidence(root)
    if evidence is None:
        return [
            "Gate 4c requires one GitHub nightly or manual-nightly green run before completion.",
            f"Record evidence in {NIGHTLY_REPORT_PATH.as_posix()} after a green run.",
        ]
    if str(evidence.get("workflow", "")).strip() != "nightly":
        return ["last_nightly_run.json: workflow must be 'nightly'"]
    if str(evidence.get("conclusion", "")).strip().lower() != "success":
        return [
            "last_nightly_run.json: conclusion must be 'success' for Gate 4c entry"
        ]
    if str(evidence.get("ref", "")).strip() not in {"main", "refs/heads/main"}:
        return ["last_nightly_run.json: ref must record a main-branch nightly run"]
    return []


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _missing_files(root: Path, rel_paths: Iterable[str]) -> list[str]:
    return [rel for rel in rel_paths if not (root / rel).is_file()]


def _missing_text(root: Path, expectations: dict[str, Iterable[str]]) -> list[str]:
    failures: list[str] = []
    for rel, needles in expectations.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"{rel}: file missing")
            continue
        haystack = _read_text(path)
        for needle in needles:
            if needle not in haystack:
                failures.append(f"{rel}: missing text {needle!r}")
    return failures


def _tag_exists(root: Path, tag_name: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", tag_name],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0


def _tracked_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _tracked_serialized_assets(root: Path) -> list[str]:
    suffixes = (".pkl", ".pickle", ".pt", ".pth")
    return [path for path in _tracked_files(root) if path.lower().endswith(suffixes)]


def _forbidden_identity_imports(root: Path) -> list[str]:
    failures: list[str] = []
    for rel in _tracked_files(root):
        if not rel.endswith(".py"):
            continue
        if rel in GATE3B_FORBIDDEN_IMPORT_ALLOWLIST:
            continue
        if rel.startswith("scripts/archive/"):
            continue
        path = root / rel
        if not path.is_file():
            continue
        text = _read_text(path)
        for needle in GATE3B_FORBIDDEN_IMPORTS:
            if needle in text:
                failures.append(f"{rel}: forbidden legacy import token {needle!r}")
                break
    return failures


def _unexpected_legacy_package_files(root: Path) -> list[str]:
    failures: list[str] = []
    for rel in _tracked_files(root):
        if rel.startswith("spectral_fl/"):
            failures.append(f"{rel}: spectral_fl package removed in Gate 6 batch 6")
    return failures


def run_gate_check(gate: str, root: Path | None = None) -> dict[str, object]:
    root = repo_root(root)
    failed_checks: list[str] = []

    if gate == "0":
        for rel in _missing_files(root, GATE0_REQUIRED_FILES):
            failed_checks.append(f"missing required Gate 0 file: {rel}")
        failed_checks.extend(_missing_text(root, GATE0_REQUIRED_TEXT))
    elif gate == "1":
        failed_checks.extend(_missing_text(root, GATE1_REQUIRED_TEXT))
        if not _tag_exists(root, "pre-graphfl-rename"):
            failed_checks.append("missing tag: pre-graphfl-rename")
        serialized = _tracked_serialized_assets(root)
        if serialized:
            failed_checks.append(
                "tracked serialized assets must be classified explicitly: "
                + ", ".join(serialized)
            )
    elif gate == "2":
        failed_checks.extend(_missing_text(root, GATE2_REQUIRED_TEXT))
    elif gate == "3a":
        failed_checks.append("Gate 3a alias bridge is superseded by full Gate 3.")
    elif gate == "3b":
        failed_checks.extend(_forbidden_identity_imports(root))
    elif gate == "3":
        failed_checks.extend(_missing_text(root, GATE3_REQUIRED_TEXT))
        failed_checks.extend(_forbidden_identity_imports(root))
        failed_checks.extend(_unexpected_legacy_package_files(root))
    elif gate == "4a":
        failed_checks.extend(_missing_text(root, GATE4A_REQUIRED_TEXT))
    elif gate == "4b":
        failed_checks.extend(_missing_text(root, GATE4A_REQUIRED_TEXT))
        failed_checks.extend(_missing_text(root, GATE4B_REQUIRED_TEXT))
    elif gate == "4c":
        failed_checks.extend(_missing_text(root, GATE4C_REQUIRED_TEXT))
        failed_checks.extend(_gate4c_nightly_evidence_checks(root))
    elif gate == "5a-prep":
        failed_checks.extend(_missing_text(root, GATE5A_PREP_REQUIRED_TEXT))
    elif gate == "5b-prep":
        failed_checks.extend(_missing_text(root, GATE5B_PREP_REQUIRED_TEXT))
    elif gate == "5c-prep":
        failed_checks.extend(_missing_text(root, GATE5C_PREP_REQUIRED_TEXT))
    elif gate == "5d-prep":
        failed_checks.extend(_missing_text(root, GATE5D_PREP_REQUIRED_TEXT))
    else:
        failed_checks.append(
            f"Gate {gate} check is not implemented yet; add it during that gate."
        )

    return {
        "gate": str(gate),
        "pass": not failed_checks,
        "failed_checks": failed_checks,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha(root),
    }


def write_report(root: Path, report: dict[str, object]) -> Path:
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate-check", help="Run a cleanup gate check.")
    gate.add_argument("gate", help="Gate id, for example 0, 4a, or 5d-prep.")
    args = parser.parse_args(argv)

    root = repo_root()
    if args.command == "gate-check":
        report = run_gate_check(str(args.gate), root)
        report_path = write_report(root, report)
        print(json.dumps(report, indent=2))
        print(f"Wrote {report_path.relative_to(root)}")
        return 0 if report["pass"] else 1
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
