"""Diagnostic metrics and artifact helpers."""

from graphfl_lab.diagnostics.logging import (
    append_client_metrics_csv,
    append_round_metrics_csv,
    init_artifact_dir,
)
from graphfl_lab.diagnostics.metrics import (
    compute_alignment,
    compute_dominance_index,
    compute_effective_client_number,
    compute_loo_distortion,
    compute_q,
    summarize_pre_post,
)
from graphfl_lab.diagnostics.schema import ClientRoundDiagnostics, RoundDiagnostics
from graphfl_lab.diagnostics.result_schema import (
    LEGACY_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    config_aliases_from_args,
    result_schema_version,
    unsupported_components_from_args,
    validate_result_schema,
    with_result_schema,
)
from graphfl_lab.diagnostics.evidence_bundle import validate_evidence_bundle

__all__ = [
    "LEGACY_RESULT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "append_client_metrics_csv",
    "append_round_metrics_csv",
    "ClientRoundDiagnostics",
    "compute_alignment",
    "compute_dominance_index",
    "compute_effective_client_number",
    "compute_loo_distortion",
    "compute_q",
    "init_artifact_dir",
    "config_aliases_from_args",
    "result_schema_version",
    "RoundDiagnostics",
    "summarize_pre_post",
    "unsupported_components_from_args",
    "validate_evidence_bundle",
    "validate_result_schema",
    "with_result_schema",
]
