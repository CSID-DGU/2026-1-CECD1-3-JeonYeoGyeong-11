"""Evidence-bundle validation for GraphFL result payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from graphfl_lab.diagnostics.result_schema import validate_result_schema


GRAPH_METHODS = {
    "ours",
    "graph_smooth",
    "dominance_aware",
}

SINGLE_RUN_DECISION_FIELDS = (
    "losses_distributed",
    "metrics_distributed",
)

SINGLE_RUN_MECHANISM_FIELDS = (
    "alignment_mean_pre",
    "alignment_mean_post",
    "di_pre",
    "di_post",
    "neff_pre",
    "neff_post",
)

SINGLE_RUN_SECONDARY_FIELDS = (
    "loo_mean_pre",
    "loo_mean_post",
    "graph_density",
    "high_frequency_energy_ratio",
)

SUITE_DECISION_FIELDS = (
    "mean_delta",
    "min_delta",
    "win_rate",
)

SUITE_MECHANISM_FIELDS = (
    "mean_alignment_pre",
    "mean_alignment_post",
    "mean_di_pre",
    "mean_di_post",
    "mean_neff_pre",
    "mean_neff_post",
)

SUITE_SECONDARY_FIELDS = (
    "mean_loo_pre",
    "mean_loo_post",
    "mean_graph_density",
    "mean_high_frequency_energy_ratio",
)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _has_metric_series(metrics: Mapping[str, Any], key: str) -> bool:
    if key not in metrics:
        return False
    value = metrics[key]
    if value is None:
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0
    return True


def _missing_keys(mapping: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in mapping]


def _graph_method_payloads(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    results = payload.get("results")
    if not _is_mapping(results):
        return []
    graph_payloads = []
    for method, method_payload in results.items():
        if not _is_mapping(method_payload):
            continue
        method_name = str(method).lower()
        metrics = method_payload.get("metrics_distributed_fit")
        if method_name in GRAPH_METHODS or _is_mapping(metrics):
            graph_payloads.append((str(method), method_payload))
    return graph_payloads


def validate_single_run_evidence_bundle(payload: Mapping[str, Any]) -> list[str]:
    """Return missing evidence-bundle fields for a single-run result JSON."""
    failures = [f"missing schema field {key!r}" for key in validate_result_schema(payload)]
    graph_payloads = _graph_method_payloads(payload)
    if not graph_payloads:
        failures.append("results: no graph method payload with diagnostics")
        return failures

    for method, method_payload in graph_payloads:
        for key in _missing_keys(method_payload, SINGLE_RUN_DECISION_FIELDS):
            failures.append(f"results.{method}: missing decision field {key!r}")

        metrics = method_payload.get("metrics_distributed_fit")
        if not _is_mapping(metrics):
            failures.append(f"results.{method}: missing metrics_distributed_fit")
            continue
        for key in SINGLE_RUN_MECHANISM_FIELDS:
            if not _has_metric_series(metrics, key):
                failures.append(f"results.{method}.metrics_distributed_fit: missing mechanism field {key!r}")
        for key in SINGLE_RUN_SECONDARY_FIELDS:
            if not _has_metric_series(metrics, key):
                failures.append(f"results.{method}.metrics_distributed_fit: missing secondary field {key!r}")
    return failures


def validate_suite_summary_evidence_bundle(payload: Mapping[str, Any]) -> list[str]:
    """Return missing evidence-bundle fields for a suite summary JSON."""
    failures = [f"missing schema field {key!r}" for key in validate_result_schema(payload)]
    summary = payload.get("summary")
    if not isinstance(summary, list) or not summary:
        failures.append("summary: missing non-empty summary rows")
        return failures

    graph_rows = [
        row
        for row in summary
        if _is_mapping(row) and str(row.get("variant", "")).lower() != "fedavg"
    ]
    if not graph_rows:
        failures.append("summary: no graph/control variant rows")
        return failures

    for index, row in enumerate(graph_rows):
        variant = str(row.get("variant", f"row{index}"))
        for key in SUITE_DECISION_FIELDS:
            if key not in row:
                failures.append(f"summary.{variant}: missing decision field {key!r}")
        for key in SUITE_MECHANISM_FIELDS:
            if key not in row:
                failures.append(f"summary.{variant}: missing mechanism field {key!r}")
        for key in SUITE_SECONDARY_FIELDS:
            if key not in row:
                failures.append(f"summary.{variant}: missing secondary field {key!r}")
    return failures


def validate_evidence_bundle(
    payload: Mapping[str, Any],
    *,
    kind: str = "auto",
) -> list[str]:
    """Validate evidence fields for ``single-run`` or ``suite-summary`` payloads."""
    if kind == "single-run":
        return validate_single_run_evidence_bundle(payload)
    if kind == "suite-summary":
        return validate_suite_summary_evidence_bundle(payload)
    if kind != "auto":
        raise ValueError(f"Unknown evidence bundle kind: {kind!r}")
    if "summary" in payload:
        return validate_suite_summary_evidence_bundle(payload)
    return validate_single_run_evidence_bundle(payload)


__all__ = [
    "SINGLE_RUN_DECISION_FIELDS",
    "SINGLE_RUN_MECHANISM_FIELDS",
    "SINGLE_RUN_SECONDARY_FIELDS",
    "SUITE_DECISION_FIELDS",
    "SUITE_MECHANISM_FIELDS",
    "SUITE_SECONDARY_FIELDS",
    "validate_evidence_bundle",
    "validate_single_run_evidence_bundle",
    "validate_suite_summary_evidence_bundle",
]
