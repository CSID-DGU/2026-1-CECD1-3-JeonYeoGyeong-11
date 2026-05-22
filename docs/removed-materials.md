# Removed Materials And Tombstones

This document records material removed or archived during the GraphFL Lab
cleanup/rename. It is the long-term pointer for closed maintenance documents
after Gate 6.

## Release Anchors

| Anchor | Status | Notes |
|---|---|---|
| `pre-graphfl-rename` | created | SHA `e647da931bb3a78cc228ac2ad31103537b5ed640`; Gate 0 workspace baseline before Gate 1 inventory. |

## Gate 6 Hard Cleanup (2026-05-22)

Execution log (findings and decisions): [`docs/maintenance/cleanup-status.md`](maintenance/cleanup-status.md) (**closed**).

| Removed surface | Batch | Replacement / policy |
|---|---|---|
| `general_suite_*` / `result_general_*` artifact **writers** | 2 | New runs emit `vision_suite_*` / `result_vision_*` only; readers still resolve legacy paths |
| `run_general_*` root entrypoints | 3 | `run_vision_*` |
| `plot_general_convergence.py`, `merge_general_fedavg_ours.py`, `deep_dive_general.py` | 3 | `plot_vision_convergence.py` and vision-named analysis helpers |
| `graphfl_lab/experiments/general/`, `graphfl_lab/experiments/suites/general/` | 4 | `graphfl_lab/experiments/vision/`, `suites/vision/` |
| `graphfl_lab/strategies/spectral/` import facades | 5 | `graphfl_lab/strategies/graphfl/` |
| `spectral_fl/` package shim | 6 | `graphfl_lab`; scan pickles via `scripts/dev/migrate_serialized_objects.py` |
| CLI `spectral_filtered_*` argparse choices; `--spectral-filter-strength` flag | 7 | `graph_filtered_*` choices; `--graph-filter-strength`; JSON key alias `spectral_filter_strength` retained in `config_io` |
| Suite launch tokens `ours_spectral_filtered_*` | 7 | `ours_graph_filtered_*`; legacy result tags still paired in `reporting.py` |

## Post-Gate-6 Cleanup (2026-05-22)

| Removed surface | Replacement / policy |
|---|---|
| `graphfl_lab/general_*`, `graphfl_lab/cli/general_*` import/CLI facades | `graphfl_lab.data.vision`, `graphfl_lab.cli.vision_*`, `experiments.suites.vision` |
| `general_suite_*` / `result_general_*` in artifact discovery | `vision_suite_*` / `result_vision_*` only; short `suite_*` aliases kept |
| Code paths reading `general_suite_summary.csv` for stress-grid skip | `vision_suite_summary.csv` via `resolve_suite_artifact` |

Legacy experiment trees with old filenames stay **gitignored** (see root `.gitignore`).

## Post-Gate-6 Phase 2 — Spectral Alias Trim (2026-05-22)

| Removed / changed | Replacement / policy |
|---|---|
| `graphfl_lab/graph/sources/spectral.py` | `graph_vectors_for_graphfl` in `graph/sources/graphfl.py` |
| `graph_vectors_for_spectral` export | removed; import `graph_vectors_for_graphfl` |
| Strategy ctor / trace field `spectral_filter_strength` | `graph_filter_strength` only in new writes |
| Dual CLI `Namespace` setattr for filter strength | JSON key alias in `config_io` only |
| Explicit `spectral_filtered_*` branches in targets | `canonical_aggregation_target()` input map |

## Remaining Compatibility Debt

| Surface | Status | Notes |
|---|---|---|
| Runtime `spectral_filtered_*` aggregation **input** aliases | retained | `targets.canonical_aggregation_target()` + `config_io` JSON keys |
| `spectral_filter_strength` JSON config key | retained | read alias → `graph_filter_strength`; not written to new result meta |
| `ours_spectral_filtered_*` in **reporting** result tags | retained | pairs historical run labels only; not suite launch tokens |
| Diagnostic trace keys `spectral_filter_gain_*` | retained | metric names, not public rename surface |

## Tombstones

| Path or material | Status | Replacement |
|---|---|---|
| `docs/maintenance/cleanup-status.md` | **closed** Gate 6 batch 8 | This file + git history; use `gate-check` when reopening maintenance |
| `spectral_fl/__init__.py` (package shim) | removed Gate 6 batch 6 | `graphfl_lab` canonical imports |
| `docs/framework/experimental-design.md` | removed duplicate bridge | `docs/framework/graph_fl_experimental_design.md`, `docs/framework/graph_fl_experimental_design_appendix.md` |

## Archive Policy

Current project docs should stay in `docs/framework/`, `docs/research/`, and
root-level docs. Prior directions and superseded material move to `docs/archive/`
or are represented by a tombstone here with the relevant tag SHA.
