# Repository Cleanup Plan

## Rules

Use this file for staged renames that affect public commands, configs, imports, or
historical outputs.

```text
Use vision/framework/research/archive/graphfl for new names.
Do not reintroduce removed general_* or spectral_* public surfaces (see removed-materials.md).
Keep "spectral" only for Laplacian/operator math and diagnostic metric field names.
Add new graph algorithms through GraphFLDesign/source/builder/target/lifecycle/diagnostics.
```

## Status (2026-05-22)

Gate 6 hard cleanup, post-Gate-6 `general_*` removal, and Phase 2 spectral
alias trim are **complete** on `main`.

| Reference | Role |
|---|---|
| `docs/maintenance/cleanup-status.md` | **closed** execution log |
| `docs/removed-materials.md` | tombstones + remaining read-only aliases |
| `docs/framework/naming-and-compatibility.md` | canonical vs compatibility policy |
| `docs/maintenance/gate-6-prep.md` | Gate 6 checklist (all done) |

Reopen maintenance only with a new finding line in `cleanup-status.md` and:

```text
python scripts/dev/run.py gate-check <gate>
```

## Completed

| Item | Status |
|---|---|
| README around project philosophy/interfaces/run path/verification | done |
| `claim.md` around diagnostic-framework claim | done |
| `graph_fl_experimental_design.md` as core experiment guide | done |
| active configs under `configs/vision/` | done |
| `result_vision_*` and `vision_suite_*` canonical writers | done |
| strategy runtime in `graphfl_lab/strategies/graphfl/` | done |
| canonical package `graphfl_lab`; `spectral_fl` shim removed | done |
| `GraphFLDiagnosticStrategy` canonical runtime | done |
| `graph_filtered_*` aggregation targets and outputs | done |
| `--graph-filter-strength` / `graph_filter_strength` | done |
| `ours_graph_filtered_*` suite tokens | done |
| Gate 6: `run_general_*`, general experiment trees, spectral strategy facades | done |
| Gate 6: CLI spectral choices, suite `ours_spectral_filtered_*` launch tokens | done |
| Post-Gate-6: `graphfl_lab/general_*`, legacy artifact readers | done |
| Phase 2: `graph_vectors_for_graphfl`; drop spectral trace key mirrors | done |

## Remaining Optional Debt

Breaking removals — only after explicit policy to stop loading old JSON/results:

| Surface | Risk | Notes |
|---|---|---|
| `configs/general/...` path alias | low | convenience for old command lines |
| `spectral_filter_strength` JSON key alias | medium | `config_io` only |
| `spectral_filtered_*` input aliases | medium | `targets.py` / lifecycle |
| `ours_spectral_filtered_*` reporting pairs | low | historical CSV/result tags |
| `__pycache__` sweep | low | hygiene only |

## Verification Commands

```text
python -m unittest discover -s tests
python scripts/dev/run.py gate-check 0
python scripts/dev/run.py gate-check 5d-prep
python scripts/checks/diagnostic_suite_preflight.py
python run_vision_experiment.py --help
python run_vision_suite.py --help
python scripts/reports/plot_vision_convergence.py --help
```

## Rename Pass Completion Criteria

```text
canonical name exists in active code
active docs/configs prefer canonical name
tests and gate-check pass
remaining read-only compatibility recorded in naming-and-compatibility.md and removed-materials.md
```
