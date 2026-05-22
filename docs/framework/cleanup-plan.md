# Repository Cleanup Plan

## Rules

Use this file for staged renames that affect public commands, configs, imports, or historical outputs.

```text
Use vision/framework/research/archive/graphfl for new names.
Keep general as compatibility only.
Keep spectral only for compatibility or actual spectral/Laplacian operators.
Add new graph algorithms through GraphFLDesign/source/builder/target/lifecycle/diagnostics.
Stage risky renames: canonical name -> alias -> internal migration -> docs/configs -> tests -> removal gate.
```

## Resumable Execution

Long-running cleanup/rename work is tracked in
`docs/maintenance/cleanup-status.md`. Use that file for current gate state,
unexpected findings, existing-plan mapping, and the next safe action.

Gate checks run through:

```text
python scripts/dev/run.py gate-check <gate>
```

## Completed

| Item | Status |
|---|---|
| README rewritten around project philosophy/interfaces/run path/verification | done |
| `claim.md` rewritten around diagnostic-framework claim | done |
| `graph_fl_experimental_design.md` added as core experiment guide | done |
| early spectral/low-pass log archived | done |
| active configs moved from `configs/general/` to `configs/vision/` | done |
| `configs/general/...` resolver alias | done |
| `result_vision_*` and `vision_suite_*` aliases | done |
| strategy runtime moved to `graphfl_lab/strategies/graphfl/` | done |
| `graphfl_lab/strategies/spectral/` compatibility wrappers | done |
| canonical package root moved to `graphfl_lab` with `spectral_fl` shim | done |
| `GraphFLDiagnosticStrategy` canonical runtime | done |
| `SpectralConflictAwareStrategy` compatibility alias | done |
| `graph_filtered_*` aggregation target aliases | done |
| `--graph-filter-strength` CLI spelling | done |
| `graph_filter_strength` canonical config/result key | done |
| `spectral_filter_strength` compatibility key | done |
| `ours_graph_filtered_*` suite tokens | done |
| `ours_spectral_filtered_*` suite aliases | done |
| `graph_filter_only` suite suffix | done |
| active configs no longer use project-level `spectral` naming | done |

## Remaining Hard Renames

| Name | Target | Risk |
|---|---|---|
| package `spectral_fl` shim | remove after Gate 6 compatibility window | very high |
| key `spectral_filter_strength` | remove after reader policy freeze | medium |
| tokens `ours_spectral_filtered_*` | remove after old result reuse policy freeze | medium |
| suffix `_spectral_only` / `_speconly` | remove after old result reuse policy freeze | low/medium |
| targets `spectral_filtered_*` | remove after lifecycle/design/result readers use `graph_filtered_*` | high |

## Recommended Order

Lower-risk public spellings should stabilize before the package root rename.

```text
1. done: result/meta key -> graph_filter_strength
2. done: suite tokens -> ours_graph_filtered_*
3. done: active configs/docs prefer new key/tokens
4. done: top-level package rename to graphfl_lab with shim
5. next: remaining spectral_filtered_* design/operator compatibility removal after Gate 6 entry
```

## Migration A. Result And Metadata Key

| Field | Value |
|---|---|
| goal | canonical `graph_filter_strength` |
| alias | `spectral_filter_strength` |
| status | canonical implemented for CLI/config, metadata, diagnostics, summaries, active configs |

Checklist:

```text
graph_filter_strength in metadata/diagnostics/filter context
reader fallback: graph_filter_strength -> spectral_filter_strength
summary writers prefer canonical key
config spellings resolve: graph_filter_strength, graph-filter-strength, spectral_filter_strength
tests for output key, fallback, config parsing
```

Removal gate:

```text
no active config writes old key
readers prefer new key
historical reader behavior frozen or archived
one full suite generated with graph_filter_strength
```

## Migration B. Suite Variant Tokens

| Field | Value |
|---|---|
| goal | canonical `ours_graph_filtered_*` |
| alias | `ours_spectral_filtered_*` |
| status | canonical implemented; active configs use it; old tokens retain historical run labels |

Canonical families:

```text
ours_graph_filtered_dense
ours_graph_filtered_uniform
ours_graph_filtered_knn_kN
ours_graph_filtered_magnitude...
ours_graph_filtered_rbf...
ours_graph_filtered_random_matched_kN
```

Run tag policy:

```text
new token -> new run tag
old token -> old run tag
reporting groups both semantic families
```

Removal gate:

```text
active configs no longer use old tokens
docs no longer recommend old tokens
readers load old and new results
one smoke suite generated with new tokens
```

## Migration C. Top-Level Package Rename

| Field | Value |
|---|---|
| target | `graphfl_lab` |
| risk | highest |
| prerequisite | A/B stable |
| status | canonical package implemented; `spectral_fl` shim retained until Gate 6 |

Stages:

```text
C0 confirm graphfl_lab (done)
C1 add graphfl_lab/ alias package (done)
C2 add Flower entrypoint bridge (done)
C3 migrate internal imports in batches (done)
C4 move implementation files (done)
C5 update docs/commands (in progress)
C6 run full checks (local smoke done; remote/nightly pending)
```

Import migration batches:

```text
1. designs, graph, diagnostics, lifecycle
2. strategies
3. experiments
4. cli, scripts, tests
```

Required checks:

```text
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
python run_vision_experiment.py --help
python run_vision_suite.py --help
python run_general_suite.py --help
tiny --engine print-flwr-run smoke
```

Removal gate:

```text
no active source imports spectral_fl except wrappers
docs use graphfl_lab
tests cover canonical and old imports
historical scripts have compatibility route
```

## Migration D. Graph Source Helper Name

| Current | Target |
|---|---|
| `graphfl_lab/graph/sources/spectral.py` | `graphfl_lab/graph/sources/graphfl.py` or `graphfl_lab/graph/sources/strategy.py` |
| `graph_vectors_for_spectral` | `graph_vectors_for_graphfl` |

Status: canonical helper/module added; active code uses
`graph_vectors_for_graphfl`; `graph_vectors_for_spectral` remains as a
compatibility alias until Gate 6.

Plan:

```text
done: add canonical helper/module
done: keep old helper alias
done: migrate strategy and baseline imports
done: add alias tests
remove old helper after package migration stabilizes
```

## P0/P1 Backlog

| Priority | Task | Risk |
|---|---|---|
| Done | `spectral_filter_strength` -> `graph_filter_strength` metadata | medium |
| Done | `ours_graph_filtered_*` suite tokens | medium |
| Done | active configs graph naming | low |
| Done | package alias `graphfl_lab` | high |
| Done | real package move to `graphfl_lab/` | very high |
| Done | graph source helper rename | medium |
| P1 | migrate `spectral_filtered_*` internals | high |
| P1 | decide removal date for `configs/general/...` alias | medium |
| P1 | decide removal date for `result_general_*` names | medium |
| P2 | generated cache cleanup | low |

## Rename Pass Completion Criteria

```text
canonical name exists
old name remains as alias unless intentionally removed
active docs/configs prefer canonical name
tests cover canonical and alias names
full tests pass
diagnostic preflight passes
remaining compatibility debt recorded in naming-and-compatibility.md
```
