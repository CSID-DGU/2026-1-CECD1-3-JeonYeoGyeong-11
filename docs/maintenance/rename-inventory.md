# Rename Inventory

This document classifies legacy names and compatibility surfaces before each
rename or cleanup pass. Update it before changing code when a new dependency or
legacy surface is discovered.

## Gate 1 Inventory Scope

| Pattern | Category | Current action |
|---|---|---|
| `spectral_fl` | package identity | Gate 3 canonical package, Gate 6 removal |
| `graphfl_lab` | target package identity | planned canonical name |
| `run_general_*` | old public runner | keep compatibility until Gate 6 |
| `run_vision_*` | current vision runner | rewire through unified path in Gate 4b |
| `run_graph_ablation.py` | Cora graph ablation runner | keep thin wrapper, rewire carefully |
| `configs/general/...` | old config path | keep resolver alias until Gate 6 |
| `result_general_*` | old result filename | compatibility mirror; new writes also emit `result_vision_*`; readers resolve either until Gate 6 |
| `general_suite_*` | old suite artifact filename | compatibility mirror; new writes emit `vision_suite_*` first via `write_suite_summary_artifacts()` until Gate 6 |
| `result_vision_*`, `vision_suite_*` | canonical output filenames | prefer in new docs, scripts, and directory listings |
| `graphfl_lab/experiments/suites/vision/artifacts.py` | artifact discovery helper | canonical-first readers for plots/smoke/sweeps |
| `spectral_filter_strength` | old config/result key | schema policy in Gate 2, removal in Gate 6 |
| `spectral_filtered_*` | old aggregation target/result spelling | internal migration in Gate 3, removal in Gate 6 |
| `ours_spectral_filtered_*` | old suite token | keep through deprecation |
| `_spectral_only`, `_speconly` | old suite suffixes | keep through deprecation |

## Gate 1 Pattern Summary

Counts below are from tracked files after Gate 0. They are inventory numbers,
not removal targets for this gate.

| Pattern | Files | Hits | Classification |
|---|---:|---:|---|
| `spectral_fl` | 163 | 625 | current package root; migrate in Gate 3, shim until Gate 6 |
| `graphfl_lab` | 6 | 18 | planned target name in docs/status only |
| `run_general` | 19 | 94 | compatibility runner/script surface |
| `run_vision` | 27 | 61 | current vision runner surface; rewire in Gate 4b |
| `run_graph_ablation.py` | 6 | 7 | Cora graph ablation runner; keep thin wrapper |
| `configs/general` | 11 | 22 | compatibility config path alias |
| `result_general_` | 23 | 304 | compatibility result filename and readers |
| `general_suite_` | 15 | 25 | compatibility suite artifact filename |
| `spectral_filter_strength` | 16 | 43 | compatibility config/result key |
| `spectral_filtered_` | 39 | 189 | compatibility aggregation target/result spelling |
| `ours_spectral_filtered` | 11 | 42 | compatibility suite token family |
| `_spectral_only` | 8 | 30 | compatibility suite suffix |
| `_speconly` | 5 | 6 | compatibility suite suffix |

## Serialized Asset Inventory

Gate 1 must search for:

```text
*.pkl
*.pickle
*.pt
*.pth
Flower state artifacts
```

This is the pickle/checkpoint compatibility inventory. If old serialized assets
reference `spectral_fl.*`, Gate 6 must either migrate
them with `scripts/dev/migrate_serialized_objects.py` or document that they are
outside compatibility guarantees after hard cleanup.

Tracked source inventory has no `*.pkl`, `*.pickle`, `*.pt`, or `*.pth` files.
Local ignored data-cache assets currently observed:

| Path | Status | Gate impact |
|---|---|---|
| `data/Cora/processed/data.pt` | ignored local dataset cache | not a package pickle/checkpoint contract |
| `data/Cora/processed/pre_filter.pt` | ignored local dataset cache | not a package pickle/checkpoint contract |
| `data/Cora/processed/pre_transform.pt` | ignored local dataset cache | not a package pickle/checkpoint contract |

## Thin Facade Protection

Top-level runners and compatibility facades are protected by:

```text
tests/structure/test_boundaries.py
```

Gate 4a and Gate 4b must keep `run_experiment.py`, `run_vision_*`,
`run_general_*`, and `run_graph_ablation.py` thin. Dispatch logic belongs in
package code, not in top-level runner files.

## Import Batch Notes

Default Gate 3 order:

```text
data/models/diagnostics/corrections
graph/designs/lifecycle
strategies
experiments
cli/scripts/tests
pyproject/Flower
```

Record any dependency-driven reorder here before applying it.

| Date | Finding | Decision |
|---|---|---|
| 2026-05-21 | Initial inventory file created. | Gate 1 will populate counts and allowlists. |
| 2026-05-21 | Gate 1 tracked-file scan found widespread current `spectral_fl` imports and compatibility names. | Treat as inventory; no removal before schema/package gates. |
| 2026-05-21 | Serialized asset scan found only ignored local Cora dataset `.pt` cache files. | Exclude data cache from pickle compatibility migration unless later inventory finds project checkpoints. |
| 2026-05-21 | Gate 3b import batch removed active `spectral_fl` import tokens outside shim tests, the gate checker, and archived scripts. | Use `graphfl_lab` for active code imports; keep `spectral_fl` only as deprecation compatibility surface. |
| 2026-05-21 | Gate 3 real move changes tracked package ownership from `spectral_fl/*` to `graphfl_lab/*`. | `spectral_fl/__init__.py` is the only legacy package file allowed until Gate 6 hard cleanup. |
