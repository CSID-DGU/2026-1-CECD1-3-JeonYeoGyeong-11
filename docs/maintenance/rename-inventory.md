# Rename Inventory

This document classifies legacy names and compatibility surfaces before each
rename or cleanup pass. Update it before changing code when a new dependency or
legacy surface is discovered.

**Post-Gate-6 (2026-05-22):** Gate 6 and post-Gate-6 cleanup are complete on
`main`. The tables below retain Gate 0–1 inventory wording for history; current
policy is in [`docs/removed-materials.md`](../removed-materials.md) and
[`docs/framework/naming-and-compatibility.md`](../framework/naming-and-compatibility.md).

| Pattern (historical) | Current status on `main` |
|---|---|
| `spectral_fl`, `run_general_*`, `general_suite_*`, `result_general_*` readers | **removed** from active code |
| `spectral_filtered_*` CLI / suite launch | **removed**; JSON/input aliases retained |
| `graphfl_lab`, `run_vision_*`, `vision_suite_*`, `graph_filter_strength` | **canonical** |
| `configs/general/...`, `spectral_filter_strength`, `spectral_filtered_*` inputs | **read-only alias** only |

## Gate 1 Inventory Scope (historical snapshot)

The **Current action** column records Gate 0–1 intent. For live policy after
Gate 6 + Phase 2, use the Post-Gate-6 table at the top of this file.

| Pattern | Category | Gate 1 action (historical) | Status on `main` now |
|---|---|---|---|
| `spectral_fl` | package identity | shim until Gate 6 | **removed** |
| `graphfl_lab` | package identity | planned canonical | **canonical** |
| `run_general_*` | old public runner | keep until Gate 6 | **removed** → `run_vision_*` |
| `run_vision_*` | vision runner | Gate 4b rewire | **canonical** |
| `run_graph_ablation.py` | Cora ablation | thin wrapper | **canonical** |
| `configs/general/...` | old config path | alias until Gate 6 | **read-only alias** |
| `result_general_*` | old result filename | mirror + readers | **removed** from readers |
| `general_suite_*` | old suite artifact | mirror + readers | **removed** from readers |
| `result_vision_*`, `vision_suite_*` | canonical outputs | prefer in new work | **canonical** |
| `graph_filter_strength` | canonical key | prefer in new writes | **canonical** |
| `spectral_filter_strength` | old key | schema policy | **JSON read alias only** |
| `spectral_filtered_*` | old target spelling | Gate 3/6 migration | **input alias only** |
| `ours_spectral_filtered_*` | old suite token | deprecation | **reporting tags only** |
| `_spectral_only`, `_speconly` | old suffixes | deprecation | parsed as legacy suffix aliases |

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
