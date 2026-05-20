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
| `result_general_*` | old result filename | keep reader/writer compatibility until Gate 6 |
| `general_suite_*` | old suite artifact filename | keep reader/writer compatibility until Gate 6 |
| `spectral_filter_strength` | old config/result key | schema policy in Gate 2, removal in Gate 6 |
| `spectral_filtered_*` | old aggregation target/result spelling | internal migration in Gate 3, removal in Gate 6 |
| `ours_spectral_filtered_*` | old suite token | keep through deprecation |
| `_spectral_only`, `_speconly` | old suite suffixes | keep through deprecation |

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
