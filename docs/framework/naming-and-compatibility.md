# Naming And Compatibility

## Canonical Names

Use canonical names in new code, docs, configs, and reports.

| Concept | Canonical |
|---|---|
| Package root | `graphfl_lab` |
| Vision experiment implementation | `graphfl_lab/experiments/vision/` |
| Vision suite grammar/reporting | `graphfl_lab/experiments/suites/vision/` |
| Vision CLI parser modules | `graphfl_lab/cli/vision_*.py` |
| Vision launchers | `run_vision_*.py` |
| Framework docs | `docs/framework/` |
| Research notes | `docs/research/` |
| Archive | `docs/archive/` |
| Graph-FL strategy runtime | `graphfl_lab/strategies/graphfl/` |
| Graph-FL runtime class | `GraphFLDiagnosticStrategy` |
| Aggregation targets | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| Filter strength key | `graph_filter_strength` |
| Suite token family | `ours_graph_filtered_*` |
| Filter-only suffix | `_graph_filter_only` |
| Validation scripts | `scripts/checks/` |
| Smoke scripts | `scripts/smoke/` |
| Historical analysis scripts | `scripts/archive/legacy-analysis/` |

## Compatibility Names

Keep compatibility names until their migration gates are satisfied.

| Compatibility name | Keep because | Target |
|---|---|---|
| `run_general_*.py` | old entrypoints | `run_vision_*.py` |
| `graphfl_lab/cli/general_*.py` | old imports | `graphfl_lab/cli/vision_*.py` |
| `graphfl_lab/experiments/general/` | old imports | `graphfl_lab/experiments/vision/` wrappers |
| `graphfl_lab/experiments/suites/general/` | old imports | `graphfl_lab/experiments/suites/vision/` wrappers |
| root `general_*` modules | boundary tests and old imports | package-level vision modules |
| `plot_general_convergence.py` | old report command | `plot_vision_convergence.py` |
| `deep_dive_general.py`, `merge_general_fedavg_ours.py` | old analysis command | vision-named helpers |
| `result_general_*.json` | historical outputs | `result_vision_*` |
| `general_suite_summary.*`, `general_suite_rows.json` | historical outputs | `vision_suite_*` |
| `configs/general/...` | old config paths | `configs/vision/...` |
| `experiments_current/` | local output root | consider `outputs/` later |
| `graphfl_lab/strategies/spectral/` | old strategy imports | `graphfl_lab/strategies/graphfl/` wrappers |
| `SpectralConflictAwareStrategy` | old class import | `GraphFLDiagnosticStrategy` |
| `spectral_fl` package root | old imports and pickled module paths | `graphfl_lab` shim until Gate 6 |
| `spectral_filtered_*` | historical configs/results | `graph_filtered_*` |
| `spectral_filter_strength` | historical config/result key | `graph_filter_strength` |
| `ours_spectral_filtered_*` | historical suite tokens | `ours_graph_filtered_*` |
| `_spectral_only`, `_speconly` | historical variant suffixes | `_graph_filter_only` |

## Planned Breaking Migrations

| Debt | Target | Status |
|---|---|---|
| `spectral_filter_strength` | `graph_filter_strength` | canonical implemented; old field retained |
| `ours_spectral_filtered_*` | `ours_graph_filtered_*` | canonical implemented; aliases retained |
| `_spectral_only` / `_speconly` | `_graph_filter_only` | canonical implemented; aliases retained |
| `spectral_filtered_*` operator outputs | `graph_filtered_*` | new outputs use canonical labels; input aliases retained |
| `spectral_fl` package root | `graphfl_lab` | canonical package implemented; shim retained until Gate 6 |

Recommended order:

```text
1. completed: graph_filter_strength
2. completed: ours_graph_filtered_*
3. completed: graphfl_lab package alias
4. completed: real package move
5. next: spectral_filtered_* lower-level compatibility removal after Gate 6 entry
```

## Future Rename Rule

Migration-required surfaces:

```text
public CLI names
config paths
generated artifact names
external import paths
```

Rename steps:

```text
1. Add canonical name.
2. Keep old name as wrapper/alias.
3. Update active docs/configs.
4. Run full tests and diagnostic preflight.
5. Remove old name after internal references and documented run paths move.
```

Generated cache:

```text
__pycache__/ can be deleted in a dedicated cleanup pass.
```
