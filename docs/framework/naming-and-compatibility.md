# Naming And Compatibility

This repository is now centered on a composable graph-FL diagnostic framework.
Some older names are intentionally retained as compatibility surfaces because
they are embedded in CLI entrypoints, config files, generated artifact names, or
historical reports.

## Canonical Names

Use these names for new implementation work.

| Concept | Canonical location/name |
|---|---|
| Vision/Torchvision experiment implementation | `spectral_fl/experiments/vision/` |
| Vision suite grammar/reporting | `spectral_fl/experiments/suites/vision/` |
| Vision CLI parser modules | `spectral_fl/cli/vision_*.py` |
| Vision launchers | `run_vision_*.py` |
| Framework docs | `docs/framework/` |
| Research notes | `docs/research/` |
| Completed migration and previous-direction material | `docs/archive/` |
| Graph-FL strategy runtime | `spectral_fl/strategies/graphfl/` |
| Graph-FL runtime class | `GraphFLDiagnosticStrategy` |
| Graph low-pass aggregation targets | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| Graph-filter strength config/result key | `graph_filter_strength` |
| Graph-filter suite token family | `ours_graph_filtered_*` |
| Graph-filter-only suite suffix | `_graph_filter_only` |
| Current validation scripts | `scripts/checks/` |
| Executable smoke scripts | `scripts/smoke/` |
| Historical analysis scripts | `scripts/archive/legacy-analysis/` |

## Compatibility Names To Keep For Now

Do not remove or rename these without a dedicated migration pass.

| Compatibility name | Why it remains | Future cleanup |
|---|---|---|
| `run_general_*.py` | Existing docs, scripts, and users may still call these entrypoints. They now wrap `run_vision_*.py` behavior. | Remove only after a deprecation window and README/config updates. |
| `spectral_fl/cli/general_*.py` | Import compatibility for old callers. | Keep thin wrappers until no internal or external references remain. |
| `spectral_fl/experiments/general/` | Import compatibility for old module paths. | Keep as wrappers around `experiments/vision/`. |
| `spectral_fl/experiments/suites/general/` | Compatibility for older suite grammar imports. | Keep as wrappers around `experiments/suites/vision/`. |
| `spectral_fl/general_client.py`, `general_data.py`, `general_models.py`, `general_suite_variants.py` | Root-level compatibility facades tested by `tests/structure/test_boundaries.py`. | Replace user-facing references with package-level `clients/vision.py`, `data/vision.py`, `models/vision.py`, and `experiments/suites/vision/`. |
| `scripts/reports/plot_general_convergence.py` | Older report commands may call this script directly. It now wraps `plot_vision_convergence.py`. | Prefer `scripts/reports/plot_vision_convergence.py` in new docs and commands. |
| `scripts/analysis/deep_dive_general.py`, `merge_general_fedavg_ours.py` | Older analysis commands may call these scripts directly. They now wrap vision-named helpers. | Prefer `deep_dive_vision.py` and `merge_vision_fedavg_ours.py`. |
| `result_general_*.json` | Existing suite merge/report scripts and prior experiment outputs depend on this artifact prefix. New runs also write `result_vision_*.json`. | Prefer `result_vision_*` in new tools; remove old names only after historical readers migrate. |
| `general_suite_summary.*`, `general_suite_rows.json` | Report scripts and historical outputs depend on these filenames. New suites also write `vision_suite_summary.*` and `vision_suite_rows.json`. | Prefer `vision_suite_*` in new tools; keep old names until report migration is complete. |
| `configs/general/...` config paths | Existing commands may still pass old config paths. `config_io.resolve_config_path` maps missing `configs/general/...` files to `configs/vision/...`. | Remove the alias only after external docs and user commands have moved to `configs/vision/`. |
| `experiments_current/` | Many configs and local outputs use this ignored output root. | Consider `outputs/` only with config migration and `.gitignore` update. |
| `spectral_fl/strategies/spectral/` | Older imports may still reference this path. It now re-exports `spectral_fl/strategies/graphfl/`. | Keep as a thin wrapper until old imports disappear from downstream scripts. |
| `SpectralConflictAwareStrategy` | Older imports may still reference this class. It is now an alias for `GraphFLDiagnosticStrategy`. | Prefer `GraphFLDiagnosticStrategy` in new code; remove the alias only in a dedicated breaking-change migration. |
| `spectral_fl` package root | The package name is embedded in every import and Flower app entrypoint. | Plan separately as a high-risk package migration. |
| `spectral_filtered_*` aggregation targets | These names are embedded in configs/results. `graph_filtered_*` aliases are now accepted for new commands. | Prefer `graph_filtered_*` in new docs; remove old target spelling only after configs/results migrate. |
| `spectral_filter_strength` | Historical config/result key. New code writes and reads `graph_filter_strength` first. | Keep as a compatibility field until historical result-reader policy is frozen. |
| `ours_spectral_filtered_*` | Historical suite token family. New active configs use `ours_graph_filtered_*`. | Keep as parser/reporting aliases until old result reuse policy is frozen. |
| `_spectral_only`, `_speconly` suffixes | Historical variant suffixes for disabling conflict floor and server momentum. New configs use `_graph_filter_only`. | Keep as parser aliases until old result reuse policy is frozen. |

## Planned Breaking Migrations

These are not permanent names. They are compatibility debt with a staged
cleanup plan in [cleanup-plan.md](cleanup-plan.md).

| Debt | Target | Required staging |
|---|---|---|
| `spectral_filter_strength` result/meta key | `graph_filter_strength` | canonical key is implemented; remove old field only after historical reader policy is clear |
| `ours_spectral_filtered_*` suite tokens | `ours_graph_filtered_*` | canonical tokens are implemented; remove old parser/reporting aliases only after old result reuse policy is clear |
| `_spectral_only` / `_speconly` suffixes | `_graph_filter_only` | canonical suffix is implemented; remove old suffix aliases only after active docs/results have moved |
| `spectral_filtered_*` lower-level operator outputs | `graph_filtered_*` | aliases already exist; changing canonical lifecycle/design output names needs a separate test-heavy pass |
| `spectral_fl` top-level package | `graphfl_lab` | add alias package first, migrate imports in batches, move implementation later, keep `spectral_fl` wrappers until all active imports are gone |

Recommended order:

1. completed: `graph_filter_strength`
2. completed: `ours_graph_filtered_*`
3. next: `spectral_filtered_*` lower-level operator outputs where safe
4. `graphfl_lab` package alias
5. real package move

## Non-Source Cleanup Notes

- `__pycache__/` directories are generated Python bytecode caches. They are not
  source and can be deleted in a dedicated cleanup pass; they may reappear after
  running tests or scripts.

## Rule For Future Renames

If a rename touches public CLI names, config paths, generated artifact names,
or import paths used outside one package, treat it as a migration:

1. Add the new canonical name.
2. Keep the old name as a thin compatibility wrapper.
3. Update active README/docs/configs to prefer the new name.
4. Run full tests and the diagnostic preflight.
5. Remove the old name only after all internal references and documented run
   paths have moved.

This keeps the repository clean for current work without breaking old results
or scripts that still need to be reproducible.
