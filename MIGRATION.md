# GraphFL Lab Migration Guide

This guide tracks the staged migration from the historical `spectral_fl`
identity to the planned `graphfl_lab` identity. The migration is not complete
yet. Gate 6 hard cleanup is complete; see `docs/removed-materials.md` for
tombstones and `docs/maintenance/cleanup-status.md` (closed) for the execution log.

## For Users

Current supported commands continue to work during the deprecation period:

```text
python run_vision_experiment.py --help
python run_vision_suite.py --help
python run_graph_ablation.py --help
python run_experiment.py --help
```

Planned public direction:

```text
python run_experiment.py --track vision ...
python run_experiment.py --track cora ...
```

During deprecation, old names remain available where documented:

```text
configs/general/...
result_general_*   (readers only; new runs no longer write mirrors)
general_suite_*    (readers only; new runs no longer write mirrors)
spectral_filter_strength   (JSON config key alias only; CLI uses --graph-filter-strength)
```

Root `run_general_*` and `plot_general_*` / `merge_general_*` /
`deep_dive_general` script wrappers were removed in Gate 6 batch 3. Prefer
`run_vision_*` and vision-named report/analysis scripts in new work.

## Output Artifacts

New vision runs write canonical filenames only (Gate 6 batch 2). Readers still
resolve legacy paths for older experiment directories:

| Canonical (new runs) | Legacy (read-only) |
|---|---|
| `result_vision_*.json` | `result_general_*.json` |
| `vision_suite_summary.json` / `.csv` / `.md` | `general_suite_summary.*` |
| `vision_suite_rows.json` | `general_suite_rows.json` |

Readers and suite helpers resolve either family, preferring canonical paths when
both exist. See `graphfl_lab/experiments/suites/vision/artifacts.py` and
`variant_helpers.resolve_result_path_for_variant`.

Single-run JSON may also record:

```text
meta.canonical_output_path
meta.compatibility_output_path
```

## Reporting Commands

Canonical report and analysis entrypoints:

```text
scripts/reports/plot_vision_convergence.py
scripts/reports/generate_dashboard_mockup.py
scripts/analysis/deep_dive_vision.py
scripts/analysis/merge_vision_fedavg_ours.py
```

Legacy `*_general_*` script names remain as thin wrappers until Gate 6.

## Schema Fields

New results will eventually include:

```text
result_schema_version
config_aliases_used
unsupported_components
```

Results without `result_schema_version` are treated as `v0` and should remain
readable with warnings during the compatibility period.

## For Maintainers

Do not start a rename by editing imports directly. Start from Gate 0 and keep
the repository resumable:

```text
python scripts/dev/run.py gate-check 0
```

Gate sequence:

```text
Gate 0  workspace/status/check contract
Gate 1  inventory
Gate 2  schema/config contract
Gate 3  graphfl_lab package migration (`spectral_fl` shim removed in Gate 6 batch 6)
Gate 4  unified runner and nightly
Gate 5  behavior-preserving modularization
Gate 6  hard cleanup and 1.0.0
```

Gate 6 batch 6 removed the `spectral_fl` import shim. Use `graphfl_lab` for all
new imports. Run `scripts/dev/migrate_serialized_objects.py` before loading
external pickle assets; tracked repo assets had no legacy module-path markers.

Migration C5 (public docs/commands) status: complete on `main`. Gate 4c nightly
evidence is recorded in `docs/maintenance/last_nightly_run.json`. Gate 6
compatibility removal is documented in `docs/maintenance/gate-6-prep.md`; seven
consecutive nightly runs are optional strict mode, not a hard prerequisite.

## Rollback

The `pre-graphfl-rename` tag must be created after the Gate 0 commit and before
Gate 1 begins. If a later gate needs rollback, revert to the last merged gate
commit or to the release anchor recorded in `docs/removed-materials.md`.
