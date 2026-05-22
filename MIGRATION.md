# GraphFL Lab Migration Guide

This guide tracks the staged migration from the historical `spectral_fl`
identity to the planned `graphfl_lab` identity. The migration is not complete
yet; follow `docs/maintenance/cleanup-status.md` for current gate status.

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
spectral_fl
run_general_*
configs/general/...
result_general_*
general_suite_*
spectral_filter_strength
spectral_filtered_*
```

Compatibility wrappers (`run_general_*`, `plot_general_convergence.py`,
`merge_general_fedavg_ours.py`, `deep_dive_general.py`) delegate to the
vision-named implementations. Prefer the vision names in new scripts and docs.

## Output Artifacts

New vision runs write canonical filenames and compatibility mirrors:

| Canonical (prefer in new work) | Compatibility mirror (Gate 6 removal) |
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
Gate 3  graphfl_lab package migration with spectral_fl shim
Gate 4  unified runner and nightly
Gate 5  behavior-preserving modularization
Gate 6  hard cleanup and 1.0.0
```

Gate 3 keeps a `spectral_fl` shim until Gate 6. The shim must cover:

```text
DeprecationWarning
GRAPHFL_LAB_SILENCE_DEPRECATION=1
sys.modules alias behavior
pickle round-trip compatibility
```

Gate 6 must provide `scripts/dev/migrate_serialized_objects.py` before removing
old import aliases. Any preserved pickle/checkpoint asset with `spectral_fl.*`
module paths must be migrated before hard cleanup or explicitly declared outside
post-Gate-6 compatibility guarantees.

Migration C5 (public docs/commands) status: largely complete locally — canonical
runners, suite artifact writes, CLI aggregation-target help, and plot/report
readers align with `docs/framework/naming-and-compatibility.md`. Gate 4c remote
nightly evidence and Gate 6 compatibility removal remain open.

## Rollback

The `pre-graphfl-rename` tag must be created after the Gate 0 commit and before
Gate 1 begins. If a later gate needs rollback, revert to the last merged gate
commit or to the release anchor recorded in `docs/removed-materials.md`.
