# GraphFL Lab Migration Guide

This guide tracks the migration from the historical `spectral_fl` identity to
`graphfl_lab`. **Public rename and Gate 6 cleanup are complete** on `main`
(2026-05-22) and are released as `1.0.0`. A small set of read-only
JSON/config aliases remains for historical artifacts; see `docs/removed-materials.md` and
`docs/framework/naming-and-compatibility.md`. The execution log is closed in
`docs/maintenance/cleanup-status.md`.

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
configs/general/...   (path alias to configs/vision in config loader)
spectral_filter_strength   (JSON config key alias only; CLI uses --graph-filter-strength)
```

Gate 6 and post-Gate-6 cleanup removed `run_general_*`, `graphfl_lab/general_*`,
`general_suite_*` / `result_general_*` code paths. Prefer `run_vision_*`,
`result_vision_*`, and `vision_suite_*` in new work. Old artifact filenames
under local experiment dirs are gitignored, not read by current code.

## Output Artifacts

New vision runs write canonical filenames only (Gate 6 batch 2). Suite helpers
also accept short `suite_*` artifact names when present:

| Canonical (new runs) | Short legacy (read-only) |
|---|---|
| `result_vision_*.json` | — |
| `vision_suite_summary.json` / `.csv` / `.md` | `suite_summary.*` |
| `vision_suite_rows.json` | `suite_rows.json` |

Readers and suite helpers prefer canonical paths when
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
The machine-readable closure check is `python scripts/dev/run.py gate-check 6`.

## Rollback

The `pre-graphfl-rename` tag must be created after the Gate 0 commit and before
Gate 1 begins. If a later gate needs rollback, revert to the last merged gate
commit or to the release anchor recorded in `docs/removed-materials.md`.
