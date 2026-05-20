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

## Rollback

The `pre-graphfl-rename` tag must be created after the Gate 0 commit and before
Gate 1 begins. If a later gate needs rollback, revert to the last merged gate
commit or to the release anchor recorded in `docs/removed-materials.md`.
