# Configs

Canonical config tree:

```text
configs/vision/
```

Compatibility path:

```text
configs/general/... -> configs/vision/... via graphfl_lab.config_io.resolve_config_path
```

Use the narrowest current track/question folder:

```text
configs/vision/diagnostic/smoke/
configs/vision/diagnostic/core/
configs/vision/diagnostic/extend/
configs/vision/probes/
configs/vision/stress/
configs/vision/sweeps/
configs/cora/ablations/
```

Suite and single-run outputs under `experiments_current/` use `vision_suite_*`
and `result_vision_*` filenames. Historical `general_suite_*` and
`result_general_*` artifact names are tombstoned after Gate 6. See
`docs/removed-materials.md`.
