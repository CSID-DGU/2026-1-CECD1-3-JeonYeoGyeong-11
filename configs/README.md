# Configs

Canonical config tree:

```text
configs/vision/
```

Compatibility path:

```text
configs/general/... -> configs/vision/... via spectral_fl.config_io.resolve_config_path
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
