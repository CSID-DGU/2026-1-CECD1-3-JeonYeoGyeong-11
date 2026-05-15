# Configs

`configs/vision/` is the canonical config tree for current torchvision/FashionMNIST/MNIST/CIFAR graph-FL diagnostics.

`configs/general/...` is no longer a real config tree. Existing commands that still pass old `configs/general/...` paths are handled by `spectral_fl.config_io.resolve_config_path`, which maps them to the matching `configs/vision/...` file when the old path is missing.

Keep new configs under the narrowest current track and question folder, for example:

```text
configs/vision/diagnostic/smoke/
configs/vision/diagnostic/core/
configs/vision/diagnostic/extend/
configs/vision/probes/
configs/vision/stress/
configs/vision/sweeps/
configs/cora/ablations/
```
