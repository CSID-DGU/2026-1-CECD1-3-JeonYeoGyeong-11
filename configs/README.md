# Config

## Canonical Tree

```text
configs/vision/
configs/cora/
```

## Vision Config Group

| Folder | 용도 |
|---|---|
| `configs/vision/diagnostic/smoke/` | 빠른 diagnostic smoke |
| `configs/vision/diagnostic/core/` | 핵심 diagnostic suite |
| `configs/vision/diagnostic/extend/` | 확장 diagnostic 설정 |
| `configs/vision/probes/` | graph/source/target probe |
| `configs/vision/stress/` | Non-IID stress 설정 |
| `configs/vision/sweeps/` | sweep 설정 |

## Cora Config Group

| Folder | 용도 |
|---|---|
| `configs/cora/ablations/` | Cora graph ablation |

## Compatibility

| Legacy | Current |
|---|---|
| `configs/general/...` | `configs/vision/...` path alias |
| `general_suite_*` | `vision_suite_*` |
| `result_general_*` | `result_vision_*` |

세부 compatibility 정책은 `docs/removed-materials.md`에 기록한다.
