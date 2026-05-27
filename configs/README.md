# Config

`configs/`는 tracked experiment preset의 canonical 위치다. Vision track은 `configs/vision/`, Cora graph ablation은 `configs/cora/`를 사용한다.

## Canonical Tree

```text
configs/
├── vision/
│   ├── baselines/
│   ├── diagnostic/
│   │   ├── smoke/
│   │   ├── core/
│   │   └── extend/
│   ├── probes/
│   ├── smoke/
│   ├── stress/
│   └── sweeps/
└── cora/
    └── ablations/
```

## Vision Config Groups

| Folder | Purpose |
|---|---|
| `configs/vision/diagnostic/smoke/` | quick diagnostic smoke |
| `configs/vision/diagnostic/core/` | core diagnostic suite |
| `configs/vision/diagnostic/extend/` | extension diagnostic configs |
| `configs/vision/probes/` | graph/source/target probes |
| `configs/vision/stress/` | Non-IID stress configs |
| `configs/vision/sweeps/` | sweep configs |

## Cora Config Groups

| Folder | Purpose |
|---|---|
| `configs/cora/ablations/` | Cora graph ablation |

## Compatibility

| Legacy | Current | Role |
|---|---|---|
| `configs/general/...` | `configs/vision/...` path alias | old path read support |
| `general_suite_*` | `vision_suite_*` | old suite artifact parse support |
| `result_general_*` | `result_vision_*` | old result artifact parse support |

상세 compatibility policy는 `docs/maintenance.md`에서 관리한다.
