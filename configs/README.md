# Config 안내

`configs/`는 실험 preset을 모아 둔 곳이다. Vision 실험은 `configs/vision/`, Cora 실험은 `configs/cora/`를 사용한다.

Config를 수정한 뒤에는 먼저 `--dry-run`으로 읽기와 component 연결이 맞는지 확인한다.

## 구조

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

## Vision Config

| 경로 | 용도 | 대표 config |
|---|---|---|
| `configs/vision/smoke/` | 빠른 single/suite 확인 | `default_similarity_knn.json`, `extension.json` |
| `configs/vision/diagnostic/smoke/` | 짧은 diagnostic 확인 | `fashionmnist_n5_r3_seed42.json` |
| `configs/vision/diagnostic/core/` | 기본 diagnostic suite | `fashionmnist_n20_alpha003_seeds5.json` |
| `configs/vision/diagnostic/extend/` | 확장 diagnostic config | `fashionmnist_n50_alpha003_seeds3.json` |
| `configs/vision/probes/` | graph/source/target probe | `graph_source/layer_slice.json` |
| `configs/vision/stress/` | Non-IID stress configs | `fedavg_collapse/stress_grid_fedavg_collapse.json` |
| `configs/vision/sweeps/` | sweep configs | `client_count/client_count_warmup3_r10.json` |

## Cora

| 경로 | 용도 | 대표 config |
|---|---|---|
| `configs/cora/ablations/graph/` | Cora graph ablation | `graph_ablation_smoke.json` |

## 대표 실행

```powershell
graphfl run single --track vision --config configs/vision/smoke/default_similarity_knn.json --dry-run
graphfl run suite --config configs/vision/smoke/extension.json --dry-run
graphfl run ablation --config configs/cora/ablations/graph/graph_ablation_smoke.json --dry-run
graphfl run stress --config configs/vision/stress/fedavg_collapse/stress_grid_fedavg_collapse.json --dry-run
graphfl run client-count --config configs/vision/sweeps/client_count/client_count_warmup3_r10.json --dry-run
```

문제가 없으면 `--dry-run`을 제거해 실제 실험을 실행한다.

## 호환 경로

| 이전 이름 | 현재 처리 | 용도 |
|---|---|---|
| `configs/general/...` | `configs/vision/...` alias | 예전 config 읽기 |
| `general_suite_*` | `vision_suite_*` alias | 예전 suite artifact 읽기 |
| `result_general_*` | `result_vision_*` alias | 예전 result artifact 읽기 |

세부 정책은 [docs/maintenance.md](../docs/maintenance.md)에 있다.
