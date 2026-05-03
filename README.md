# Spectral Client Update Graph

Federated Learning의 client update를 graph signal로 보고, shared/disagreement frequency로 분해하는 실험 코드.

본 저장소는 각 client update를 독립 벡터로만 평균하지 않고, **client relation graph 위의 graph signal**로 해석한다. 목적은 즉시 성능 향상을 주장하는 것이 아니라, client update population 안에 존재하는 graph-frequency structure를 관찰하고 검증하는 데 있다.

## Overview

| 질문 | 구현 방향 |
|---|---|
| client들이 비슷한 update 방향을 공유하는가? | update delta로 client relation graph를 구성 |
| graph 위에서 disagreement를 볼 수 있는가? | graph Fourier basis로 low/mid/high-frequency 성분 분해 |
| kNN graph가 random sparse graph와 다른가? | kNN과 density-matched random graph 비교 |
| high-frequency는 noise인가, signal인가? | client-level residual, alpha, label histogram 동시 기록 |

## Documents

| 파일 | 내용 |
|---|---|
| [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) | 연구 가설, graph choice, spectral diagnostic, aggregation 확장 축 |
| [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md) | smoke experiment 결과와 현재 해석 |

raw dataset과 생성된 experiment output은 Git에서 제외한다. 재현이 필요한 경우 로컬에서 실험을 실행해 `data/`와 `experiments_current/`를 생성한다.

## Formulation

round `t`에서 server는 각 client model을 받아 full update delta를 계산한다.

$$
g_i = w_i - w_{\mathrm{global}}
$$

full update는 graph construction과 spectral diagnosis를 위해서만 projection한다.

$$
z_i = R^T g_i,\quad
Z = [z_1; \ldots; z_n],\quad
L = D - W
$$

주요 graph-conditioned diagnostic:

$$
H_{\mathrm{spec}}(Z \mid W)
= \frac{\mathrm{Tr}(Z^T L(W) Z)}{\lVert Z\rVert_F^2}
$$

핵심 분리:

- `z_i`: graph construction, spectral diagnostic, client interpretation에 사용
- `g_i`: 실제 model aggregation에 사용하는 full update
- high-frequency energy: 자동으로 제거해야 할 noise가 아니라 conflict, minority structure, useful heterogeneity 후보

기본 aggregation은 다음 구조를 따른다.

```text
update delta g_i
  -> projection z_i
  -> client relation graph W
  -> graph-spectral residual / conflict score
  -> aggregation weight alpha_i
  -> weighted average of original full update g_i
```

## Implemented Scope

실험 확장을 위해 graph construction representation과 aggregation target은 분리되어 있다. 현재 구현 범위는 다음과 같다.

구현된 선택 축:

| 선택 축 | config/CLI key | 현재 옵션 | 적용 위치 |
|---|---|---|---|
| graph construction representation | `graph_source` | `update`, `normalized_update`, `weight` | client relation graph를 만들 때 사용하는 입력 벡터 |
| aggregation target | `aggregation_target` | `update`, `weight` | server가 global model을 갱신할 때 합치는 대상 |
| graph construction rule | `graph_mode` | 아래 graph mode 표 참고 | 입력 벡터 사이의 edge weight 계산 방식 |

`graph_source` 옵션:

| 옵션 | 의미 |
|---|---|
| `update` | `g_i = w_i - w_global` update delta로 graph 구성 |
| `normalized_update` | update delta를 client별 L2 normalization한 뒤 graph 구성 |
| `weight` | local model weight `w_i` 자체로 graph 구성 |

`aggregation_target` 옵션:

| 옵션 | 의미 |
|---|---|
| `update` | `sum_i alpha_i g_i`를 global model에 더함 |
| `weight` | `sum_i alpha_i w_i`로 global model을 직접 구성 |

`graph_mode` 옵션:

| 옵션 | 의미 |
|---|---|
| `dense` | positive cosine dense graph |
| `knn` | top-k positive cosine graph |
| `mutual_knn` | mutual top-k positive cosine graph |
| `threshold` | cosine threshold graph |
| `random` | kNN edge count matched random graph |
| `uniform` | all off-diagonal edges equal to 1 |
| `magnitude`, `magnitude_aware` | cosine graph down-weighted by client signal magnitude mismatch |
| `global_alignment` | cosine graph weighted by alignment to the mean client signal |

현재 `update`는 raw gradient가 아니라 local training 이후의 model update delta다. 따라서 `g_i`는 pseudo-gradient처럼 해석할 수 있지만, client가 mini-batch/raw gradient tensor를 server에 보내는 구현은 아니다.

아직 구현하지 않은 범위:

| 항목 | 상태 | 필요한 변경 |
|---|---|---|
| raw gradient collection | 미구현 | client가 gradient vector 또는 trajectory summary를 반환하도록 protocol 확장 |
| raw gradient graph | 미구현 | server가 반환된 gradient payload를 parsing하고 graph source로 선택 |
| raw gradient aggregation | 미구현 | gradient를 model parameter update로 적용하는 별도 aggregation path |

## Repository Layout

```text
spectral-client-update-graph/
├── README.md                         # 프로젝트 개요와 실행 경로
├── pyproject.toml                    # Flower App component/config 정의
├── requirements.txt                  # dependency 범위
├── .gitignore                        # data/output/local note 제외 규칙
│
├── configs/                          # 반복 실험용 JSON config
│   ├── general_diagnostic_smoke.json
│   ├── general_frequency_smoke.json
│   ├── general_extension_smoke.json
│   └── cora_graph_ablation_smoke.json
│
├── run_general_suite.py              # General FL suite 실행: variant/seed 반복
├── run_general_experiment.py         # General FL 단일 experiment 실행
├── run_graph_ablation.py             # Cora/FGL graph ablation suite 실행
├── run_experiment.py                 # Cora/FGL 단일 experiment 실행
│
├── spectral_fl/
│   ├── flower_app.py                 # Flower ClientApp/ServerApp entrypoint
│   ├── flower_runner.py              # local App runner와 flwr run helper
│   ├── strategy.py                   # server-side orchestration
│   ├── projection.py                 # full update flattening, random projection
│   ├── update_graph.py               # client relation graph 생성
│   ├── spectral_diagnostics.py       # Laplacian, H_spec, frequency energy 계산
│   ├── aggregation.py                # spectral residual 기반 aggregation weight 계산
│   │
│   ├── general_data.py               # General FL dataset loading/partitioning
│   ├── general_models.py             # General FL model: MLP/CNN
│   ├── general_client.py             # General FL Flower client
│   │
│   ├── data.py                       # Cora/FGL data partitioning
│   ├── model.py                      # Cora/FGL model: GCN
│   └── client.py                     # Cora/FGL Flower client
│
├── scripts/
│   ├── spectral_decomposition_report.py  # result JSON -> round/client frequency CSV
│   ├── deep_dive_seed.py                 # seed 단위 client-level 분석
│   ├── deep_dive_general.py              # General FL 결과 연결
│   ├── merge_general_fedavg_ours.py      # FedAvg/Ours JSON merge helper
│   ├── aggregate_graph_ablation.py       # graph ablation summary 재계산
│   └── print_round_table.py              # result JSON 터미널 표 출력
│
├── docs/
│   ├── EXPERIMENT_DESIGN.md          # 실험 질문과 확장 축
│   └── EXPERIMENT_RESULTS.md         # 실험 결과와 해석 기록
│
├── tests/                            # graph/spectrum/aggregation/config smoke unit test
│
├── .github/
│   └── workflows/ci.yml              # py_compile + unit smoke CI
│
├── data/                             # local-only dataset cache
└── experiments_current/              # local-only experiment output
```

## Environment

권장 환경:

```text
Python 3.11
pip
```

dependency 설치:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m pip install -r requirements.txt
```

## Execution Flow

기본 실행은 Flower `ClientApp`/`ServerApp` 구조를 사용한다. 로컬 script는 같은 App component를 직접 실행해 기존 result JSON 형식을 유지한다.

```text
run_general_suite.py
  -> run_general_experiment.py
    -> spectral_fl/flower_runner.py
      -> spectral_fl/flower_app.py
        -> spectral_fl/strategy.py
          -> projection / update_graph / spectral_diagnostics / aggregation
```

단일 실험 entrypoint:

```text
General FL: python run_general_experiment.py ...
Cora/FGL:   python run_experiment.py ...
```

반복 실험은 `configs/*.json`에 조건을 저장해 실행한다.

```powershell
python run_general_suite.py --config configs/general_diagnostic_smoke.json
```

`pyproject.toml`에는 Flower App component가 정의되어 있어, 같은 코드를 `flwr run` 흐름으로도 연결할 수 있다.

## Smoke Experiments

diagnostic-only run. aggregation은 FedAvg와 동일하게 유지하고 graph/spectral trace만 기록한다.

```powershell
python run_general_suite.py --config configs/general_diagnostic_smoke.json
```

frequency-decomposition run. conservative spectral residual 기반 aggregation을 켜고 low/mid/high graph-frequency structure를 기록한다.

```powershell
python run_general_suite.py --config configs/general_frequency_smoke.json
```

extension-option run. `graph_source`, 신규 graph mode, `aggregation_target` 확장 경로를 작게 확인한다.

```powershell
python run_general_suite.py --config configs/general_extension_smoke.json
```

## Frequency Decomposition

`ours_*` run 이후 round-level/client-level spectral table export:

```powershell
python scripts/spectral_decomposition_report.py `
  --result experiments_current/phaseB_frequency_decomp_smoke_v2/result_general_ours_seed42_phaseB_frequency_decomp_smoke_v2_ours_knn_k2_seed42.json
```

생성 파일:

```text
*_spectral_decomposition/
  round_frequency_decomposition.csv
  client_frequency_decomposition.csv
  README.md
```

## Verification

dataset download 없이 코드 구조를 확인할 때는 unit smoke test를 먼저 실행한다.

```powershell
python -m unittest discover -s tests
```

runner까지 확인할 때는 smoke config 중 하나를 실행한다. GitHub Actions에서는 `py_compile`과 unit smoke test를 수행한다.

## Datasets

| track | dataset | entrypoint |
|---|---|---|
| General FL | `fashionmnist`, `mnist`, `cifar10` | `run_general_experiment.py`, `run_general_suite.py` |
| Cora/FGL | `cora` | `run_experiment.py`, `run_graph_ablation.py` |

dataset download cache는 로컬 `data/` 아래에 저장하며 Git에서는 제외한다.

## Git Policy

| 대상 | Git 포함 여부 | 비고 |
|---|---|---|
| source code | 포함 | `spectral_fl/`, runner, analysis script |
| docs | 포함 | `README.md`, `docs/*.md` |
| experiment configs | 포함 | `configs/*.json` |
| tests and CI | 포함 | `tests/`, `.github/workflows/ci.yml` |
| dependency metadata | 포함 | `requirements.txt`, `pyproject.toml` |
| dataset cache | 제외 | `data/` 아래의 downloaded dataset |
| generated experiment output | 제외 | `experiments_current/`, `outputs/`, `runs/` |
| local environment | 제외 | `.venv/`, editor 설정, 로컬 연구 메모 |
