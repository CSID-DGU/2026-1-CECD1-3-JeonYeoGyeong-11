# Graph-FL Design Lab

조립식 graph-based federated learning 실험 프레임워크.

이 저장소는 좋은 그래프 알고리즘 하나를 주장하기보다, FL round 안에서
client state, relation, topology, aggregation, diagnostics를 부품처럼 바꾸며
graph correction의 효과가 어디서 나오는지 검증하기 위한 코드다.

기본 실행 대상은 vision FL. 옛 `general`/`spectral` 이름은 호환 경로다.

## Quick Start

Run commands from the repository root.

### 1. Environment

```text
Python 3.11
pip
PowerShell or bash
```

Windows PowerShell:

```powershell
cd graph-fl-design-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

macOS/Linux:

```bash
cd graph-fl-design-lab
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
```

Existing local environment check:

```powershell
D:\jongseol\.venv311\Scripts\python.exe --version
```

### 2. Verify The Install

```powershell
python -m unittest discover -s tests
python run_vision_experiment.py --help
python run_vision_suite.py --help
```

Without activation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

### 3. Run The Smallest Graph Smoke

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
```

First run downloads torchvision data under `data/torchvision/`.
Result path: `experiments_current/default_similarity_knn_smoke/`.

Expected core metadata:

```text
graph_method=default_similarity_knn
graph_design=default_similarity_knn
graph_source=update
graph_mode=rbf_knn
aggregation_target=graph_filtered_update
graph_empty=false
```

### 4. Run A Suite Path Check

```powershell
python scripts/checks/diagnostic_suite_preflight.py
```

```powershell
python run_vision_suite.py --config configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json
```

### 5. Try A Custom Assembly

```powershell
python run_vision_experiment.py `
  --method ours `
  --dataset fashionmnist `
  --model mlp `
  --num-clients 2 `
  --rounds 1 `
  --train-subset-size 20 `
  --test-subset-size 20 `
  --graph-method default_similarity_knn `
  --knn-k 1 `
  --out-dir experiments_current/manual_default_graph `
  --run-tag manual_k1
```

New relation/topology path:
`register_graph_builder(...)` -> `--graph-plugin` -> `--graph-mode` ->
`--aggregation-target`. See
[docs/framework/extension-guide.md](docs/framework/extension-guide.md).

## Overview

| 질문 | 구현 방향 |
|---|---|
| client 사이 관계를 어떤 state에서 만들 것인가? | `graph_source`로 update, EMA update, weight, classifier head 등을 선택 |
| relation을 어떤 graph로 바꿀 것인가? | `graph_mode`와 graph builder registry로 dense, kNN, RBF, QP proxy 등을 선택 |
| graph를 실제 어디에 적용할 것인가? | `aggregation_target`으로 update, EMA update, weight aggregation을 분리 |
| 완성된 조합을 어떻게 다시 실행할 것인가? | `GraphFLDesign`과 `--graph-method`로 runnable method/profile을 선택 |
| graph gain이 진짜 relation 때문인가? | random, shuffled, identity, clustering-only, graph-free control로 비교 |

## Main Interfaces

| 인터페이스 | 역할 | 위치 |
|---|---|---|
| `GraphFLDesign` | method를 lifecycle component 조합으로 선언 | `spectral_fl/designs/` |
| `graph_method` | 실행 가능한 method/profile을 CLI에서 선택 | `spectral_fl/graph/method_specs.py`, `spectral_fl/graph/presets.py` |
| `graph_source` | client representation 추출 | `spectral_fl/graph/sources/`, `spectral_fl/graph/signals/` |
| `graph_mode` | relation/topology 생성 | `spectral_fl/graph/registry.py`, `spectral_fl/graph/builders.py` |
| `aggregation_target` | graph가 update, EMA update, weight 중 어디에 붙는지 결정 | `spectral_fl/strategies/graphfl/targets.py` |
| diagnostics | graph density, dominance, alignment, frequency, LOO distortion 기록 | `spectral_fl/diagnostics/`, `spectral_fl/strategies/graphfl/diagnostics.py` |
| suite grammar | 반복 실험 variant token 정의 | `spectral_fl/experiments/suites/vision/variants.py` |

## Assembly Model

그래프 알고리즘은 하나의 큰 strategy를 새로 만드는 방식이 아니라, 아래 부품을
조립해서 만든다.

```text
method profile
  -> client_state          # 어떤 client vector/state를 볼 것인가
  -> relation_estimator    # client 사이 관계를 어떻게 계산할 것인가
  -> topology_operator     # relation을 어떤 graph로 만들 것인가
  -> aggregation_operator  # graph를 update/EMA/weight 중 어디에 적용할 것인가
  -> diagnostics           # graph 효과를 어떤 control과 지표로 검증할 것인가
```

현재 실행 CLI에서는 이 조립이 다음 knob로 내려온다.

| 조립 부품 | 실행 knob | 예시 |
|---|---|---|
| client state | `--graph-source` | `update`, `ema_update`, `classifier_head_update`, `weight` |
| relation/topology | `--graph-mode` | `knn`, `rbf_knn`, `pfedgraph_qp`, custom builder |
| aggregation | `--aggregation-target` | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| method profile | `--graph-method` | `default_similarity_knn`, `pfedgraph`, custom profile |
| exact preset | `--graph-preset` | registered `GraphFLDesign` name or compatibility alias |

예를 들어 기본 그래프 method는 이렇게 조립된다.

```text
default_similarity_knn
  = update client state
  + RBF similarity
  + kNN topology
  + graph-filtered update aggregation
```

## Default Graph Method

대표 기본 알고리즘은 다음 method profile이다.

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
```

내부 조합:

```text
--graph-method default_similarity_knn
  -> graph_source=update
  -> graph_mode=rbf_knn
  -> knn_k=2
  -> graph_scale_sigma=0.0
  -> aggregation_target=graph_filtered_update
```

`--graph-method`는 실행 가능한 method/profile을 고르는 상위 옵션이다.
필요하면 `--knn-k`, `--graph-source`, `--graph-mode`,
`--aggregation-target` 같은 하위 knob를 명시적으로 덮어쓸 수 있다.

`--graph-preset`: 등록된 `GraphFLDesign`을 그대로 적용.

## Adding A Graph Algorithm

새 graph algorithm은 `strategy.py`를 직접 키우지 않고 아래 순서로 붙인다.

1. method profile을 먼저 적는다:
   `client_state`, `relation`, `topology`, `aggregation`, `delivery`,
   `local_objective`, `state_store`, `diagnostics`.
2. support level을 정한다:
   `core-supported`, `proxy-supported`, `interface-target`, `out-of-scope`.
3. 새 client representation이 필요할 때만 `graph_source`를 추가한다.
4. 새 relation/topology가 필요할 때만 `graph_mode` 또는 graph builder를 추가한다.
5. runnable 조합이면 `GraphFLDesign`과 `--graph-method`로 노출한다.
6. suite token이나 config는 lower-level path가 검증된 뒤 추가한다.
7. shape, determinism, metadata, diagnostics, control 비교 테스트를 추가한다.

최소 graph builder 예시:

```python
from spectral_fl.graph import GraphBuildContext, register_graph_builder


@register_graph_builder("my_relation_graph")
def build_my_relation_graph(context: GraphBuildContext):
    z = context.z_mat
    adj = z @ z.T
    adj[adj < 0.0] = 0.0
    return adj, {"base_graph_kind": "my_relation_graph"}
```

실행 예시:

```powershell
python run_vision_experiment.py `
  --method ours `
  --graph-plugin my_project.graph_plugins.my_method `
  --graph-method my_method `
  --graph-source classifier_head_update `
  --graph-mode my_relation_graph `
  --aggregation-target graph_filtered_update
```

## Repository Layout

```text
graph-fl-design-lab/
├── README.md                         # 프로젝트 개요와 기본 실행 경로
├── pyproject.toml                    # Flower App component/config 정의
├── requirements.txt                  # dependency 범위
│
├── configs/                          # 반복 실험 JSON config
│   ├── README.md                     # config namespace 설명
│   ├── vision/                       # 현재 canonical vision config
│   │   ├── smoke/                    # 가장 작은 실행성 확인
│   │   ├── diagnostic/               # claim/control 진단 suite
│   │   ├── baselines/                # baseline smoke
│   │   ├── probes/                   # frequency, graph_source, structure, tau probe
│   │   ├── stress/                   # stress grid
│   │   └── sweeps/                   # client-count sweep
│   └── cora/                         # Cora/FGL ablation config
│
├── run_vision_experiment.py          # vision 단일 experiment 실행
├── run_vision_suite.py               # vision variant/seed suite 실행
├── run_vision_client_count_sweep.py  # client 수 sweep
├── run_vision_stress_grid.py         # stress grid 실행
├── run_general_*.py                  # legacy compatibility wrappers
│
├── spectral_fl/
│   ├── app/                          # Flower app config/runtime glue
│   ├── cli/                          # argparse parser only
│   ├── clients/                      # Flower client 구현
│   ├── data/                         # dataset loading/partitioning
│   ├── designs/                      # GraphFLDesign composer/registry/presets
│   ├── diagnostics/                  # schema, metrics, CSV/JSONL writer
│   ├── graph/                        # client relation graph 생성
│   │   ├── method_specs.py           # prior-work/method support metadata
│   │   ├── registry.py               # pluggable graph builder registry
│   │   ├── builders.py               # source, relation, sparsification 연결
│   │   ├── sources/                  # graph_source option/config
│   │   ├── signals/                  # client state extraction
│   │   ├── similarity/               # pairwise relation score
│   │   └── sparsification.py         # dense/kNN/random/threshold/uniform rules
│   ├── lifecycle/                    # component contracts, traces, state, counterfactuals
│   ├── strategies/
│   │   ├── graphfl/                  # graph-FL aggregation runtime
│   │   ├── baselines/                # FedAvg/FedOpt/FedSim/etc.
│   │   └── spectral/                 # old import compatibility
│   └── experiments/
│       ├── vision/                   # single run, suite, stress, sweep orchestration
│       ├── cora/                     # Cora/FGL execution path
│       ├── general/                  # old module compatibility
│       └── suites/vision/            # variant grammar and reporting
│
├── scripts/
│   ├── checks/                       # non-training validation
│   ├── smoke/                        # executable smoke runs
│   ├── reports/                      # plotting/dashboard helpers
│   └── analysis/                     # analysis helpers and legacy wrappers
│
├── docs/
│   ├── README.md                     # docs index
│   ├── structure.md                  # responsibility boundary and routing rules
│   ├── framework/                    # active framework docs
│   ├── research/                     # literature/design notes
│   └── archive/                      # previous direction and migration history
│
├── tests/                            # unit, structure, suite, graph, strategy tests
├── data/                             # local dataset cache, Git ignored
└── experiments_current/              # local experiment outputs, Git ignored
```

더 자세한 책임 경계: [docs/structure.md](docs/structure.md)

## Documents

| 문서 | 내용 |
|---|---|
| [docs/README.md](docs/README.md) | 문서 전체 index |
| [docs/structure.md](docs/structure.md) | 폴더 구조, 변경 위치, compatibility facade 규칙 |
| [docs/framework/interfaces.md](docs/framework/interfaces.md) | 조립식 graph algorithm 구현 인터페이스 |
| [docs/framework/extension-guide.md](docs/framework/extension-guide.md) | 새 graph source/builder 추가 방법 |
| [docs/framework/prior-work-mapping.md](docs/framework/prior-work-mapping.md) | 선행연구 exact/proxy/interface 경계 |
| [docs/framework/diagnostics.md](docs/framework/diagnostics.md) | 진단 지표 해석 규칙 |
| [docs/framework/naming-and-compatibility.md](docs/framework/naming-and-compatibility.md) | 남겨둔 옛 이름과 정리 계획 |
| [docs/framework/project-prompt.md](docs/framework/project-prompt.md) | 이후 작업자/agent에게 넘길 전체 프롬프트 |

## Execution Flow

기본 실행 구조:

```text
run_vision_suite.py
  -> run_vision_experiment.py
    -> spectral_fl/flower_runner.py
      -> spectral_fl/flower_app.py
        -> spectral_fl/strategies/graphfl/strategy.py
          -> graph source / graph builder / filtering / aggregation / diagnostics
```

대표 실행 경로:

```powershell
python run_vision_experiment.py --help
python run_vision_suite.py --help
python run_vision_client_count_sweep.py --help
python run_vision_stress_grid.py --help
```

## Smoke Runs

가장 작은 default graph method smoke:

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
```

diagnostic suite launch preflight:

```powershell
python scripts/checks/diagnostic_suite_preflight.py
```

작은 diagnostic suite:

```powershell
python run_vision_suite.py --config configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json
```

prior-work proxy assembly smoke:

```powershell
python scripts/smoke/prior_work_proxy.py
```

## Verification

```powershell
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
python scripts/checks/prior_work_proxy_parity.py --summary experiments_current/prior_work_proxy_smoke/<stamp>/prior_work_proxy_summary.json
```

`diagnostic_suite_preflight.py`: no-training suite launch/config check.
`scripts/smoke/prior_work_proxy.py`: executable small Flower smoke.

## Compatibility

새 코드의 기준 이름은 `vision`, `graphfl`, `graph_filtered_*`다.

| 옛 이름 | 현재 역할 |
|---|---|
| `run_general_*.py` | `run_vision_*.py`로 가는 compatibility wrapper |
| `spectral_fl/experiments/general/` | old import path wrapper |
| `spectral_fl/strategies/spectral/` | `strategies/graphfl/` wrapper |
| `spectral_filtered_*`, `spectral_filter_strength` | 기존 config/result 호환 alias |

새 로직은 compatibility path에 추가하지 않는다.

## Git Policy

| 항목 | Git 포함 여부 | 비고 |
|---|---|---|
| source code | 포함 | `spectral_fl/`, runner, scripts |
| docs | 포함 | `README.md`, `docs/*.md` |
| configs | 포함 | `configs/**/*.json` |
| tests and CI | 포함 | `tests/`, `.github/` |
| dependency metadata | 포함 | `requirements.txt`, `pyproject.toml` |
| dataset cache | 제외 | `data/` |
| generated output | 제외 | `experiments_current/`, `reports/`, `outputs/`, `runs/` |
| local environment | 제외 | `.venv/`, `.venv311/`, editor cache |
