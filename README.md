# Graph-FL Design Lab

Graph-FL Design Lab는 Graph-FL에서 보이는 성능 변화가 실제 client relation graph 때문인지, 아니면 smoothing, control, dominance correction, optimizer 같은 다른 효과 때문인지 나눠 보기 위한 실험 코드다.

이 레포는 새 방법 하나를 강하게 주장하기보다, Graph-FL 계열 실험을 같은 구조로 실행하고 비교할 수 있게 만드는 데 초점을 둔다.

## 무엇을 보나

| 질문 | 레포에서 확인하는 방식 |
|---|---|
| client 사이 관계가 의미 있는가 | real graph와 random, shuffled, uniform, identity control 비교 |
| graph 없이도 비슷한 효과가 나는가 | graph-free correction, clustering-only, dominance correction 비교 |
| 어떤 부분이 결과를 움직이는가 | `graph_source`, `graph_builder`, `aggregation_target`을 나눠 기록 |
| 결과를 어떻게 해석할 수 있는가 | `DI`, `N_eff`, alignment, `LOO`, graph statistics 저장 |

기본 흐름은 다음과 같다.

```text
client local training
-> graph_source
-> graph_builder
-> aggregation_target
-> diagnostics / artifacts
```

## 설치

Python 3.11 이상을 사용한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
graphfl --help
```

`graphfl` 명령을 바로 쓸 수 없으면 module 명령을 사용한다.

```powershell
python -m graphfl_lab.cli.main --help
```

## 실행

먼저 `--dry-run`으로 config와 component가 의도대로 연결되는지 확인한다.

```powershell
graphfl run single --track vision --config configs/vision/smoke/default_similarity_knn.json --dry-run
```

`--dry-run`을 빼면 실제 실험을 실행한다.

| 목적 | 명령 |
|---|---|
| Vision single | `graphfl run single --track vision --config configs/vision/smoke/default_similarity_knn.json` |
| Vision suite | `graphfl run suite --config configs/vision/smoke/extension.json` |
| Cora ablation | `graphfl run ablation --config configs/cora/ablations/graph/graph_ablation_smoke.json` |
| Stress grid | `graphfl run stress --config configs/vision/stress/fedavg_collapse/stress_grid_fedavg_collapse.json` |
| Client-count sweep | `graphfl run client-count --config configs/vision/sweeps/client_count/client_count_warmup3_r10.json` |

기존 entrypoint도 유지한다.

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
python run_vision_suite.py --config configs/vision/smoke/extension.json
python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json
```

Config 위치와 용도는 [configs/README.md](configs/README.md)에 정리되어 있다.

## 확장

새로운 graph source, graph builder, aggregation target은 같은 등록 API로 붙인다.

```python
register_graph_source
register_graph_builder
register_aggregation_target
register_design
```

CLI 흐름은 다음 정도만 알면 된다. 자세한 contract는 [docs/framework.md](docs/framework.md)에 있다.

```powershell
graphfl component new <source|builder|aggregation> <name>
graphfl component validate <plugin-path>
graphfl design compose ...
graphfl run single --track vision --config <config.json> --dry-run
```

## 산출물

| 파일 | 내용 |
|---|---|
| `round_metrics.csv` | round별 loss, accuracy, aggregate metric |
| `client_metrics.csv` | client별 contribution, update norm, alignment |
| `graph_stats.csv` | graph density, degree, entropy, spectral metric |
| `counterfactual_metrics.csv` | real graph와 control graph 비교 |
| `module_traces.jsonl` | 사용된 component, parameter, shape, metadata |
| `design_space_matrix.csv` | component 조합별 계산 check |
| `extension_contract_summary.csv` | custom component contract check |

## 검증

검증은 특정 방법의 성능이 항상 좋다는 뜻이 아니다. 코드 구조가 의도대로 이어지고, graph/control/diagnostic을 같은 기준으로 비교할 수 있는지 확인하는 절차다.

```powershell
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
python scripts/validation/graph_evidence_report.py `
  --profile smoke `
  --include-external `
  --out-dir tmp/evidence_smoke
```

최근 확인 기준:

| Check | Result |
|---|---|
| unit tests | 306 / 306 pass |
| component contract | Registry, Shape, Finite, Metadata, Trace/Artifact pass |
| CLI dry-run | 대표 5개 실행 경로 pass |

더 자세한 검증 수치와 한계는 [docs/evidence.md](docs/evidence.md)에 있다.

## 저장소 구조

```text
graphfl_lab/
├── cli/              graphfl 명령과 기존 wrapper 연결
├── designs/          GraphFLDesign preset과 component 조합
├── extensions/       custom component 등록과 검증
├── graph/            graph source, builder, control graph
├── lifecycle/        실행 중 component contract와 trace
├── strategies/       baseline과 Graph-FL strategy
├── diagnostics/      metric schema와 artifact writer
├── experiments/      Vision, Cora 실행 경로
└── validation/       검증 report 생성

configs/              실험 config preset
scripts/              check, report, validation script
tests/                unit / contract test
docs/                 세부 설명과 코드 위치 안내
```

## 문서

| 보고 싶은 내용 | 문서 |
|---|---|
| component 구조와 metric | [docs/framework.md](docs/framework.md) |
| 검증 결과와 한계 | [docs/evidence.md](docs/evidence.md) |
| prior work 대응 | [docs/research.md](docs/research.md) |
| 코드 위치 | [docs/repository.md](docs/repository.md) |
| compatibility | [docs/maintenance.md](docs/maintenance.md) |
| 변경 이력 | [CHANGELOG.md](CHANGELOG.md) |
