# Graph-FL Design Lab

Graph-FL Design Lab은 client update와 model state로 client graph를 만들고, matched control graph와 diagnostic metric으로 Graph-FL gain의 원인을 분해하는 실험 framework다.

핵심 질문:

```text
Graph-FL gain이 실제 client relation structure에서 오는가,
아니면 dominance, norm, smoothing, optimizer effect로 설명되는가?
```

## 핵심 구조

| Layer | 역할 |
|---|---|
| `graph_source` | client를 update, weight, EMA, classifier head 등으로 표현 |
| `graph_mode` | relation score와 topology 구성 |
| `aggregation_target` | graph를 update, EMA update, weight 등에 적용 |
| `correction_family` | real graph, random, shuffled, uniform, identity, graph-free control 비교 |
| `diagnostics` | alignment, `DI`, `N_eff`, `LOO`, graph metric 기록 |

## Evidence Claim

| Claim | Evidence |
|---|---|
| graph construction 재현성 | `graph_parity_summary.csv`, edge F1, drift |
| paper mechanism 대응 | `external_mechanism_alignment.csv` |
| diagnostic sensitivity | `scenario_manifest.json`, `metric_validity_summary.csv` |
| design-space coverage | `design_space_matrix.csv`, `design_space_summary.csv` |
| extensibility | `extension_contract_summary.csv` |

Framework-quality 근거는 `docs/framework/evidence.md`에 정리한다.

## 문서 지도

| 문서 | 용도 |
|---|---|
| `docs/framework/overview.md` | project claim, research position, experiment design |
| `docs/framework/metrics.md` | metric 정의, 수식, diagnostic 해석 |
| `docs/framework/components.md` | lifecycle, interface, extension guide, prior-work mapping |
| `docs/framework/evidence.md` | framework-quality evidence와 provenance |
| `docs/maintenance/migration-and-compatibility.md` | migration, compatibility, gate-check contract |
| `docs/archive/README.md` | archive summary |

## 설치

Repository root에서 실행한다.

| Step | Command |
|---|---|
| dependency 설치 | `python -m pip install -r requirements.txt` |
| editable install | `python -m pip install -e .` |

## 검증

| Check | Command |
|---|---|
| unit tests | `python -m unittest discover -s tests` |
| vision CLI | `python run_vision_experiment.py --help` |
| suite CLI | `python run_vision_suite.py --help` |
| evidence report | `python scripts/validation/graph_evidence_report.py --profile smoke --include-external --out-dir <out-dir>` |

## 주요 실행 경로

| 목적 | Command |
|---|---|
| vision single run | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| vision suite | `python run_vision_suite.py --config configs/vision/diagnostic/smoke/default.json` |
| Cora graph ablation | `python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json` |

## Repository 구조

```text
graphfl_lab/
├── designs/                 GraphFLDesign registry and presets
├── graph/                   graph source, builder, control, diagnostics
├── lifecycle/               lifecycle contracts and traces
├── strategies/graphfl/      Graph-FL runtime strategy
├── diagnostics/             result schema and artifact writers
└── experiments/             vision and Cora orchestration

configs/
├── vision/
└── cora/

docs/
├── framework/
├── research/
├── maintenance/
└── archive/
```
