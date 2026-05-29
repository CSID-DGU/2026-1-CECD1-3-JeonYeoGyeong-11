# Graph-FL Design Lab

Graph-FL Design Lab는 Graph-FL gain이 실제 client relation graph에서 오는지 검증하기 위한 실험 framework다. 이 repository는 federated learning 실험, graph construction, matched controls, graph-free controls, diagnostics, Evidence artifact를 한 구조로 묶어 Graph-FL claim을 재현 가능한 표와 수치로 설명한다.

## Project Overview

Graph-FL 계열 방법은 client 사이의 relation graph를 이용해 aggregation 또는 personalization을 바꾼다. 그러나 성능 향상은 relation graph 자체가 아니라 smoothing, dominance correction, clustering, optimizer 차이에서도 생길 수 있다. 이 repository의 목적은 그 효과를 분리해서 측정하는 것이다.

| 질문 | Repository 기준 답 |
|---|---|
| Graph-FL gain은 어디서 오는가 | `graph_source`, `graph_mode`, `aggregation_target`, `correction_family`를 분리해 비교 |
| real graph가 control보다 의미 있는가 | random, shuffled, uniform, identity, graph-free controls와 같은 artifact row에서 비교 |
| prior work mechanism을 설명할 수 있는가 | FedAMP, SFL, pFedGraph, FedAGA mechanism을 component slot으로 매핑 |
| metric이 해석 가능한가 | `DI`, `N_eff`, alignment, `LOO`, graph stats를 round/client/counterfactual artifact로 기록 |
| framework로 확장 가능한가 | custom source, builder, preset, target이 trace와 artifact contract를 통과하는지 검증 |

핵심 claim:

```text
Graph-FL gain
= relation-specific effect
+ generic smoothing effect
+ clustering effect
+ dominance/norm correction effect
+ optimizer effect
```

## Conclusion Matrix

이 framework의 주장은 “항상 graph가 필요하다”가 아니다. 실험 결론이 어느 쪽으로 나오든 다음 결정을 할 수 있게 만드는 것이 핵심이다.

| 최종 결론 | Evidence pattern | 우리가 할 수 있는 결정 | 남는 framework 가치 |
|---|---|---|---|
| graph가 필수적이다 | real graph가 matched controls보다 일관되게 높고, real-control gap이 alignment/`LOO`/graph stats와 함께 움직임 | `graph_source`, `graph_mode`, `aggregation_target`을 method design 축으로 채택하고 Graph-FL mechanism을 강화 | graph gain을 relation-specific effect로 설명하고 prior work mechanism과 연결 |
| graph가 필수적이지 않다 | graph-free correction, clustering-only, uniform/identity control이 real graph 효과를 대부분 설명 | graph construction을 줄이고 dominance/norm correction 또는 simpler control-based baseline으로 간소화 | Graph-FL gain이 graph 때문이 아니라는 결론을 artifact로 증명 |
| 결론이 setting-dependent다 | seed, alpha, client count, graph_source, graph_mode에 따라 real-control gap이 달라짐 | 조건별로 Graph-FL 적용 구간을 제한하고 Evidence pattern 기준으로 method selection | 애매한 결과를 실패가 아니라 boundary와 decision rule로 정리 |

## Repository Contents

| Area | Role | Main Paths |
|---|---|---|
| Graph-FL runtime | graph-aware aggregation과 diagnostics 실행 | `graphfl_lab/strategies/graphfl/`, `graphfl_lab/lifecycle/` |
| graph construction | client state를 relation graph로 변환 | `graphfl_lab/graph/`, `graphfl_lab/designs/` |
| diagnostics/artifacts | metric schema와 CSV/JSON row 작성 | `graphfl_lab/diagnostics/` |
| experiment tracks | vision Non-IID, Cora graph ablation 실행 | `graphfl_lab/experiments/`, `run_experiment.py` |
| Evidence pack | framework validity 검증 | `graphfl_lab/validation/`, `scripts/validation/` |
| configs | tracked experiment presets | `configs/vision/`, `configs/cora/` |
| tests | component, strategy, experiment, validation contract | `tests/` |
| docs/demo | framework 설명, Evidence, repository layout, HTML demo | `docs/`, `docs/demos/graphfl-assembly-scratch.html` |

## Framework Flow

```text
client local training
├── graph_source              client state -> representation z_i
├── graph_mode                relation score + topology -> adjacency A
├── aggregation_target        graph filtering -> update / EMA update / weight
├── correction_family         real graph vs matched controls
├── diagnostics               DI, N_eff, alignment, LOO, graph stats
└── artifact contract         round/client/graph/counterfactual/Evidence rows
```

| Layer | 역할 |
|---|---|
| `graph_source` | client를 update, weight, EMA update, classifier-head update로 표현 |
| `graph_mode` | relation score와 topology를 adjacency로 구성 |
| `aggregation_target` | graph filtering을 update, EMA update, weight에 적용 |
| `correction_family` | real graph를 random, shuffled, uniform, identity, clustering-only, graph-free control과 비교 |
| `diagnostics` | alignment, `DI`, `N_eff`, `LOO`, graph metric 기록 |

## Evidence Snapshot

| Evidence Axis | Verdict | Primary Artifact |
|---|---|---|
| construction drift | 18 graph modes pass, max abs diff `2.21e-12`, edge F1 `1.0` | `graph_parity_summary.csv` |
| paper-mechanism alignment | pFedGraph, FedAMP, SFL, FedAGA mapping 5 / 5 rows pass | `external_mechanism_alignment.csv` |
| diagnostic sensitivity | 60 framework diagnostic rows pass | `scenario_manifest.json`, `metric_validity_summary.csv` |
| design-space coverage | 16 sources x 18 modes x 5 targets x 6 correction profiles = 8,640 / 8,640 checks pass | `design_space_matrix.csv`, `design_space_summary.csv` |
| extensibility | custom source, builder, preset, target 4 / 4 contract checks pass | `extension_contract_summary.csv` |
| measurement integrity | real/random/uniform measured nonzero, identity expected-zero control | `real_diagnostic_consistency.csv` |

Evidence의 의미:

| Claim | Repository Evidence |
|---|---|
| graph semantics 보존 | lifecycle assembly와 reference builder drift 비교 |
| paper mechanism 대응 | prior work mechanism을 component slot으로 매핑 |
| metric 해석 가능성 | synthetic expected-direction check와 real/control consistency |
| framework 조합성 | built-in design space 전체 row-level calculation check |
| 확장 가능성 | custom component가 trace, metadata, diagnostics, artifact contract를 보존 |

## Primary Artifacts

| Artifact | 내용 |
|---|---|
| `round_metrics.csv` | round-level pre/post aggregate, `DI`, `N_eff`, alignment, `LOO` |
| `client_metrics.csv` | client contribution, update norm, alignment |
| `graph_stats.csv` | density, degree, entropy, spectral graph metrics |
| `counterfactual_metrics.csv` | real graph와 control graph gap |
| `metric_validity_summary.csv` | synthetic expected-direction result |
| `design_space_matrix.csv` | source/mode/target/control/diagnostic row validity |
| `extension_contract_summary.csv` | custom component trace와 artifact preservation |

## Main Run Paths

| 목적 | Command |
|---|---|
| unified runner help | `python run_experiment.py --help` |
| vision single run | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| vision suite | `python run_vision_suite.py --config configs/vision/diagnostic/smoke/default.json` |
| vision stress grid | `python run_vision_stress_grid.py --help` |
| vision client-count sweep | `python run_vision_client_count_sweep.py --help` |
| Cora graph ablation | `python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json` |
| Evidence report | `python scripts/validation/graph_evidence_report.py --profile smoke --include-external --out-dir <out-dir>` |

## Install

Repository root에서 실행한다.

| Step | Command |
|---|---|
| dependency 설치 | `python -m pip install -r requirements.txt` |
| editable install | `python -m pip install -e .` |

## Verification

| Check | Command |
|---|---|
| unit tests | `python -m unittest discover -s tests` |
| vision CLI | `python run_vision_experiment.py --help` |
| suite CLI | `python run_vision_suite.py --help` |
| Evidence report | `python scripts/validation/graph_evidence_report.py --profile smoke --include-external --out-dir <out-dir>` |

## Documentation

| Document | 내용 |
|---|---|
| `docs/README.md` | 문서 index |
| `docs/framework.md` | framework claim, lifecycle, components, metric |
| `docs/evidence.md` | framework 정당성 실험, pass criteria, verdict, provenance |
| `docs/research.md` | prior work positioning, design pattern survey |
| `docs/repository.md` | repository tree, package/script/test layout, change routing |
| `docs/maintenance.md` | migration, compatibility, removed surface, golden/asset policy |
| `docs/history.md` | legacy experiment observation, migration phase 기록 |
| `docs/demos/graphfl-assembly-scratch.html` | Graph-FL assembly scratch demo |

## Repository Layout

```text
.
├── graphfl_lab/
│   ├── designs/              GraphFLDesign registry and presets
│   ├── graph/                graph source, builder, control, diagnostics
│   ├── lifecycle/            lifecycle contracts and traces
│   ├── strategies/
│   │   ├── baselines/        graph-free and baseline strategies
│   │   └── graphfl/          Graph-FL runtime strategy modules
│   ├── diagnostics/          result schema and artifact writers
│   ├── experiments/
│   │   ├── vision/           vision single run, suite, stress, sweeps
│   │   ├── cora/             Cora single run and graph ablation
│   │   └── suites/vision/    suite features, variants, reporting
│   └── validation/           Evidence pack validation logic
├── configs/
│   ├── vision/               baseline, diagnostic, probe, smoke, stress, sweep configs
│   └── cora/                 graph ablation configs
├── scripts/
│   ├── checks/               preflight, evidence bundle, parity checks
│   ├── validation/           Evidence pack entry points
│   ├── reports/              plot and dashboard helpers
│   └── smoke/                smoke command wrappers
├── tests/                    CLI, graph, lifecycle, strategy, experiment, validation tests
└── docs/
    ├── *.md                  canonical project documentation
    └── demos/                HTML demo artifacts
```

상세 layout과 change routing은 `docs/repository.md`에서 관리한다.
