# Project History

이 문서는 현재 Graph-FL gain attribution claim이 만들어지기 전의 실험 관찰, migration phase, 방향 전환을 보존한다. 기준은 repository에 남아 있는 current docs와 testable surface다.

## History Scope

| Item | Canonical Treatment |
|---|---|
| phase conclusion | 현재 claim과 연결되는 결론만 보존 |
| experiment setting | dataset, partition, client count, seed, graph source, metric 보존 |
| observation | 수치/현상/비교 기준 중심으로 보존 |
| current mapping | 현재 Evidence, metric, artifact contract와 연결 |
| runtime command log | canonical docs에 포함하지 않음 |

## Legacy Experiment Reports

| Report | Setting | Observations | Current Mapping |
|---|---|---|---|
| Phase 1 diagnostics | Fashion-MNIST, Dirichlet, 5 clients, seeds 42/43/44, update norm, `DI`, `N_eff`, alignment, `LOO` | client contribution imbalance, seed variation, interaction pathology signal | metrics and multi-seed diagnostic summaries |
| Phase 2 graph informativeness | Fashion-MNIST, Dirichlet, 5 clients, classifier-head update, update/random/shuffled/uniform/identity variants | weak control graph boundary, seed variation, update graph sensitivity | construction and diagnostic Evidence pack |
| Phase 2 signed conflict kNN | `signed_conflict_knn`, dense graph, real/update and controls | update graph effect limited, matched controls important, graph construction validation needed | source/mode attribution and graph parity |
| Phase 2 n20 alpha 0.1 | 20 clients, Dirichlet alpha 0.1, `signed_conflict_knn` | weak real graph advantage over controls, seed variation, graph/control separation needed | multi-seed and control-family Evidence |
| Phase 2 n20 alpha 0.3 | 20 clients, Dirichlet alpha 0.3, `signed_conflict_knn` | weak graph-control separation, alpha sensitivity, control-family stability needed | Non-IID stress calibration |
| Phase 2 pFedSim-like | pFedSim-like graph, real/update and controls | proxy graph quality sensitivity, limited control gap, diagnostic-centered interpretation needed | paper-mechanism alignment |
| Phase 2 graph source sanity | update, EMA update, classifier-head update, update/random/identity variants | sources can run, metrics differ by source, artifact paths are produced | source contract and artifact contract |
| Phase 2.5 smoothing failure | graph smoothing without relation signal | smoothing-only effect exists, random/uniform controls matter, graph quality sensitivity appears | matched controls and graph-free control |
| Phase 3 dominance-aware | Fashion-MNIST, Dirichlet, dominance-aware baseline, accuracy/loss/update norm/`DI`/`N_eff` | dominance correction signal exists, compare with graph smoothing | graph-free dominance correction |

## Phase 1 Diagnostics

Purpose: Non-IID setting에서 Graph-FL diagnostic metrics가 의미 있는 signal을 보이는지 확인했다.

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| partition | Dirichlet |
| clients | 5 |
| seeds | 42, 43, 44 |
| metrics | update norm, `DI`, `N_eff`, alignment, `LOO` |

| Observation | Meaning |
|---|---|
| client contribution imbalance appears | dominance diagnostics가 필요하다 |
| variation differs by seed | multi-seed summary가 필요하다 |
| interaction pathology signal appears | Phase 2 graph informativeness check로 이어진다 |

Phase 1의 결론은 accuracy만으로 Graph-FL gain을 설명하기 어렵다는 점이다. Contribution imbalance와 seed variation이 같이 나타났기 때문에 이후 문서와 artifact는 per-client metric, round metric, counterfactual metric을 분리해 기록하는 방향으로 이동했다.

## Phase 2 Graph Informativeness

Purpose: update-based graph가 random, shuffled, uniform, identity controls보다 informative한지 평가했다.

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| partition | Dirichlet |
| clients | 5 |
| graph source | classifier head update |
| variants | update, random, shuffled, uniform, identity |

| Observation | Meaning |
|---|---|
| control graph boundary is weak | graph-specific explanation에는 matched controls가 필요하다 |
| seed variation appears | multi-seed Evidence가 필요하다 |
| update graph sensitivity appears | source와 topology attribution을 분리해야 한다 |

이 단계에서 real graph와 control graph의 차이가 항상 강하지 않다는 점이 드러났다. 따라서 현재 framework는 graph gain을 단일 성능 수치로 주장하지 않고, `graph_source`, `graph_mode`, `aggregation_target`, control family, diagnostics를 같은 row에서 비교한다.

## Phase 2 Signed Conflict Variants

| Setting | Details | Observation | Current Meaning |
|---|---|---|---|
| signed conflict kNN | `signed_conflict_knn`, dense graph, real/update and controls | update graph effect is limited | source/mode attribution이 필요하다 |
| n20 alpha 0.1 | 20 clients, Dirichlet alpha 0.1, `signed_conflict_knn` | real graph advantage over controls is weak | stronger graph validation과 multi-seed summary가 필요하다 |
| n20 alpha 0.3 | 20 clients, Dirichlet alpha 0.3, `signed_conflict_knn` | graph-control separation is weak and alpha-sensitive | Non-IID stress calibration이 필요하다 |
| pFedSim-like | pFedSim-like graph, real/update and controls | proxy graph quality is sensitive and control gap is limited | paper-mechanism alignment가 필요하다 |

Signed conflict series는 graph mechanism이 source와 topology 선택에 민감하다는 근거다. 현재 `design_space_matrix.csv`가 16 sources, 18 modes, 5 targets, 6 correction profiles를 모두 계산하는 이유가 이 관찰에서 나온다.

## Phase 2 Source And Smoothing Checks

| Report | Setting | Observation | Current Meaning |
|---|---|---|---|
| graph source sanity | `update`, `ema_update`, `classifier_head_update`; update/random/identity variants | all sources can run and produce artifacts, but metrics differ by source | source contract와 source attribution이 필요하다 |
| smoothing failure | graph smoothing without relation signal | smoothing-only effect exists and can move metrics without relation Evidence | real graph, random, uniform, identity, graph-free controls를 분리해야 한다 |

Smoothing check는 중요한 전환점이다. Graph operation 자체가 metric을 움직일 수 있으므로 Graph-FL claim은 relation-specific effect와 generic smoothing effect를 분리해야 한다. 현재 `counterfactual_metrics.csv`, `real_diagnostic_consistency.csv`, `metric_validity_summary.csv`가 이 분리를 담당한다.

## Phase 3 Dominance-Aware Check

Purpose: graph gain처럼 보이는 결과가 dominant-client correction으로 설명되는지 확인했다.

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| partition | Dirichlet |
| method | dominance-aware baseline |
| metrics | accuracy, loss, update norm, `DI`, `N_eff` |

| Observation | Meaning |
|---|---|
| dominance correction signal appears | graph-free controls가 필요하다 |
| dominance correction and graph smoothing must be compared | attribution은 relation-specific effect와 low-order-statistic effect를 분리해야 한다 |

이 관찰 때문에 current framework는 `DI`, `N_eff`, alignment, `LOO`를 artifact contract에 포함한다. Graph-FL gain을 relation graph 효과로 주장하려면 dominance/norm/contribution correction과 구분되어야 한다.

## Previous Direction

| Earlier Focus | Current Focus |
|---|---|
| raw update graph performance | graph gain attribution |
| spectral-only smoothing | matched control and diagnostic Evidence |
| single graph variant | composable graph design space |
| semantic graph roadmap | graph source + relation mechanism |
| smoothing gain | control-specific diagnostic gap |
| raw update limitation | source/mode/target attribution |
| roadmap item | framework extension contract |

Earlier experiment design axes:

| Axis | Content |
|---|---|
| graph signal | raw update, local weight, EMA update |
| graph operator | cosine similarity, kNN, spectral smoothing |
| controls | random, shuffled, uniform, identity |
| metrics | accuracy, loss, graph metric, alignment |

## Migration Phases

| Phase | Scope | Result |
|---:|---|---|
| 0 | repository simplification, archive routing, `graphfl_lab` naming preparation, protected facade, cleanup criteria | current-claim docs separated from legacy material |
| 1 | trace record, design identity, support level, diagnostics writer compatibility | trace vocabulary and design-space keys introduced |
| 2 | lifecycle context, module contracts, support status, legacy source/mode/target connection | relation, topology, aggregation boundaries fixed; silent fallback blocked |
| 3 | `GraphFLDesign` composer, registry, prior-work proxy profiles, CLI bridge | methods represented as component combinations with support levels |
| 4 | state envelope, relation module, topology module, graph metadata | graph construction decomposed into source, relation, topology |
| 5 | actual real-graph path, shadow controls, `DI`, `N_eff`, alignment, `LOO`, graph stats | counterfactual diagnostic runner and artifact rows added |
| 6 | aggregation target, delivery policy, state store, local objective hook | graph application point separated and personalized method slots defined |
| 7 | lifecycle/graph/diagnostics/design tests, smoke checks, docs, artifact contract | migration completed with validation path |

## Migration Phase Details

| Phase | Purpose | Scope | Result |
|---:|---|---|---|
| 0 | lifecycle migration 전에 repository surface를 정리 | archive routing, canonical naming preparation, protected facade, cleanup criteria | active docs와 previous direction material을 분리 |
| 1 | lifecycle component가 공유할 trace schema 정의 | stable top-level trace fields, design identity, support level, diagnostics writer compatibility | trace vocabulary와 design-space key가 artifact claim과 연결 |
| 2 | lifecycle context와 module contract 정의 | round/client/state/config context, module input/output boundary, unsupported/interface-target status | module boundary 확정 |
| 3 | `GraphFLDesign`과 registry로 method profile 표현 | design metadata, preset lookup, prior-work proxy profiles, CLI bridge | method profile이 component combination으로 전환 |
| 4 | graph construction을 state, relation, topology로 분해 | update/weight/history state envelope, cosine/RBF/QP relation, dense/kNN/control topology, metadata | graph construction이 source/relation/topology 기준으로 auditable해짐 |
| 5 | real graph path와 control graph path를 shared diagnostics로 비교 | actual path, shadow path, `DI`, `N_eff`, alignment, `LOO`, graph stats | counterfactual rows가 attribution Evidence가 됨 |
| 6 | aggregation, delivery, state, local objective hook 정의 | graph-filtered targets, global/personalized delivery, EMA/previous graph state, proximal/local hook | graph application point와 personalized slot이 명시됨 |
| 7 | tests, smoke checks, docs, artifact contract로 migration 검증 | lifecycle, graph, diagnostics, design tests; runner smoke; docs sync | current Evidence path 완성 |

Migration current status:

| Field | Value |
|---|---|
| phase | completed |
| final phase | Phase 7 |
| package | `graphfl_lab` |
| core framework | lifecycle component structure |
| validation | unit tests, smoke checks, diagnostic preflight |

Completed areas:

| Area | Result |
|---|---|
| trace schema | standardized |
| lifecycle context | implemented |
| design registry | implemented |
| state/relation/topology modules | implemented |
| counterfactual diagnostics | implemented |
| aggregation/delivery hooks | implemented |
| docs and validation | completed |

## Migration Workflow Rules

| Rule | Standard |
|---|---|
| lifecycle boundary | state, relation, topology, aggregation 역할 분리 |
| compatibility | public-surface change를 해당 phase scope 안에서 관리 |
| trace | support level과 status를 명시 |
| diagnostics | counterfactual path를 server model update와 분리 |
| design | Graph-FL method를 component combination으로 표현 |
