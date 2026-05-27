# Graph-FL Framework

Graph-FL Design Lab는 Graph-FL gain을 원인 단위로 분해하는 framework다. 핵심은 real client relation graph 효과를 matched controls, graph-free controls, diagnostic metric, artifact로 검증하는 것이다.

## Claim

Graph-FL gain은 하나의 단일 효과가 아니라 다음 항으로 분해한다.

```text
Graph-FL gain
= client state effect
+ relation effect
+ topology effect
+ graph filtering effect
+ optimizer effect
+ low-order statistic effect
```

| Axis | 검증 질문 | 근거 |
|---|---|---|
| `graph_source` | client를 어떤 state로 표현하는가 | update, EMA update, classifier head, weight |
| `graph_mode` | relation과 topology를 어떻게 구성하는가 | cosine, kNN, RBF, learned smoothness, pFedGraph-QP |
| `aggregation_target` | graph를 어느 signal에 적용하는가 | update, EMA update, weight |
| `correction_family` | real graph를 어떤 control과 비교하는가 | random, shuffled, uniform, identity, clustering-only, graph-free |
| `diagnostics` | 어떤 mechanism 변화가 관찰되는가 | alignment, `DI`, `N_eff`, `LOO`, graph metric |

## Lifecycle

```text
round start
├── delivery policy
├── local objective hook
├── client local training
├── client state extraction
├── relation estimation
├── topology construction
├── graph-conditioned aggregation
├── diagnostics and counterfactual traces
└── state store
```

| Layer | Interface | 위치 |
|---|---|---|
| method design | `GraphFLDesign`, `ComponentSpec` | `graphfl_lab/designs/` |
| client state | `register_graph_source`, `GraphSourceResult` | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| relation estimator | `register_graph_builder`, `GraphBuildContext` | `graphfl_lab/graph/registry.py` |
| topology operator | graph builder, sparsification helper | `graphfl_lab/graph/sparsification.py`, `graphfl_lab/graph/controls.py` |
| aggregation operator | `aggregation_target` | `graphfl_lab/strategies/graphfl/targets.py` |
| runtime strategy | `GraphFLDiagnosticStrategy` | `graphfl_lab/strategies/graphfl/strategy.py` |
| diagnostics | metric and artifact writer | `graphfl_lab/diagnostics/` |
| suite grammar | variant parser | `graphfl_lab/experiments/suites/vision/variants.py` |

## GraphFLDesign Slot

Required slots:

```text
client_state
relation
topology
aggregation
```

Extension slots:

```text
delivery
local_objective
state_store
diagnostics
```

## Component Contract

| Component | Contract |
|---|---|
| `graph_source` | client order 유지, fixed-length vector, finite numeric values, metadata |
| `graph_builder` | `(num_clients, num_clients)` adjacency, finite non-negative weights, zero diagonal |
| `aggregation_target` | output shape 유지, finite values, target metadata |
| `GraphFLDesign` | source/mode/target/preset metadata와 support level 기록 |
| artifact | trace, diagnostics, graph stats, Evidence row 기록 |

## Built-In Preset

| Preset | Method profile | Runnable knobs | Support |
|---|---|---|---|
| `fedamp_proxy` | FedAMP attentive message passing | `weight + rbf + graph_filtered_weight` | proxy-supported |
| `sfl_proxy` | SFL graph-structured aggregation | `weight + learned_smooth + graph_filtered_weight` | proxy-supported |
| `pfedgraph_proxy` | pFedGraph inferred collaboration graph | `update + pfedgraph_qp + graph_filtered_update` | proxy-supported |
| `ema_magnitude_knn_filtered` | FedAGA accumulated-gradient graph | `ema_update + magnitude_knn + graph_filtered_ema_update` | proxy-supported |
| `fedpub_like` | functional embedding + personalized aggregation | interface target | interface-target |

## Support Vocabulary

| Level | 의미 |
|---|---|
| `core-supported` | framework component로 실행하고 diagnostics를 기록 |
| `proxy-supported` | paper mechanism을 diagnostic path에 투영 |
| `paper-kernel` | paper 수식 또는 설명 기반 independent kernel |
| `exact-reference` | official implementation/version 기준 비교 |
| `interface-target` | hook 위치와 design slot을 정의한 확장 대상 |

## Experiment Design

| Experiment | 목적 | 주요 관찰값 |
|---|---|---|
| Non-IID stress calibration | data heterogeneity 강도 확인 | accuracy, loss, update norm, `DI`, `N_eff` |
| real graph vs controls | relation-specific signal 분리 | real-control gap, alignment, `LOO` |
| graph-free correction | dominance와 norm effect 분리 | contribution share, `DI`, `N_eff` |
| source/mode/target ablation | component별 기여 확인 | graph metric, target metric, artifact row |
| paper mechanism alignment | prior work mechanism 대응 확인 | pFedGraph, FedAMP, SFL, FedAGA rows |
| Evidence pack | framework 정당성 검증 | parity, design-space, extensibility artifacts |

## Implementation Surface

| 구현 단위 | 현재 상태 | 역할 |
|---|---|---|
| package/runtime | `graphfl_lab`, Flower entrypoint, top-level runner | Graph-FL diagnostic experiment 실행 |
| default design | `default_similarity_knn` | update 기반 similarity-kNN graph |
| graph source | update, EMA update, classifier-head update, weight | client representation |
| graph construction | cosine/kNN, magnitude/RBF, learned smooth, pFedGraph-QP | relation graph 구성 |
| control family | identity, random, shuffled, uniform, clustering-only, graph-free | 대안 설명 분리 |
| aggregation target | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` | graph 적용 위치 |
| diagnostics | alignment, `DI`, `N_eff`, `LOO`, graph metrics | mechanism 해석 |
| Evidence pack | graph parity, mechanism alignment, design-space coverage | framework 정당성 검증 |

## Metrics

| Symbol | 의미 |
|---|---|
| $N$ | client 수 |
| $g_i^t$ | round $t$의 client $i$ raw local update |
| $\tilde g_i^t$ | corrected 또는 graph-filtered update |
| $p_i^t$ | FedAvg sample-size weight |
| $q_i^t$ | contribution share |
| $A^t$ | client graph adjacency |
| $L^t$ | graph Laplacian |
| $z_i^t$ | graph construction에 쓰인 client representation |
| $\epsilon$ | numerical stability constant |

Basic update:

$$
g_i^t = w_i^t - w^t
$$

FedAvg aggregation:

$$
g_{\mathrm{avg}}^t = \sum_i p_i^t g_i^t
$$

Pre/post update:

$$
\Delta_{\mathrm{pre}} = \sum_i p_i^{pre} g_i,
\qquad
\Delta_{\mathrm{post}} = \sum_i p_i^{post} \tilde g_i
$$

Contribution share:

$$
q_i = \frac{p_i \lVert g_i \rVert}{\sum_j p_j \lVert g_j \rVert + \epsilon}
$$

Alignment:

$$
\cos(x,y)=\frac{x^\top y}{\lVert x\rVert_2 \lVert y\rVert_2+\epsilon}
$$

Leave-one-out:

$$
\mathrm{LOO}_i = 1 - \cos(\Delta, \Delta_{-i})
$$

| Metric | 정의 | 해석 |
|---|---|---|
| `DI` | $\max_i q_i$ | dominant client 집중도 |
| `N_eff` | $1 / \sum_i q_i^2$ | effective contributing clients |
| alignment | $\cos(g_i, \Delta)$ | aggregate direction 대표성 |
| `LOO` | client 제거 전후 aggregate direction 변화 | single-client sensitivity |
| real-control gap | $Score_{real} - Score_{control}$ | relation-specific signal |

| Graph metric | 의미 |
|---|---|
| density | active edge 비율 |
| degree | client별 연결도 |
| entropy | edge weight 분산도 |
| smoothness | graph 위 update 변화량 |
| spectral energy | Laplacian 기준 high-frequency component |
| temporal stability | round 간 graph 변화 |

| Pattern | 해석 |
|---|---|
| `DI` down, `N_eff` up | dominance 완화 |
| alignment up | aggregate direction 대표성 증가 |
| `LOO` down | single-client sensitivity 완화 |
| real-control gap up | relation-specific signal 증가 |
| graph density up | smoothing tendency 증가 |

## Artifact Fields

| Artifact | 주요 field |
|---|---|
| `round_metrics.csv` | pre/post aggregate, `DI`, `N_eff`, alignment, `LOO` |
| `client_metrics.csv` | client contribution, norm, alignment |
| `graph_stats.csv` | density, degree, entropy, spectral metric |
| `counterfactual_metrics.csv` | real graph와 control graph gap |
| `metric_validity_summary.csv` | synthetic expected-direction result |

## Evidence 연결

Framework 정당성 근거는 `docs/evidence.md`에서 관리한다.

| Evidence axis | Verdict |
|---|---|
| Construction drift | 18 graph modes pass, max abs diff `2.21e-12`, edge F1 `1.0` |
| Paper-mechanism alignment | pFedGraph, FedAMP, SFL, FedAGA mapping 5 / 5 rows pass |
| Diagnostic sensitivity | 60 framework diagnostic rows pass |
| Design-space coverage | 8,640 / 8,640 calculation checks pass |
| Extensibility | custom source, builder, preset, target 4 / 4 pass |

## Extension Workflow

1. Method profile을 `client_state`, `relation`, `topology`, `aggregation` slot으로 작성한다.
2. 기존 `graph_source`, `graph_mode`, `aggregation_target` 재사용 범위를 확인한다.
3. client representation이 새로우면 `graph_source`를 추가한다.
4. relation/topology가 새로우면 `graph_builder`를 추가한다.
5. invalid 조합은 `require_graph_context(...)`로 분류한다.
6. `GraphFLDesign` preset을 등록한다.
7. diagnostics와 artifact field를 함께 추가한다.
8. deterministic adjacency, metadata, graph stats, control path test를 추가한다.
