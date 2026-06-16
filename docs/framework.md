# Graph-FL Framework

이 문서는 Graph-FL Design Lab의 실행 구조를 설명한다. 핵심은 client relation graph를 만드는 단계와 graph를 적용하는 단계를 나눠 기록하는 것이다.

## 기본 구조

```text
client local training
-> graph_source
-> graph_builder
-> aggregation_target
-> diagnostics / artifacts
```

| 단계 | 역할 | 주요 위치 |
|---|---|---|
| `graph_source` | client update, weight, EMA update 등을 graph 입력 vector로 바꿈 | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| `graph_builder` | client vector로 adjacency matrix를 만듦 | `graphfl_lab/graph/registry.py`, `graphfl_lab/graph/builders.py` |
| `aggregation_target` | graph filtering을 update, EMA update, weight 중 어디에 적용할지 정함 | `graphfl_lab/strategies/graphfl/targets.py` |
| control graph | real graph와 비교할 random, shuffled, uniform, identity graph 생성 | `graphfl_lab/graph/controls.py` |
| diagnostics | 결과 해석에 필요한 metric과 artifact row 작성 | `graphfl_lab/diagnostics/`, `graphfl_lab/strategies/graphfl/` |

## 실행 흐름

```text
round start
├── client local training
├── client state extraction
├── relation / topology construction
├── graph-conditioned aggregation
├── control graph comparison
├── diagnostics
└── artifact writing
```

`GraphFLDesign`은 실행에 필요한 component 조합을 묶는다.

| Slot | 예시 |
|---|---|
| `client_state` | update, EMA update, classifier-head update, weight |
| `relation` | cosine, RBF, learned smoothness, pFedGraph-QP |
| `topology` | dense, kNN, mutual-kNN, threshold |
| `aggregation` | graph-filtered update, graph-filtered EMA update, graph-filtered weight |

## Component Contract

| Component | 지켜야 하는 조건 |
|---|---|
| `graph_source` | client 순서 유지, client 수만큼 vector 반환, fixed length, finite value |
| `graph_builder` | `(N, N)` adjacency 반환, finite non-negative weight, zero diagonal |
| `aggregation_target` | client/layer shape 유지, finite value, target metadata 기록 |
| `GraphFLDesign` | source, builder, target, preset metadata 기록 |
| artifact | trace, diagnostics, graph stats row를 남김 |

확장 API는 `graphfl_lab.extensions`에서 제공한다.

```text
register_graph_source
register_graph_builder
register_aggregation_target
register_design
```

## 비교 축

| 축 | 확인하려는 것 |
|---|---|
| source | client를 어떤 state로 표현할 때 graph가 달라지는가 |
| builder | relation과 topology 선택이 결과를 어떻게 바꾸는가 |
| target | graph를 update, EMA update, weight 중 어디에 적용하는가 |
| control | real graph 효과와 smoothing/control 효과를 구분할 수 있는가 |
| diagnostics | 결과 변화가 어떤 metric과 함께 움직이는가 |

## Metric

| Symbol | 의미 |
|---|---|
| $N$ | client 수 |
| $g_i^t$ | round $t$의 client $i$ local update |
| $\tilde g_i^t$ | correction 또는 graph filtering 이후 update |
| $p_i^t$ | FedAvg sample-size weight |
| $q_i^t$ | contribution share |
| $A^t$ | client graph adjacency |
| $L^t$ | graph Laplacian |

기본 update:

$$
g_i^t = w_i^t - w^t
$$

FedAvg aggregate:

$$
g_{\mathrm{avg}}^t = \sum_i p_i^t g_i^t
$$

Contribution share:

$$
q_i = \frac{p_i \lVert g_i \rVert}{\sum_j p_j \lVert g_j \rVert + \epsilon}
$$

| Metric | 의미 |
|---|---|
| `DI` | 가장 큰 client contribution share |
| `N_eff` | 실제로 기여하는 client 수에 가까운 값 |
| alignment | client update와 aggregate 방향의 cosine similarity |
| `LOO` | client 하나를 뺐을 때 aggregate 방향이 얼마나 바뀌는지 |
| real-control gap | real graph와 control graph 결과 차이 |

## Artifact

| Artifact | 내용 |
|---|---|
| `round_metrics.csv` | round별 aggregate metric, `DI`, `N_eff`, alignment, `LOO` |
| `client_metrics.csv` | client별 contribution, norm, alignment |
| `graph_stats.csv` | density, degree, entropy, spectral metric |
| `counterfactual_metrics.csv` | real graph와 control graph 비교 |
| `module_traces.jsonl` | 사용 component, parameter, shape, metadata |
| `metric_validity_summary.csv` | synthetic case에서 metric 방향 확인 |

## 확장 흐름

1. 새 method를 `client_state`, `relation`, `topology`, `aggregation`으로 나눈다.
2. 기존 source, builder, target을 재사용할 수 있는지 확인한다.
3. 새 component가 필요하면 register API로 추가한다.
4. `GraphFLDesign` preset을 등록한다.
5. dry-run과 contract test로 shape, metadata, artifact가 유지되는지 확인한다.

대표 CLI:

```powershell
graphfl component new <source|builder|aggregation> <name>
graphfl component validate <plugin-path>
graphfl design compose ...
graphfl run single --track vision --config <config.json> --dry-run
```
