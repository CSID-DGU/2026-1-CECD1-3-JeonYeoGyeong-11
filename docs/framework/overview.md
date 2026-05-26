# Graph-FL Overview

## 핵심 Claim

Graph-FL Design Lab은 Graph-FL gain을 graph construction, matched control, diagnostic metric, design-space coverage로 분해해 검증하는 framework다.

중심 질문:

```text
Graph-FL gain이 실제 client relation graph에서 오는가,
아니면 dominance, norm, smoothing, optimizer effect로 설명되는가?
```

## Claim 분해

```text
Graph-FL gain
= client state effect
+ relation effect
+ topology effect
+ graph filtering effect
+ optimizer effect
+ low-order statistic effect
```

| 축 | 검증 질문 | Evidence |
|---|---|---|
| `graph_source` | client를 어떤 state로 표현하는가 | update, EMA update, classifier head, weight |
| `graph_mode` | relation과 topology를 어떻게 계산하는가 | kNN, RBF, learned smoothness, pFedGraph-QP |
| `aggregation_target` | graph를 어느 signal에 적용하는가 | update, EMA update, weight |
| `correction_family` | 어떤 control과 비교하는가 | real, random, shuffled, uniform, identity, clustering-only |
| `diagnostics` | 어떤 mechanism 변화가 나타나는가 | alignment, `DI`, `N_eff`, `LOO`, graph metric |

## Experiment Design

| 실험 | 목적 | 주요 관찰값 |
|---|---|---|
| Non-IID stress calibration | data heterogeneity 강도 확인 | accuracy, loss, update norm, `DI`, `N_eff` |
| real graph vs controls | relation-specific signal 분리 | real-control gap, alignment, `LOO` |
| graph-free correction | dominance와 norm 효과 분리 | contribution share, `DI`, `N_eff` |
| source/mode/target ablation | component별 기여 확인 | graph metric, target metric, artifact row |
| paper mechanism alignment | prior work mechanism 대응 확인 | pFedGraph, FedAMP, SFL, FedAGA row |
| framework evidence pack | framework 품질 검증 | parity, design-space, extensibility artifact |

## Research Position

| Work Family | Mechanism | Framework Mapping |
|---|---|---|
| FedAMP | model-distance relation, attentive aggregation | `weight + rbf + graph_filtered_weight` |
| SFL | client relation graph, personalized model generation | `weight + learned_smooth + graph_filtered_weight` |
| pFedGraph | collaboration graph, sample-size prior, QP | `update + pfedgraph_qp + graph_filtered_update` |
| FedAGA | accumulated-gradient relation, dynamic graph | `ema_update + magnitude_knn + graph_filtered_ema_update` |
| FED-PUB / GPFL | functional embedding similarity | interface-target |

## Poster Claim

Graph-FL gain을 해석하기 전에 graph construction drift, paper-mechanism alignment, diagnostic sensitivity, composability, design-space coverage, extensibility를 검증한다.

관련 문서:

- `docs/framework/metrics.md`
- `docs/framework/components.md`
- `docs/framework/evidence.md`
