# Graph-FL Gain Attribution Framework

## 연구 위치

Graph-based Federated Learning(Graph-FL)은 client relation을 학습 과정에 반영한다.
Non-IID 환경에서는 client마다 data distribution, update direction, model state가 달라지고, 이 차이가 aggregation과 personalization 설계의 핵심 신호가 된다.

이 project의 위치:

> Graph-FL gain을 relation-specific effect, generic smoothing effect, clustering effect, dominance correction effect, optimizer effect로 분해하는 attribution framework.

## 연구 질문

| ID | 질문 |
|---|---|
| Q1 | Graph-FL gain이 실제 client relation graph에서 오는가 |
| Q2 | smoothing, clustering, dominance correction으로도 설명되는가 |
| Q3 | real graph와 control gap이 어떤 mechanism metric과 함께 움직이는가 |
| Q4 | 새로운 graph intervention을 같은 구조 안에서 조립하고 비교할 수 있는가 |

## Framework 구조

```text
client representation
-> relation score
-> topology construction
-> edge weight / normalization
-> aggregation target
-> control or correction family
-> shared diagnostics
```

## 구현 범위

| 구현 단위 | 현재 상태 | 역할 |
|---|---|---|
| package/runtime | `graphfl_lab`, Flower entrypoint, top-level runner | Graph-FL diagnostic 실험 실행 |
| default design | `default_similarity_knn` | update 기반 similarity-kNN graph |
| graph source | update, EMA update, classifier-head update, weight | client representation |
| graph construction | cosine/kNN, magnitude/RBF, learned smooth, pFedGraph-QP | relation graph 구성 |
| control family | identity, random, shuffled, uniform, clustering-only, graph-free | 대안 설명 분리 |
| aggregation target | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` | graph 적용 위치 |
| diagnostics | alignment, `DI`, `N_eff`, `LOO`, graph metrics | mechanism 해석 |
| evidence pack | graph parity, mechanism alignment, design-space coverage | framework-quality 검증 |

## Prior Work 위치

| Work Family | Mechanism | Framework Mapping |
|---|---|---|
| FedAMP | model-distance relation, attentive aggregation | `weight + rbf + graph_filtered_weight` |
| SFL | client relation graph, personalized model generation | `weight + learned_smooth + graph_filtered_weight` |
| pFedGraph | learned collaboration graph, sample-size prior, QP | `update + pfedgraph_qp + graph_filtered_update` |
| FedAGA | accumulated-gradient relation, dynamic graph | `ema_update + magnitude_knn + graph_filtered_ema_update` |
| FED-PUB / GPFL | functional embedding similarity | interface-target |

## Evidence 논리

| Evidence | 질문 | Artifact |
|---|---|---|
| construction drift | 조립식 graph가 기존 builder를 재현하는가 | `graph_parity_summary.csv` |
| paper-mechanism alignment | 논문 mechanism과 component가 대응되는가 | `external_mechanism_alignment.csv` |
| diagnostic sensitivity | metric이 expected direction으로 움직이는가 | `metric_validity_summary.csv` |
| composability | 조합이 명시적으로 pass/unsupported/needs-review로 분류되는가 | `composability_matrix.csv` |
| design-space coverage | built-in 조합 전체가 계산 계약을 통과하는가 | `design_space_matrix.csv` |
| extensibility | custom source/builder/preset이 artifact까지 이어지는가 | `extension_contract_summary.csv` |

## Poster 주장

> Graph-FL Design Lab은 Graph-FL gain을 graph construction, matched controls, diagnostic metrics, design-space coverage로 분해해 검증하는 framework다.
