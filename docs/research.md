# Research Positioning

이 프로젝트의 연구 위치는 Graph-FL gain attribution framework다. 핵심 novelty는 graph-specific explanation을 matched controls, graph-free controls, diagnostic metrics, artifact contract로 검증 가능한 protocol로 바꾸는 데 있다.

## Research Questions

| ID | Question | Repository Handle |
|---|---|---|
| Q1 | Graph-FL gain이 실제 client relation graph에서 오는가 | real graph와 random/shuffled/uniform/identity controls 비교 |
| Q2 | smoothing, clustering, dominance correction만으로 gain이 설명되는가 | graph-free correction과 matched graph control 분리 |
| Q3 | real graph와 control graph gap이 어떤 mechanism metric과 함께 움직이는가 | `DI`, `N_eff`, alignment, `LOO`, graph stats |
| Q4 | 새로운 graph intervention을 같은 구조 안에서 조립하고 비교할 수 있는가 | `graph_source`, `graph_mode`, `aggregation_target`, diagnostics slot |

| Topic Question | Judgment |
|---|---|
| topic | Graph-FL gain attribution |
| novelty | mechanism decomposition을 통한 matched control and graph-free control |
| research value | Graph-FL claim을 testable mechanism 단위로 전환 |
| implementation risk handling | `exact-reference`, `paper-kernel`, `proxy-reference`, `interface-target` level 분리 |
| expansion axis | privacy, robustness, fairness, scalability diagnostics |

## Framework Gap

Repository가 답하려는 runnable question은 다음과 같다.

```text
meaningful relation edges vs generic smoothing
matched random/shuffled/identity graph controls
coarse clustering vs fine-grained edges
dominance/norm/contribution correction
source/topology/target attribution
```

이 gap은 “accuracy가 올랐다”가 아니라 “어떤 relation mechanism 때문에 올랐는가”를 묻는다. 따라서 framework는 graph construction, graph application target, counterfactual control, diagnostic artifact를 같은 row에서 연결해야 한다.

## Prior Work Mapping

| Work Family | Mechanism | Framework Mapping |
|---|---|---|
| FedAMP | model-distance relation, attentive aggregation | `weight + rbf + graph_filtered_weight` |
| SFL | client relation graph, personalized model generation | `weight + learned_smooth + graph_filtered_weight` |
| pFedGraph | collaboration graph, sample-size prior, QP | `update + pfedgraph_qp + graph_filtered_update` |
| FedAGA | accumulated-gradient relation, dynamic graph | `ema_update + magnitude_knn + graph_filtered_ema_update` |
| FED-PUB / GPFL | functional embedding similarity | `interface-target` |
| Hypergraph/attention/hypernetwork | rich topology and personalized operators | `interface-target` |

| Method | Paper Mechanism | Current Mapping | Support |
|---|---|---|---|
| FedAMP | model-distance relation, attentive message passing, proximal update | `fedamp_proxy` | proxy-supported |
| SFL | client relation graph, server GCN, personalized model | `sfl_proxy` | proxy-supported |
| pFedGraph | cosine difference, sample-size prior, simplex QP, neighbor mixture | `pfedgraph_proxy` | proxy-supported |
| FedAGA | accumulated-gradient similarity, dynamic graph, adaptive aggregation | `fedaga_like` | proxy-supported |
| FED-PUB / GPFL | functional embedding, client-specific aggregation | functional embedding source + personalized target | interface-target |
| pFedGAT / FedAGHN / FedHyperGraph | attention, hypernetwork, hypergraph personalized operator | personalized aggregation operator | interface-target |

## Design Pattern Survey

Graph-FL/PFL method는 lifecycle slot 조합으로 분류한다.

```text
client_state
-> relation_estimator
-> topology_operator
-> aggregation_target
-> delivery / state / local_objective
-> diagnostics / counterfactual controls
```

| Method | Client State | Relation | Topology | Aggregation | Personalized Component | Support |
|---|---|---|---|---|---|---|
| FedAMP | model weights | distance/RBF/attention | dense weighted graph | personalized weight mixture | cloud model + proximal | proxy-supported |
| FedFomo | local models + validation utility | validation utility | directed top-M | weight mixture | collaborator selection | proxy-supported |
| APPLE | core models | learned relationship vector | directed sparse/full graph | client-side weight mixture | download budget | proxy-supported |
| SFL | model/client relation graph | learned relation | client-wise graph | personalized graph sharing | graph regularization | proxy-supported |
| pFedGraph | personalized models + sample prior | cosine + dataset-size prior/QP | row-stochastic graph | weight mixture | local regularization | proxy-supported |
| GCFL | gradient sequence | gradient norm/DTW | cluster block graph | cluster FedAvg | cluster model | proxy-supported |
| FED-PUB | functional embedding | functional similarity | community graph | weight averaging + mask | sparse mask | interface-target |
| GPFL | marginal parameters + graph descriptor | graph autoencoder score | reconstructed sparse graph | GNN-guided update aggregation | dynamic client network | interface-target |
| pFedGAT | model parameters | GAT attention | learned dynamic graph | personalized mixture | loss feedback | interface-target |
| FedAGHN | params/updates | graph hypernetwork | layer-wise graph | generated personalized weights | hypernetwork | interface-target |

| Purpose | Usage |
|---|---|
| lifecycle classification | method를 slot combination으로 표현 |
| support level | `core-supported`, `proxy-supported`, `interface-target` 할당 |
| trace design | 필요한 module trace 결정 |
| evidence design | `exact-reference`, `paper-kernel`, `proxy-reference` 구분 |

## Framework Design Notes

Graph-FL method는 component composition으로 표현한다.

```text
client state extraction
-> relation estimation
-> topology construction
-> aggregation target
-> delivery / personalization
-> local objective hook
-> state store
-> diagnostics and controls
```

| Responsibility | Path |
|---|---|
| method metadata | `graphfl_lab/designs/` |
| graph source/signal | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| graph builder/registry | `graphfl_lab/graph/builders.py`, `graphfl_lab/graph/registry.py` |
| controls/clustering | `graphfl_lab/graph/controls.py`, `graphfl_lab/graph/clustering.py` |
| Graph-FL runtime | `graphfl_lab/strategies/graphfl/` |
| baselines | `graphfl_lab/strategies/baselines/` |
| lifecycle | `graphfl_lab/lifecycle/` |
| diagnostics | `graphfl_lab/diagnostics/` |
| vision orchestration | `graphfl_lab/experiments/vision/` |
| vision suite/reporting | `graphfl_lab/experiments/suites/vision/` |
| configs | `configs/vision/`, `configs/cora/` |

| Rule | Standard |
|---|---|
| source reuse | client representation이 같으면 기존 `graph_source`를 재사용 |
| builder reuse | relation/topology behavior가 같으면 기존 `graph_mode`를 재사용 |
| target reuse | graph application point가 같으면 기존 `aggregation_target`를 재사용 |
| diagnostics | 새로운 mechanism마다 trace와 artifact field를 함께 추가 |

## Evidence Vocabulary

| Term | Meaning |
|---|---|
| `exact-reference` | official implementation/version과 직접 비교 |
| `paper-kernel` | paper equation 또는 description 기반 독립 kernel |
| `proxy-reference` | paper mechanism을 repository component로 대리 표현 |
| `interface-target` | framework slot과 hook이 정의된 확장 대상 |

## Capstone Framing

Position:

> Graph-FL gain is decomposed into relation-specific effect, generic smoothing effect, clustering effect, dominance correction effect, and optimizer effect.

| Evidence | Question | Artifact |
|---|---|---|
| construction drift | assembled graph가 reference builder를 재현하는가 | `graph_parity_summary.csv` |
| paper-mechanism alignment | component가 paper mechanism과 대응하는가 | `external_mechanism_alignment.csv` |
| diagnostic sensitivity | metric이 expected direction으로 움직이는가 | `metric_validity_summary.csv` |
| composability | 조합이 pass, unsupported, needs-review로 분류되는가 | `composability_matrix.csv` |
| design-space coverage | built-in 조합 전체가 calculation contract를 통과하는가 | `design_space_matrix.csv` |
| extensibility | custom source/builder/preset row가 artifact까지 도달하는가 | `extension_contract_summary.csv` |

짧은 요약:

> Graph-FL Design Lab는 graph construction, matched controls, diagnostic metrics, design-space checks로 Graph-FL 결과를 나눠 볼 수 있게 한다.
