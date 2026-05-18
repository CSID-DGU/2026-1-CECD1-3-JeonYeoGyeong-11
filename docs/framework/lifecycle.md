# Graph-FL Lifecycle Intervention Framework

이 문서는 현재 프로젝트의 큰그림을 고정한다. 목표는 선행 graph-FL 연구들을 전부 별도 구현체로 복사하는 것이 아니다. FL round 안에서 graph가 개입하는 자연스러운 지점을 모듈화해서, 대부분의 graph-conditioned FL 방법을 공통 pipeline의 조합으로 표현하고, 최종 accuracy 이전의 내부 mechanism을 같은 방식으로 진단하는 것이다.

## 1. 핵심 관점

graph-FL 선행연구들은 겉으로는 서로 다른 graph algorithm처럼 보이지만, 대부분은 아래 lifecycle 중 어느 지점에 무엇을 꽂는지가 다를 뿐이다.

```text
Round t start
-> decide model/state delivered to each client
-> optional local objective hook
-> client local training
-> extract client state
-> estimate client-client relation
-> construct graph/topology
-> apply graph-conditioned aggregation or personalization
-> record diagnostics and counterfactual traces
-> store state for round t+1
```

따라서 프레임워크의 핵심 추상화는 `graph_source`, `graph_mode`, `aggregation_target` 그 자체가 아니다. 그것들은 하위 실행 노브다. 상위 구조는 FL lifecycle의 개입 지점이고, 연구자가 실제로 조립하는 단위는 `GraphFLDesign`이다.

```text
GraphFLDesign
  -> lifecycle modules
  -> standard trace
  -> counterfactual diagnostics
```

이 design 단위가 있어야 연구자가 state만 바꾸거나, relation estimator만 바꾸거나, topology만 바꾸는 식으로 새 graph-FL 설계를 만들 수 있다.

## 2. Lifecycle Modules

| 모듈 | 개입 시점 | 역할 | 예시 |
|---|---|---|---|
| `DeliveryPolicy` | round 시작, client training 전 | client에게 내려줄 model/state를 결정 | global model, previous personalized model, client-specific cloud model |
| `LocalObjectiveHook` | local training 중 | client loss에 graph/personality 관련 항을 추가 | FedAMP proximal term, pFedGraph cluster regularization, FED-PUB mask regularization |
| `ClientStateExtractor` | local training 중/후 | graph를 만들 client representation을 추출 | update, weight, EMA update, accumulated gradient, classifier head, functional embedding |
| `RelationEstimator` | server aggregation 전 | client-client relation score를 계산 | cosine, RBF, attention, QP score, learned metric |
| `TopologyOperator` | relation 계산 후 | relation score를 graph/topology로 변환 | dense, kNN, threshold, row-stochastic, clustering-only, control graph |
| `AggregationOperator` | server aggregation 시점 | graph를 사용해 update/model/personalized state를 만든다 | global update, graph-filtered update, weight smoothing, client-wise model mixture |
| `StateStore` | round 전후 | round 간 필요한 상태를 저장 | EMA graph, accumulated gradient, previous personalized model, previous relation |
| `DiagnosticProtocol` | 모든 단계 | 내부 값과 counterfactual 비교를 기록 | DI, N_eff, alignment, LOO, graph entropy, matched controls |

이 구조의 장점은 선행연구별 adapter를 무한히 늘리지 않아도 된다는 점이다. pFedGraph, FedAMP, FED-PUB 같은 이름은 별도 코드 덩어리가 아니라 위 모듈들의 조합으로 설명된다.

## 3. 선행연구 대응 방식

| 방법 | lifecycle 조합 | 현재 목표 |
|---|---|---|
| FedAMP | `DeliveryPolicy`: personalized cloud model, `ClientStateExtractor`: model weight, `RelationEstimator`: attentive distance kernel, `AggregationOperator`: client-specific attentive aggregate, `LocalObjectiveHook`: proximal term | relation/weight smoothing은 proxy 가능, exact personalized delivery는 확장 대상 |
| pFedGraph | `ClientStateExtractor`: global 대비 update + sample size, `RelationEstimator`: cosine difference + simplex QP, `TopologyOperator`: row-stochastic collaboration graph, `AggregationOperator`: client-wise neighbor model mixture, `LocalObjectiveHook`: cluster model regularization | QP relation은 구현 가능, row-wise personalized mixture는 확장 대상 |
| FedAGA | `ClientStateExtractor`: accumulated gradient, `RelationEstimator`: gradient similarity, `TopologyOperator`: dynamic graph, `AggregationOperator`: adaptive graph aggregation | EMA update 기반 proxy 가능 |
| SFL | `ClientStateExtractor`: model state, `RelationEstimator`: client relation graph, `AggregationOperator`: server GCN personalized model generation | graph-filter proxy 가능, full server GCN은 interface target |
| FED-PUB / GPFL | `ClientStateExtractor`: proxy graph functional embedding, `RelationEstimator`: embedding cosine, `TopologyOperator`: row-normalized relation, `AggregationOperator`: personalized model per client, `LocalObjectiveHook`: mask/previous model regularization | source plugin 위치는 명확, personalized target과 client hook 확장 필요 |

정확한 재현 여부는 네 단계로 표시한다.

| 등급 | 의미 |
|---|---|
| `core-supported` | 현재 framework module만으로 실행과 진단이 가능 |
| `proxy-supported` | 핵심 relation/aggregation 효과는 표현하지만 논문 전체 exact reproduction은 아님 |
| `interface-target` | 필요한 hook 위치는 명확하지만 구현이 아직 필요 |
| `out-of-scope` | 별도 architecture 자체가 본체라 aggregation-level 진단 프레임워크 밖에 가까움 |

## 4. Trace가 핵심이다

이 프레임워크는 최종 결과만 보는 것이 아니다. 각 모듈은 output뿐 아니라 표준 trace를 남겨야 한다.

```text
module_output, module_trace = module.run(context)
diagnostics.collect(module_trace)
```

이 원칙이 있어야 “accuracy가 올랐다”가 아니라 “어떤 내부 mechanism이 바뀌어서 성능 변화로 이어졌는가”를 볼 수 있다.

## 5. Standard Trace Schema

| 모듈 | 반드시 남길 trace | 해석 질문 |
|---|---|---|
| `DeliveryPolicy` | delivered model id/type, global-personalized distance, per-client delivery norm | client별 model을 내려주는 것이 실제로 다른 상태를 만들었나 |
| `LocalObjectiveHook` | hook loss, base loss 대비 hook loss 비율, hook gradient norm | local hook이 실제 optimization에 영향을 줬나 |
| `ClientStateExtractor` | state norm, layer/head norm, cosine distribution, sample-size prior | graph 재료가 어떤 client를 크게 보이게 만들었나 |
| `RelationEstimator` | raw relation matrix stats, pairwise score distribution, relation entropy | relation이 informative한가, 거의 uniform인가 |
| `TopologyOperator` | graph density, degree stats, graph entropy, connected components, row entropy | topology가 dense/sparse/cluster/균일 중 어디에 가까운가 |
| `AggregationOperator` | alpha, q_i, alpha entropy, pre/post aggregate update, client contribution | 특정 client dominance가 줄었나, 실질 참여 client가 늘었나 |
| `StateStore` | EMA/update history norm, previous graph distance, state drift | round 간 memory가 효과를 만들었나 |
| `DiagnosticProtocol` | DI, N_eff, alignment, LOO, matched control deltas | graph 효과가 smoothing/cluster/dominance 효과와 분리되는가 |

현재 우리가 원하는 내부 값은 이 schema로 대부분 잡을 수 있다.

| 보고 싶은 값 | 잡히는 위치 |
|---|---|
| client update 크기와 지배력 | `ClientStateExtractor`, `AggregationOperator` |
| 특정 client가 전체 update를 지배하는지 | `q_i`, `DI`, `N_eff`, `alpha_entropy` |
| graph가 실제 관계 정보를 담았는지 | `RelationEstimator`, `TopologyOperator`, matched controls |
| 단순 smoothing 효과인지 | uniform/identity graph, graph-free correction 비교 |
| clustering 효과인지 | clustering-only counterfactual |
| fine-grained edge 효과인지 | real graph vs clustering-only |
| sample-size prior 효과인지 | pFedGraph-style prior on/off comparison |
| update 방향이 개선됐는지 | alignment pre/post |
| 특정 client 제거에 민감한지 | LOO pre/post |
| graph가 너무 dense/sparse한지 | density, degree, entropy |
| personalized aggregation이 의미 있는지 | row entropy, global-personalized distance, per-client mixture drift |
| local hook이 작동했는지 | hook loss, hook gradient norm |

## 6. Counterfactual Diagnostic Runner

실제 training path는 한 가지일 수 있다. 하지만 진단은 같은 round의 client artifacts를 가지고 여러 counterfactual path를 shadow 계산해야 한다.

```text
actual path:
  real graph -> aggregation -> model update

shadow diagnostic paths:
  same client states -> random matched graph -> diagnostic metrics
  same client states -> shuffled graph -> diagnostic metrics
  same client states -> uniform graph -> diagnostic metrics
  same client states -> identity graph -> diagnostic metrics
  same client states -> clustering-only graph -> diagnostic metrics
  same client states -> graph-free correction -> diagnostic metrics
```

이 구조가 있어야 다음 질문을 round 내부에서 검토할 수 있다.

- real graph가 matched control보다 나은가
- graph가 단순히 update norm을 줄인 것인가
- coarse clustering만으로 충분한가
- fine-grained edge가 추가 정보를 주는가
- dominance correction만으로 비슷한 결과가 나는가
- sample-size prior가 relation estimator에 어떤 영향을 주는가

즉 final accuracy는 마지막 확인일 뿐이고, 주장은 shadow diagnostics에서 먼저 만들어져야 한다.

## 7. Framework Claim

이 프레임워크의 claim은 “새 graph algorithm이 최고 성능을 낸다”가 아니다.

> Graph-based FL methods can be decomposed into lifecycle-level intervention modules. By standardizing module traces and counterfactual diagnostics, we can analyze whether their gains come from relation information, topology, smoothing, clustering, sample-size prior, dominance suppression, or local objective effects.

한국어로는 다음이 핵심이다.

> 본 프레임워크는 개별 graph-FL 알고리즘을 별도로 재구현하는 것이 아니라, FL round 안에서 graph가 개입하는 지점을 모듈화한다. 이를 통해 다양한 graph-conditioned FL 방법을 공통 pipeline의 조합으로 표현하고, 동일한 trace와 counterfactual diagnostic protocol로 성능 이득의 원인을 분해한다.

## 8. 구현 설계 원칙

코드로 옮길 때는 다음 원칙을 지킨다.

1. module은 output과 trace를 함께 반환한다.
2. trace schema는 method별로 흩어지지 않고 공통 key를 가진다.
3. actual path와 shadow diagnostic path를 분리한다.
4. exact reproduction이 어려운 방법은 억지로 구현하지 않고 `proxy-supported` 또는 `interface-target`으로 둔다.
5. `graph_source`, `graph_mode`, `aggregation_target`은 유지하되 lifecycle module 아래의 실행 노브로 재배치한다.
6. 새 방법을 추가할 때는 “어느 hook에 무엇을 꽂는가”와 “어떤 trace를 남기는가”를 먼저 명시한다.

이 큰그림을 기준으로 다음 코드 작업은 lifecycle module contract와 trace schema를 먼저 고정한 뒤, `GraphFLDiagnosticStrategy` runtime을 그 contract 위로 점진적으로 옮기는 순서가 맞다.

현재 작업 우선순위는 [experimental-design.md](experimental-design.md)의 core experiment와
[cleanup-plan.md](cleanup-plan.md)의 staged cleanup 기준을 따른다. 과거 phase 문서는
archive에 남아 있지만 새 작업의 출발점으로 쓰지 않는다.
