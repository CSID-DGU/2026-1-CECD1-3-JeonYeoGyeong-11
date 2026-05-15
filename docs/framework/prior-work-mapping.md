# Prior Graph-FL Implementation Mapping

이 문서는 선행연구 이름을 그대로 붙이는 대신, 실제 구현에서 어떤 state를 보내고 어떤 relation을 추정하며 그 relation을 어디에 적용하는지로 정리한다. 결론부터 말하면 `graph_method`는 runnable method/profile을 고르는 상위 실행 단위이고, `graph_source`, `graph_mode`, `aggregation_target`은 그 아래의 실행 knob다.

## Support Vocabulary

| Level | Meaning |
|---|---|
| `core-supported` | 현재 framework component로 직접 실행 가능하며 의미가 좁게 정의되어 있다. |
| `proxy-supported` | 선행연구의 일부 mechanism을 현재 global-model diagnostic path에 투영했다. 논문 exact reproduction은 아니다. |
| `interface-target` | 필요한 interface는 보이지만 아직 runnable component가 없다. |
| `out-of-scope` | 현재 repo 목적과 맞지 않거나 별도 시스템이 필요하다. |

Compatibility aliases such as `fedamp_like`, `sfl_like`, and `pfedgraph_like` remain accepted, but internally resolve to `fedamp_proxy`, `sfl_proxy`, and `pfedgraph_proxy`. For new runs, prefer `--graph-method default_similarity_knn`, `--graph-method pfedgraph`, or another runnable method/profile when the goal is to select an algorithm family. FED-PUB-style functional embedding and personalized model delivery remain `interface-target`, not runnable presets.

## Method Profile Slots

| Slot | Meaning | Why it matters |
|---|---|---|
| `client_state` | server가 client를 표현하는 state. update, weight, gradient history, functional embedding 등 | 같은 cosine graph라도 무엇을 cosine하는지가 달라지면 claim이 달라진다. |
| `relation_estimator` | client-state에서 relation/edge weight를 추정하는 규칙 | kNN, RBF, row-simplex QP, attention, functional embedding cosine은 서로 다른 가정이다. |
| `topology_operator` | dense, sparse, row-stochastic, directed, hypergraph, learned topology 등 | graph gain이 relation 때문인지 sparsity/cluster 때문인지 분해해야 한다. |
| `aggregation_operator` | relation을 실제 model/update에 적용하는 방식 | 단일 global smoothing인지, client-specific mixture인지, server GCN인지가 핵심 차이다. |
| `personalization_site` | personalization이 server, client, local objective 중 어디에서 생기는가 | 많은 선행연구는 단일 global model이 아니라 client별 model을 만든다. |
| `local_objective_hook` | local training loss에 추가되는 proximal/regularization/mask term | graph가 평균만 바꾸는지 local optimization까지 바꾸는지 구분한다. |
| `diagnostic_projection` | 현재 framework에서 exact, proxy, future interface 중 무엇인가 | 비교 실험에서 baseline과 mechanism proxy를 혼동하지 않게 한다. |

## Current Mapping

| Method family | Paper-level behavior | Current runnable mapping | Support |
|---|---|---|---|
| FedAMP | Client별 personalized model parameter 사이 거리로 attentive message passing을 만들고, client는 personalized cloud model에 대한 proximal local update를 수행한다. | `fedamp_proxy`: `weight + rbf + graph_filtered_weight`. 단일 global model만 있으므로 relation/weight smoothing proxy다. | diagnostic proxy |
| SFL | Client-wise relation graph를 server GCN에 넣어 client-specific model을 만든다. | `sfl_proxy`: `weight + learned_smooth + graph_filtered_weight`. server GCN은 없으므로 graph-filter proxy다. | diagnostic proxy |
| pFedGraph | Local model과 global parameter 차이에서 cosine difference를 만들고, dataset-size prior와 함께 row-wise simplex QP로 collaboration graph를 추론한다. 이후 각 graph row로 neighbor model을 섞어 client별 model/cluster model을 만든다. | `pfedgraph_proxy`: `update + pfedgraph_qp + graph_filtered_update`. QP relation은 구현했지만 row-wise personalized delivery는 아직 proxy다. | diagnostic proxy |
| FedAGA | Local accumulated gradient similarity로 dynamic graph topology를 만들고 convergence/divergence criteria로 adaptive aggregation timing을 조절한다. | `fedaga_like`: `ema_update + magnitude_knn + graph_filtered_ema_update`. EMA update는 accumulated gradient의 proxy다. | diagnostic proxy |
| FED-PUB / GPFL | Proxy graph에 대한 model response를 functional embedding으로 만들고, embedding similarity로 client별 personalized aggregation을 수행한다. | Functional embedding graph source와 `personalized_weight` target이 필요하다. | interface target |
| pFedGAT / FedAGHN / FedHyperGraph | Latent relation, attention, hypernetwork, layer-wise/hyperbolic graph와 personalized operator가 결합된다. | Graph builder plugin은 붙일 수 있지만 exact reproduction은 personalized aggregation operator가 필요하다. | interface target |

## pFedGraph Boundary

공식 pFedGraph 구현은 대략 다음 구조를 갖는다.

1. Client local model과 initial/global parameter의 차이를 계산한다.
2. Update 차이의 cosine similarity를 difference 형태로 만든다.
3. Local data 비율을 prior로 넣는다.
4. 각 client row마다 simplex 제약 QP를 푼다.
5. 추정된 row weight로 neighbor model을 섞어 client별 personalized model이나 cluster model을 만든다.

현재 코드는 1-4의 relation estimator를 `pfedgraph_qp` graph mode로 넣었다. 하지만 runtime은 아직 Laplacian/filter 기반 단일 global aggregation이므로 QP row graph를 diagnostic graph로 투영한다. 즉 pFedGraph의 graph estimator를 비교 가능한 형태로 붙인 것이지, pFedGraph 전체 알고리즘의 exact reproduction은 아니다.

## FED-PUB Boundary

FED-PUB류는 `graph_source=functional_embedding`만으로는 부족하다. 필요한 확장 지점은 두 개다.

- Graph source plugin: proxy graph forward pass에서 functional embedding을 생성한다.
- Personalized aggregation target: `personalized_weight`처럼 client별 row-stochastic aggregation result를 client별로 내려보낸다.

그래서 FED-PUB류는 현재 runnable preset이 아니라 `interface-target`이다.

## Interpretation Rule

이 framework의 목적은 선행연구 이름을 빌려 성능 숫자를 높이는 것이 아니라, graph-based aggregation gain이 무엇에서 오는지 분해하는 것이다. Prior-work-inspired preset에는 항상 아래 비교가 따라야 한다.

| Comparison | Question |
|---|---|
| real graph vs matched random/shuffled/uniform/identity | relation 정보 자체가 있는가? |
| real graph vs clustering-only | fine-grained edge가 필요한가, coarse group만으로 충분한가? |
| real graph vs graph-free norm/cap/reweight | graph gain이 사실 dominance/magnitude correction인가? |
| pFedGraph-like vs learned_smooth/RBF | QP와 sample-size prior가 별도 효과를 내는가? |
| functional embedding source plugin vs update/weight source | representation source가 graph gain의 원인인가? |

`graph_filtered_*` is the preferred public spelling for new commands. Older `spectral_filtered_*` targets remain accepted for compatibility with existing configs and result metadata.
