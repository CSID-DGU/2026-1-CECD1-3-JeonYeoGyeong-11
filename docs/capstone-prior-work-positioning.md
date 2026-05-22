# Graph-FL Gain Attribution Framework

## 1. Position

Graph-based Federated Learning(Graph-FL)은 client 사이의 관계를 학습 과정에
반영한다. Non-IID 환경에서는 client마다 데이터 분포, update 방향, 모델 상태가
다르다. 이 차이를 client relation graph로 표현하면 aggregation과
personalization을 더 세밀하게 설계할 수 있다.

Graph-FL의 성능 향상에는 여러 설명이 함께 섞인다. 실제 client relation이 도움이
됐을 수도 있고, 이웃 update를 섞는 smoothing 효과가 컸을 수도 있다. 유사한
client를 대략적인 group으로 묶은 효과, 큰 update를 가진 client의 영향력을 줄인
효과, optimizer나 aggregation path의 차이도 함께 작동할 수 있다.

```text
relation-specific effect
generic smoothing effect
coarse clustering effect
dominance correction effect
optimizer or aggregation-path effect
```

이 프레임워크는 Graph-FL gain을 위 설명들로 나누어 관찰한다. 한 graph 방법의
최종 accuracy만 보는 방식에서 벗어나, real graph와 control graph, graph-free
correction을 같은 실행 조건에서 비교한다. 여기에 alignment, DI, N_eff,
graph/spectral diagnostics를 함께 남겨 성능 차이의 원인을 해석한다.

핵심 질문은 네 가지다.

```text
Q1. Graph-FL의 성능 향상은 실제 client relation graph에서 오는가?
Q2. smoothing, clustering, dominance correction 같은 설명으로도 충분한가?
Q3. real graph와 control의 차이는 어떤 mechanism metric과 함께 움직이는가?
Q4. 새로운 graph-based intervention을 같은 구조 안에서 조립하고 비교할 수 있는가?
```

---

## 2. Framework Flow

Graph-FL 방법은 다음 구성요소의 조합으로 표현된다.

```text
client representation
-> relation score
-> topology construction
-> edge weight / normalization
-> aggregation target
-> control or correction family
-> shared diagnostics
```

새로운 graph 후보는 이 흐름의 한 지점에 들어온다. client representation을
바꾸는 방법, relation score를 바꾸는 방법, topology를 바꾸는 방법, aggregation
target을 바꾸는 방법을 같은 실행 구조 위에 올릴 수 있다.

한 run은 같은 seed, 같은 non-IID split, 같은 aggregation target을 공유한다.
차이는 graph factor와 control factor에서 생긴다.

```text
Input:
- client updates
- global model state
- non-IID split configuration
- graph construction configuration
- control variant configuration

Process:
- build real client relation graph
- build relation-destroyed controls
- build relation-free or graph-free controls
- run each variant under the same FL setting
- compute performance and mechanism metrics

Output:
- accuracy / loss
- real-control gap
- graph-free control gap
- alignment
- DI
- N_eff
- optional topology / spectral / LOO diagnostics
```

결과는 score table과 diagnostic trace를 함께 가진다. accuracy와 loss는 graph
intervention의 성능 효과를 보여준다. alignment, DI, N_eff는 relation quality와
client dominance의 변화를 보여준다. topology와 spectral metric은 graph shape와
smoothing 정도를 설명한다.

---

## 3. Current Implementation

현재 구현의 중심은 client relation graph 기반 aggregation-level intervention과
diagnostic attribution이다. 여기서 graph는 federated clients 사이의 relation
graph다. node는 client이고, edge는 update, model state, history signal에서
계산한 client 간 관계를 나타낸다.

현재 repo에서 실행 가능한 단위는 다음과 같다.

| 구현 단위 | 현재 상태 | 역할 |
|---|---|---|
| package/runtime | `graphfl_lab`, Flower entrypoint, top-level runner wrappers | graph-FL diagnostic 실험 실행 |
| default graph design | `default_similarity_knn` | update 기반 similarity-kNN graph |
| graph source | update, EMA update, classifier-head update, weight 계열 source | client representation 선택 |
| graph construction | cosine/kNN, magnitude/RBF/proxy, uniform/random/shuffled/control graph 계열 | real graph와 relation-destroyed controls 구성 |
| aggregation target | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` | graph를 적용할 학습 신호 선택 |
| control family | identity, matched random, shuffled, uniform, clustering-only, graph-free dominance reweight | gain의 대안 설명 분리 |
| diagnostics/schema | `result_schema_version`, `config_aliases_used`, `unsupported_components`, alignment, DI, N_eff, graph/spectral metrics | 결과 계약과 mechanism 해석 |
| executable configs | vision diagnostic configs, Cora extension/ablation config | vision setting 중심 실행, Cora는 graph-structured input 확장 확인 |
| validation tests | `tests/designs`, `tests/graph`, `tests/diagnostics`, schema/golden/runner tests | design registry, graph builder, control graph, result contract 확인 |

대표 pipeline은 다음 흐름이다.

```text
client update delta
-> cosine similarity
-> kNN graph
-> row-normalized adjacency
-> graph-filtered update aggregation
-> shared result schema and diagnostics
```

조립식 확장은 이 pipeline의 특정 조각을 교체한다. classifier-head relation은
source 교체, EMA-history relation은 source와 target 교체, RBF-style relation은
relation score 또는 edge weight 교체, graph-free dominance control은 correction
family 교체로 표현된다.

| 분류 | 의미 |
|---|---|
| core-supported | runner/config/test로 직접 실행 가능한 graph design과 diagnostic path |
| proxy-supported | 원 논문의 exact reproduction보다 mechanism 근사에 가까운 path |
| interface-target | 설계 위치가 잡힌 확장 대상 |
| out-of-scope | aggregation-level framework 밖의 별도 시스템 |

---

## 4. Modular Graph Design

조립 가능한 graph design space는 다음과 같다.

| 조립 위치 | 만들 수 있는 것 | 의미 |
|---|---|---|
| client representation | update delta, model weight, EMA update, classifier-head update, classifier-head weight | client를 표현하는 신호 |
| relation score | cosine similarity, magnitude-aware similarity, RBF-style similarity, learned/proxy score | client 사이의 관계 계산 |
| topology construction | kNN graph, threshold graph, dense graph, matched random graph, shuffled graph, clustering-only graph | score에서 edge 구조 생성 |
| edge weight / normalization | binary edge, similarity weight, row-normalized adjacency, smoothing strength, sample-size prior | 연결 강도와 mixing 정도 |
| aggregation target | graph-filtered update, graph-filtered EMA update, graph-filtered weight, personalized/model-mixture target | graph가 적용되는 학습 신호 |
| correction/control family | real graph, matched random, shuffled, uniform, identity, clustering-only, graph-free dominance reweight | gain의 설명 후보 |
| diagnostics | accuracy/loss, real-control gap, graph-free gap, alignment, DI, N_eff, LOO, topology/spectral metrics | graph effect 해석 |

이 설계 공간은 graph method가 어느 구성요소를 바꾸는지 보여준다. 새로운 방법이
들어오면 source, score, topology, weight, target, control 중 어디가 달라졌는지
표시하고 기존 variants와 비교한다. 선행연구도 같은 방식으로 배치된다. FedAMP
계열은 weight/RBF relation, FedAGA 계열은 EMA-history relation, pFedGraph 계열은
QP/proxy relation, FED-PUB/GPFL 계열은 functional embedding과 personalized
delivery 쪽으로 연결된다.

---

## 5. Research Questions

| 구분 | 질문 | 주요 비교 |
|---|---|---|
| attribution | real client relation graph는 relation-destroyed control보다 나은가? | real_graph vs matched_random / shuffled / identity |
| attribution | graph gain은 smoothing, clustering, dominance correction으로 설명되는가? | real_graph vs uniform / clustering_only / graphfree_dominance_reweight |
| mechanism | real graph와 control의 차이는 alignment, DI, N_eff와 함께 움직이는가? | performance gap + mechanism metrics |
| design | graph gain은 client representation 또는 topology choice에 민감한가? | update / EMA / classifier-head source, kNN / threshold / dense topology |
| framework | 새로운 graph-based intervention은 공통 interface 안에서 비교 가능한가? | graph_source / relation_score / topology / target 조합 |

초기 분석의 중심은 relation, smoothing, coarse community, dominance correction이다.
topology, representation, spectral/smoothness는 보조 진단으로 함께 기록된다.

| 범위 | 가설 | 관찰할 증거 | 해석 |
|---|---|---|---|
| primary | Relation-specific effect | real_graph가 matched_random, shuffled, uniform, identity보다 높고 alignment도 개선 | 실제 client relation이 단순 smoothing 이상 정보를 제공 |
| primary | Coarse community effect | clustering_only가 real_graph와 비슷함 | fine-grained edge보다 cluster/homophily가 핵심 |
| primary | Generic smoothing effect | uniform 또는 matched_random이 real_graph와 비슷함 | graph relation보다 평균화/smoothing 자체가 주요 원인 |
| primary | Dominance correction effect | graphfree_dominance_reweight가 real_graph와 비슷하고 DI/N_eff가 안정화됨 | graph relation보다 큰 update/client 영향력 보정이 주요 원인 |
| secondary | Topology effect | 같은 source에서도 density, degree, entropy, sparsity 변화에 따라 결과가 달라짐 | graph shape가 성능을 좌우 |
| secondary | Representation effect | graph_source 변경에 따라 결과가 달라짐 | client representation이 relation quality를 좌우 |
| secondary | Spectral/smoothness effect | smoothness, low/high-frequency energy 변화가 성능 변화와 연결됨 | graph signal 관점의 smoothing 효과 |

---

## 6. Controls

real graph와 controls는 가능한 한 같은 graph construction pipeline을 공유한다.

```text
same representation
same density or comparable edge budget
same normalization
same aggregation target
different tested factor only
```

| Control | Preserved factor | Changed factor | Purpose |
|---|---|---|---|
| real_graph | client representation, similarity score, topology, edge weight | 없음 | 의도한 client relation |
| identity | original FedAvg-style update path | graph intervention 제거 | graph 개입 자체의 효과 |
| matched_random | node 수, edge 수, 평균 degree 또는 weight scale | relation-specific structure 제거 | topology/smoothing만의 효과 |
| uniform | aggregation scale, smoothing operation | relation-specific weight 제거 | relation-free smoothing |
| graphfree_dominance_reweight | graph-free aggregation path, contribution control | relation structure 제거 | dominance correction |
| shuffled | graph structure, weight distribution | client identity-relation correspondence 변경 | relation assignment |
| clustering_only | coarse community structure | fine-grained edge relation 제거 | cluster 수준 정보 |

matched_random과 shuffled는 relation 의미를 깨는 control이다. 동시에 edge 재배치
과정에서 degree, entropy, smoothness, spectral energy도 달라질 수 있다.
real-control gap은 topology/spectral diagnostics와 함께 읽는다.

초기 분석에서는 `identity`, `matched_random`, `uniform`이 기본 비교군이다.
`graphfree_dominance_reweight`는 dominance correction 설명을 확인할 때 중요하다.
`shuffled`와 `clustering_only`는 relation assignment와 coarse community를 더 자세히
볼 때 붙는다.

---

## 7. Metrics And Evidence

지표는 세 단계로 읽는다.

| 분류 | 지표 | 역할 |
|---|---|---|
| primary decision | accuracy/loss, real-control gap, graph-free control gap | graph intervention의 성능 효과 |
| primary mechanism | alignment, DI, N_eff | relation quality와 dominance suppression |
| secondary diagnostics | LOO, density/degree/entropy, smoothness/spectral energy | 구조와 signal 변화 |

| Metric | 보는 이유 |
|---|---|
| real-control gap | real graph와 relation-destroyed control의 차이 |
| graph-free control gap | graph relation이 dominance correction보다 추가 정보를 주는지 |
| alignment | aggregate update가 client update 방향을 더 잘 대표하는지 |
| DI / N_eff | 소수 client dominance와 유효 참여 수의 변화 |
| density / degree / entropy | topology나 mixing 양의 영향 |
| smoothness / spectral energy | graph 위 update signal의 평탄화 |
| LOO | 특정 client 영향력의 크기 |

관찰 패턴은 다음처럼 읽힌다.

| Pattern | Observation | Interpretation |
|---|---|---|
| Relation-specific gain | real_graph > matched_random / shuffled / uniform, alignment 증가 | 실제 client relation이 단순 smoothing 이상 정보를 제공 |
| Generic smoothing | real_graph ≈ uniform 또는 matched_random | smoothing 자체가 주요 원인 |
| Coarse community | real_graph ≈ clustering_only | coarse group 정보가 충분 |
| Dominance correction | real_graph ≈ graphfree_dominance_reweight, DI 감소, N_eff 증가 | dominance suppression이 주요 원인 |
| Representation-sensitive gain | graph_source 변경에 따라 gap이 크게 달라짐 | relation 효과가 client representation에 민감 |
| Weak graph evidence | real_graph가 identity 또는 controls와 거의 차이 없음 | graph intervention의 설명력이 약함 |

---

## 8. Initial Evaluation

초기 실험은 update-cosine-kNN pipeline에서 시작한다.

| 항목 | 초기 설정 |
|---|---|
| Default graph source | client update delta |
| Similarity | cosine similarity |
| Topology | kNN graph |
| Normalization | row-normalized adjacency |
| Aggregation target | client update |
| Baselines | FedAvg, FedAvgM 또는 FedOpt |
| Essential controls | identity, matched_random, uniform |
| Important control | graphfree_dominance_reweight |
| Primary metrics | accuracy/loss, real-control gap, graph-free control gap |
| Core mechanism metrics | alignment, DI, N_eff |
| Secondary diagnostics | LOO checkpoint, graph density/degree/entropy, smoothness/spectral energy |

초기 dataset과 non-IID setting 후보는 다음과 같다.

| 항목 | 후보 |
|---|---|
| Vision dataset | FashionMNIST, CIFAR-10 |
| Optional extension | Cora-level graph-structured input dataset |
| Non-IID split | Dirichlet label skew |
| Alpha | 0.03, 0.1, 0.5 |
| Clients | 5, 10, 20 |
| Seeds | 최소 3개 |

Cora와 같은 graph-structured input dataset은 client relation graph와 구분해서
다룬다. 기본 실험은 federated clients 사이의 relation graph를 대상으로 한다.

---

## 9. Claim Shape

이 프레임워크의 claim은 controlled empirical attribution의 형태를 가진다.
직접적인 causal proof와 구분되는 경험적 해석이다. real graph가 matched controls와
graph-free controls를 모두 넘고, alignment, DI, N_eff에서도 일관된 변화가 보이면
graph-specific explanation의 강도가 높아진다. control과 거의 구분되지 않는
setting에서는 smoothing, clustering, dominance correction 쪽 설명이 더 강해진다.

주요 표현은 다음과 같다.

```text
relation-specific gain
graph-specific explanation
real-control gap
client-relation-dependent effect
```

---

## 10. Summary

Graph-FL 성능 향상은 graph relation, smoothing, clustering, dominance correction이
함께 만든 결과일 수 있다. 이 프레임워크는 Graph-FL 방법을 조립 가능한 구성요소로
나누고, real graph, relation-destroyed controls, graph-free controls를 같은 조건에서
비교한다.

그 결과는 새로운 graph를 만들기 위한 설계 공간이면서, 만든 graph가 어떤 조건에서
의미 있는지 확인하는 검증 구조가 된다.
