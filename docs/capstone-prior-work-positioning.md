# Graph-FL Gain Attribution Framework

## 1. Project Position

Graph-based Federated Learning(Graph-FL)은 client를 서로 독립적인 평균 대상이
아니라 관계를 가진 집합으로 본다. Non-IID 환경에서는 client마다 데이터 분포,
update 방향, 모델 상태가 다르기 때문에 client relation graph를 만들고 이를
aggregation이나 personalization에 활용하는 접근이 자연스럽게 등장한다.

하지만 Graph-FL에서 성능이 올랐다고 해서 곧바로 graph가 의미 있는 client
relation을 잘 잡았다고 말할 수는 없다. 같은 성능 향상은 여러 요인이 섞인 결과일
수 있다.

```text
1. relation-specific effect
2. generic smoothing effect
3. coarse clustering effect
4. dominance correction effect
5. optimizer or aggregation-path effect
```

예를 들어 real client graph를 사용했을 때 성능이 좋아졌더라도, 그것이 실제
client relation 때문인지, 단순히 update를 부드럽게 섞은 smoothing 효과 때문인지,
유사한 client를 대략적으로 묶은 clustering 효과 때문인지, 큰 update를 가진
client의 영향력을 줄인 dominance correction 효과 때문인지는 분리해서 봐야 한다.

이 프로젝트는 **Graph-FL gain의 원인을 control graph와 diagnostic metric을 통해
분해해 보는 attribution framework**를 만드는 데 초점을 둔다. 특정 graph 알고리즘
하나에 고정되지 않고, Graph-FL 방법을 조립 가능한 설계 요소로 나누어 같은 조건에서
비교한다.

핵심 질문은 다음과 같다.

```text
Q1. Graph-FL의 성능 향상은 실제 client relation graph 때문인가?
Q2. 아니면 smoothing, clustering, dominance correction 같은 대안적 설명으로도 충분한가?
Q3. real graph가 control보다 낫다면, 어떤 mechanism metric이 그 차이를 설명하는가?
Q4. 새로운 graph-based intervention을 같은 구조 안에서 조립하고 비교할 수 있는가?
```

---

## 2. Framework Flow

프레임워크의 큰 흐름은 graph를 만들고, 같은 조건의 control과 비교하고, 같은 schema로
진단 결과를 남기는 것이다.

```text
client representation
-> relation score
-> topology construction
-> edge weight / normalization
-> aggregation target
-> control or correction family
-> shared diagnostics
```

이 흐름은 graph authoring과 graph validation을 분리하지 않는다. 새로운 graph
후보를 만들면 같은 control suite에 올리고, 그 graph가 실제 relation 정보를
살리는지, 단순 smoothing에 가까운지, coarse clustering만으로 충분한지, dominance
correction으로도 대체 가능한지 판단한다.

하나의 실험 run은 가능한 한 동일한 seed, 동일한 non-IID split, 동일한 aggregation
target에서 real graph와 control variants를 함께 실행한다. 각 variant의 성능과
mechanism metric은 같은 schema로 저장한다.

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

결과 해석은 `real_graph가 accuracy가 높다`에서 끝나지 않는다. real graph가 어떤
control보다 높은지, relation-destroyed control에서도 차이가 유지되는지,
graph-free dominance correction으로도 비슷한 결과가 나는지, alignment, DI,
N_eff가 성능 변화와 함께 움직이는지를 함께 본다.

---

## 3. Current Implementation Snapshot

현재 구현은 단순한 아이디어 스케치가 아니라, client relation graph를 만들고,
control과 함께 실행하며, 결과를 같은 schema로 남기는 연구형 runtime에 가깝다.
다만 모든 Graph-FL/PFL 선행연구를 완전히 재현하는 시스템은 아니다. 현재 구현의
중심은 **client relation graph 기반 aggregation-level intervention과 diagnostic
attribution**이다.

여기서 말하는 graph는 기본적으로 **client relation graph**다. node는 client이고,
edge는 client 사이의 update/model/relation similarity를 나타낸다. Cora와 같은
graph-structured input data는 확장 검증으로 다룰 수 있지만, 기본 관심사는 input
data graph가 아니라 federated clients 사이의 relation graph다.

현재 코드에서 확인되는 실행 단위는 다음과 같다.

| 구현 단위 | 현재 상태 | 의미 |
|---|---|---|
| package/runtime | `graphfl_lab` package, Flower entrypoint, top-level runner wrappers | graph-FL diagnostic 실험을 실행하는 기본 runtime |
| default graph design | `default_similarity_knn` | update 기반 similarity-kNN graph의 기본 design |
| graph source | update, EMA update, classifier-head update, weight 계열 source | client를 어떤 신호로 표현할지 선택 |
| graph construction | cosine/kNN 계열, magnitude/RBF/proxy 계열, uniform/random/shuffled/control graph 계열 | relation graph와 relation-destroyed control을 구성 |
| aggregation target | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` | graph를 update, EMA update, weight signal에 적용 |
| control family | identity, matched random, shuffled, uniform, clustering-only, graph-free dominance reweight | graph gain의 대안 설명을 분리 |
| diagnostics/schema | `result_schema_version`, `config_aliases_used`, `unsupported_components`, alignment, DI, N_eff, graph/spectral metric | 결과를 같은 schema로 저장하고 해석 |
| executable configs | vision diagnostic configs, Cora graph ablation initial config | vision setting과 graph-dataset extension에서 초기 실행 가능 |
| validation tests | graph/design/diagnostic/schema/runner 관련 unit tests | graph design과 result contract가 코드로 고정됨 |

현재 직접 지원하는 대표 pipeline은 다음 흐름이다.

```text
client update delta
-> cosine similarity
-> kNN graph
-> row-normalized adjacency
-> graph-filtered update aggregation
-> shared result schema and diagnostics
```

이 pipeline은 가장 해석하기 쉬운 출발점이다. client의 원본 데이터를 직접 모으지
않는 FL setting에서, client update delta를 relation signal로 사용해 client graph를
간접적으로 구성한다. 이후 같은 setting에서 real graph와 controls를 실행해
relation-specific effect, generic smoothing, coarse clustering, dominance
correction 가능성을 비교한다.

현재 구현은 조립식 확장을 위한 entrypoint도 갖고 있다. 새로운 방법은 전체 runtime을
다시 만드는 방식이 아니라 `graph_source`, `graph_mode` 또는 graph design,
`aggregation_target`, `correction_family` 중 어느 조각을 바꾸는지로 들어온다.
예를 들어 classifier-head relation은 source 교체, EMA-history relation은 source와
target 교체, RBF-style relation은 relation score/weight 교체, graph-free dominance
control은 correction family 교체로 표현한다.

구현 범위는 다음처럼 나누어 둔다.

| 분류 | 의미 |
|---|---|
| core-supported | 현재 runner/config/test로 직접 실행 가능한 graph design과 diagnostic path |
| proxy-supported | 원 논문의 exact reproduction은 아니지만 핵심 mechanism을 근사해 비교 가능한 path |
| interface-target | hook이나 설계 위치는 있지만 별도 구현이 필요한 확장 대상 |
| out-of-scope | 현재 aggregation-level framework 밖의 별도 시스템 |

따라서 현재 구현의 핵심은 “모든 graph 알고리즘을 완성했다”가 아니라, graph-based
intervention을 공통 runtime 위에 올리고, control과 diagnostic으로 검증할 수 있는
구조를 이미 갖추었다는 점이다.

---

## 4. Modular Design Space

프레임워크에서 graph-based intervention은 다음 조각들로 분해된다.

| 조립 위치 | 만들 수 있는 것 | 의미 |
|---|---|---|
| client representation | update delta, model weight, EMA update, classifier-head update, classifier-head weight | client를 어떤 신호로 표현할지 선택 |
| relation score | cosine similarity, magnitude-aware similarity, RBF-style similarity, learned/proxy score | client 사이의 관계를 어떻게 계산할지 선택 |
| topology construction | kNN graph, threshold graph, dense graph, matched random graph, shuffled graph, clustering-only graph | relation score를 edge 구조로 바꾸는 방식 |
| edge weight / normalization | binary edge, similarity weight, row-normalized adjacency, smoothing strength, sample-size prior | 연결 강도와 mixing 정도 조절 |
| aggregation target | graph-filtered update, graph-filtered EMA update, graph-filtered weight, personalized/model-mixture target | graph를 어떤 학습 신호에 적용할지 선택 |
| correction/control family | real graph, matched random, shuffled, uniform, identity, clustering-only, graph-free dominance reweight | 성능 향상의 설명 후보 분리 |
| diagnostics | accuracy/loss, real-control gap, graph-free gap, alignment, DI, N_eff, LOO, topology/spectral metrics | graph effect를 해석하기 위한 지표 |

이 설계 공간은 각 graph method가 어느 구성요소를 바꾸는지 확인하기 위한 것이다.
각 변화가 어떤 설명과 연결되는지 작은 단위로 나누어 보면, graph gain을 relation,
smoothing, clustering, dominance, topology, representation 효과로 더 명확히
해석할 수 있다.

이 점에서 프레임워크의 장점은 Graph-FL 방법을 단일 알고리즘이 아니라 조립 가능한
실험 공간으로 다룬다는 데 있다. 새로운 graph construction 방식을 추가하더라도,
동일한 control suite와 metric을 그대로 사용해 기존 variant와 비교할 수 있다.

---

## 5. Success Conditions

이 프로젝트의 성공은 real graph가 항상 최고 성능을 내는 것이 아니다. 오히려
성공은 graph gain이 어떤 설명과 더 잘 맞는지 분리해 말할 수 있는 상태에 가깝다.

```text
1. client update 기반 real graph를 구성할 수 있다.
2. real graph와 matched control graph를 같은 조건에서 실행할 수 있다.
3. graph-free control과도 비교할 수 있다.
4. performance gap과 mechanism metric을 같은 schema로 기록할 수 있다.
5. 관찰된 gain을 relation-specific, smoothing, clustering, dominance correction 중 어느 설명과 더 잘 연결되는지 판단할 수 있다.
```

따라서 real graph가 control보다 높게 나오지 않아도 프로젝트가 실패하는 것은 아니다.
예를 들어 uniform control이 real graph와 비슷하다면, 해당 setting에서는
relation-specific effect보다 generic smoothing effect가 더 크다는 해석이 가능하다.
`graphfree_dominance_reweight`가 real graph와 비슷하다면, graph relation보다
dominance correction이 주요 원인일 수 있다.

---

## 6. Research Questions and Hypotheses

| 구분 | RQ | 질문 | 주요 비교 |
|---|---|---|---|
| attribution | RQ1 | real client relation graph는 relation-destroyed control보다 일관되게 나은가? | real_graph vs matched_random / shuffled / identity |
| attribution | RQ2 | 관찰된 graph gain은 generic smoothing, coarse clustering, dominance correction으로 대체 설명 가능한가? | real_graph vs uniform / clustering_only / graphfree_dominance_reweight |
| mechanism | RQ3 | real graph가 control을 넘을 때 alignment, DI, N_eff는 어떤 패턴을 보이는가? | performance gap + mechanism metrics |
| design | RQ4 | graph gain은 graph operator 자체보다 client representation 또는 topology choice에 더 민감한가? | update / EMA / classifier-head source, kNN / threshold / dense topology |
| framework | RQ5 | 새로운 graph-based intervention을 공통 interface 안에서 조립하고 같은 조건으로 비교할 수 있는가? | graph_source / relation_score / topology / target 조합 비교 |

이 RQ들은 real graph의 우위를 전제하지 않고, Graph-FL gain을 어떤 설명으로 해석할
수 있는지 구분하는 데 초점을 둔다.

| 범위 | 가설 | 관찰할 증거 | 해석 |
|---|---|---|---|
| primary | H1. Relation-specific effect | real_graph가 matched_random, shuffled, uniform, identity보다 높고 alignment도 개선 | 실제 client relation이 단순 smoothing 이상 정보를 제공 |
| primary | H2. Coarse community effect | clustering_only가 real_graph와 비슷함 | fine-grained edge보다 cluster/homophily가 핵심 |
| primary | H3. Generic smoothing effect | uniform 또는 matched_random이 real_graph와 비슷함 | graph relation보다 평균화/smoothing 자체가 주요 원인 |
| primary | H4. Dominance correction effect | graphfree_dominance_reweight가 real_graph와 비슷하고 DI/N_eff가 안정화됨 | graph relation보다 큰 update/client 영향력 보정이 주요 원인 |
| secondary | H5. Topology effect | 같은 relation source에서도 density, degree, entropy, sparsity 변화에 따라 결과가 달라짐 | graph shape가 성능을 좌우 |
| secondary | H6. Representation effect | graph_source 변경에 따라 결과가 달라짐 | 어떤 client representation으로 relation을 만드는지가 중요 |
| secondary | H7. Spectral/smoothness effect | smoothness, low/high-frequency energy 변화가 성능 변화와 연결됨 | graph signal 관점에서 smoothing 효과 설명 가능 |

---

## 7. Control Design

Graph-specific effect를 해석하려면 real graph와 controls가 가능한 한 같은 graph
construction pipeline을 공유해야 한다.

```text
same representation
same density or comparable edge budget
same normalization
same aggregation target
different tested factor only
```

control 간 차이는 relation 의미, client identity 대응, fine-grained edge 구조,
graph-free correction처럼 검증하려는 요인에만 둔다. 이 원칙을 통해 real-control
gap을 더 안정적으로 해석한다.

matched_random과 shuffled는 relation 정보를 약화하는 비교군이지만, edge 재배치
과정에서 topology나 spectral property도 함께 달라질 수 있다. 따라서 real-control
gap은 단독으로 해석하지 않고, density, degree, entropy, smoothness, spectral
energy와 함께 본다.

| Control | Preserved factor | Broken / removed factor | Purpose |
|---|---|---|---|
| real_graph | client representation, similarity score, topology, edge weight | 없음 | 의도한 client relation을 사용한 graph |
| matched_random | node 수, edge 수, 평균 degree 또는 weight scale | relation-specific structure | topology/smoothing만으로 gain이 나는지 확인 |
| shuffled | graph structure, weight distribution | client identity-relation correspondence | relation assignment가 중요한지 확인 |
| uniform | aggregation scale, smoothing operation | relation-specific weight | 단순 relation-free smoothing과 비교 |
| identity | original FedAvg-style update path | graph intervention | graph 개입 자체의 효과 확인 |
| clustering_only | coarse community structure | fine-grained edge relation | cluster 수준 정보만으로 충분한지 확인 |
| graphfree_dominance_reweight | graph-free aggregation path, contribution control | relation structure | dominance correction만으로 gain을 대체할 수 있는지 확인 |

초기 분석에서는 모든 control을 같은 비중으로 다루지 않는다. 우선순위는 다음과 같이
둔다.

| 우선순위 | Control | 이유 |
|---|---|---|
| essential | identity | graph intervention 자체의 효과 확인 |
| essential | matched_random | relation-specific structure 제거 |
| essential | uniform | generic smoothing과 비교 |
| important | graphfree_dominance_reweight | dominance correction 대체 설명 확인 |
| optional | shuffled | client identity-relation correspondence 확인 |
| optional | clustering_only | coarse community effect 확인 |

---

## 8. Metrics and Evidence Patterns

주요 지표는 같은 result schema 안에 함께 남기되, 해석 우선순위를 나눈다. 본문에서는
attribution 판단에 직접 필요한 지표를 중심으로 설명하고, 수식과 세부 caveat는
appendix의 metric reference로 분리한다.

| 우선순위 | 지표 | 역할 |
|---|---|---|
| primary decision | accuracy/loss, real-control gap, graph-free control gap | graph intervention이 성능상 의미 있는지 판단 |
| primary mechanism | alignment, DI, N_eff | relation quality와 dominance suppression이 함께 움직이는지 확인 |
| secondary diagnostics | LOO, density/degree/entropy, smoothness/spectral energy | 보조적인 구조 해석 제공 |

초기 분석에서는 핵심 mechanism metric을 `alignment`, `DI`, `N_eff` 중심으로 둔다.
나머지 LOO, topology, spectral metric은 결과 해석을 보조하는 diagnostic으로 사용한다.

| Group | Metric | Role |
|---|---|---|
| Intervention / aggregate behavior | real-control gap, graph-free control gap, alignment | real graph가 control보다 나은지, aggregate 방향이 client update를 잘 대표하는지 확인 |
| Client contribution confounders | contribution share, DI, N_eff | graph gain이 relation 때문인지, 특정 client의 dominance 완화 때문인지 확인 |
| Graph structure confounders | density, degree, entropy | edge 수, hub 구조, weight 분포가 결과를 만든 것인지 확인 |
| Graph mechanism descriptors | smoothness, spectral energy, temporal stability | graph 위 update signal이 어떤 방식으로 변하는지 보조적으로 해석 |

Core metric은 다음처럼 해석한다.

| Metric | 간단한 정의 | 보는 이유 |
|---|---|---|
| real-control gap | real_graph와 control_graph의 score 차이 | relation-specific effect의 1차 증거 |
| graph-free control gap | real_graph와 graph-free correction 사이의 차이 | graph relation이 dominance correction보다 추가 정보를 주는지 확인 |
| alignment | client update와 aggregate update의 방향 유사도 | real graph가 update 방향을 더 잘 대표하는지 확인 |
| contribution share | sample-size weight와 update norm을 함께 고려한 client 영향력 근사 | 특정 client가 aggregation을 지배하는지 확인 |
| DI | 가장 큰 contribution share | 한 client dominance가 graph gain을 설명하는지 확인 |
| N_eff | contribution이 실질적으로 몇 client에 분산되는지 | 여러 client가 고르게 반영되는지 확인 |
| density / degree / entropy | graph 연결 수, hub 구조, weight 분포 | topology나 smoothing 양이 결과를 만든 것인지 확인 |
| smoothness | graph 위 update signal의 평탄성 | real graph가 update relation을 반영하는지, 또는 over-smoothing인지 확인 |
| LOO | 특정 client 제거 시 aggregate 방향 변화 | 특정 client 영향력이 과도한지 확인 |

각 metric은 단독으로 결론을 내기보다 함께 해석한다. 예를 들어 real-control gap이
양수라도 DI가 크게 줄고 N_eff가 증가했다면, 성능 향상은 relation-specific effect보다
dominance correction에 가까울 수 있다. 반대로 real graph가 matched_random과
uniform을 넘고 alignment도 함께 개선된다면, relation-specific explanation의 강도가
높아진다.

실험 결과를 사후적으로 끼워 맞추지 않기 위해, 가능한 관찰 패턴과 해석을 미리
정리한다.

| Pattern | Observation | Supporting metric | Interpretation |
|---|---|---|---|
| P1. Relation-specific gain | real_graph > matched_random / shuffled / uniform | alignment 증가, real-control gap 양수 | 실제 client relation이 단순 smoothing 이상 정보를 제공 |
| P2. Generic smoothing | real_graph ≈ uniform 또는 matched_random | smoothness 증가, alignment 개선 약함 | graph relation보다 smoothing 자체가 주요 원인 |
| P3. Coarse community | real_graph ≈ clustering_only | cluster-level consistency | fine-grained edge보다 coarse group 정보가 충분 |
| P4. Dominance correction | real_graph ≈ graphfree_dominance_reweight | DI 감소, N_eff 증가 | graph relation보다 dominance suppression이 주요 원인 |
| P5. Representation-sensitive gain | graph_source 변경에 따라 real-control gap이 크게 달라짐 | source별 gap 변화 | relation 효과는 client representation에 민감 |
| P6. Topology-sensitive gain | 같은 source에서도 kNN/dense/threshold에 따라 결과가 달라짐 | density, degree, entropy 변화 | relation signal뿐 아니라 topology shape가 성능을 좌우 |
| P7. Weak graph evidence | real_graph가 identity 또는 controls와 거의 차이 없음 | gap 작음, mechanism metric 변화 약함 | 해당 setting에서는 graph intervention의 설명력이 약함 |

---

## 9. Initial Evaluation Protocol

초기 실험은 가장 해석 가능한 최소 단위에서 시작한다.

| 항목 | 초기 설정 |
|---|---|
| Default graph source | client update delta |
| Similarity | cosine similarity |
| Topology | kNN graph |
| Normalization | row-normalized adjacency |
| Aggregation target | client update |
| Baselines | FedAvg, FedAvgM 또는 FedOpt |
| Essential controls | identity, matched_random, uniform |
| Important controls | graphfree_dominance_reweight |
| Optional controls | shuffled, clustering_only |
| Primary metrics | accuracy/loss, real-control gap, graph-free control gap |
| Core mechanism metrics | alignment, DI, N_eff |
| Secondary diagnostics | LOO checkpoint, graph density/degree/entropy, smoothness/spectral energy |

초기 dataset과 non-IID setting은 다음처럼 둔다.

| 항목 | 후보 |
|---|---|
| Vision dataset | FashionMNIST, CIFAR-10 |
| Optional extension | Cora-level graph-structured input dataset |
| Non-IID split | Dirichlet label skew |
| Alpha | 0.03, 0.1, 0.5 |
| Clients | 5, 10, 20 |
| Seeds | 최소 3개 |
| Rounds | prototype에서는 짧은 round, main experiment에서는 충분한 round로 확장 |

초기 실험은 같은 setting에서 real_graph와 controls가 어떤 차이를 보이는지 확인하고,
그 차이가 어떤 설명과 더 잘 맞는지 판단하는 데 둔다.

Cora와 같은 graph-structured input dataset은 현재 client relation graph와 구분해서
다룬다. 기본 대상은 client relation graph이며, graph input data setting은 선행연구
맥락을 확인하기 위한 확장 검증으로 둔다.

---

## 10. Prior Work Absorption and Extension Boundary

선행연구는 현재 실행 가능한 부분과 확장 대상으로 남는 부분을 분리해 흡수한다.

현재 중심은 update-based client relation graph와 aggregation-level intervention이다.
기존 Graph-FL/PFL 방법들은 이 프레임워크의 확장 대상으로 정리하되, 모든 방법의
exact reproduction은 현재 범위에 포함하지 않는다.

| 연구 계열 | 처리 |
|---|---|
| FedAMP-inspired | weight/RBF relation을 이용한 graph-filtered target proxy |
| SFL-inspired | learned/server-side graph operator proxy |
| pFedGraph-inspired | QP relation, sample-size prior, graph-filtered update proxy |
| FedAGA-inspired | EMA update/history-aware graph source proxy |
| FED-PUB/GPFL | functional embedding, personalized delivery는 interface-target |
| hypernetwork/GNN server aggregation | future extension 또는 interface-target |

이 구분을 통해 exact reproduction과 mechanism proxy를 분리한다. proxy-supported 항목은
“해당 논문에서 영감을 받은 mechanism proxy와 비교한다”고 표현한다.

중요한 점은 이 섹션이 현재 구현 범위를 넓히기 위한 것이 아니라, 프레임워크가 향후
어떤 Graph-FL/PFL 계열을 흡수할 수 있는지 정리하기 위한 확장 지도라는 점이다.

---

## 11. Claim Boundary

현재 산출물의 중심은 graph-based intervention을 구성하고 비교할 수 있는 연구형
prototype과 평가 구조다.

```text
1. Graph-FL/PFL 선행연구를 공통 축으로 분해한 비교 틀
2. graph_source / graph_mode / aggregation_target / correction_family 구조
3. update-based client relation graph construction
4. real graph와 control graph를 같은 조건에서 실행하는 runner
5. primary decision/mechanism metric과 secondary diagnostic을 남기는 result schema
6. 대표 vision dataset에서 초기 검증이 가능한 prototype
7. claim boundary와 해석 틀이 정리된 실험 프로토콜
```

확장 대상은 다음과 같다.

```text
1. 모든 선행연구의 exact reproduction
2. personalized delivery, hypernetwork, GNN server aggregation의 완전 구현
3. SOTA accuracy 달성
4. semantic gain의 순수 causal proof
5. 모든 graph generator 조합의 완전 탐색
6. graph-structured input dataset에 대한 완전한 일반화
```

이 프레임워크의 목표는 controlled empirical attribution이다. 다음 표현은 사용하지
않는다.

```text
Semantic Graph - Random Graph = Pure Semantic Gain
```

claim은 다음과 같이 정리한다.

```text
real graph가 matched controls와 graph-free controls를 모두 넘고,
alignment/DI/N_eff에서도 일관된 변화가 보이면
graph-specific explanation의 강도가 높아진다.
```

`semantic gain` 대신 다음 표현을 우선 사용한다.

```text
relation-specific gain
graph-specific explanation
real-control gap
client-relation-dependent effect
```

이 연구는 Graph-FL gain이 relation-destroyed controls 및 graph-free controls와
비교했을 때 얼마나 graph-specific하게 설명될 수 있는지를 평가한다.

---

## 12. Summary

Graph-FL 방법의 성능 향상이 곧 의미 있는 client relation을 활용했다는 증거는 아니다.

이 프레임워크는 Graph-FL 방법을 구성요소 단위로 분해하고, real graph,
relation-destroyed controls, graph-free controls를 같은 조건에서 비교한다.

이를 통해 Graph-FL gain이 relation-specific effect인지, generic smoothing인지,
coarse clustering인지, dominance correction인지, 또는 해당 setting에서 약한
evidence인지 판단한다.
