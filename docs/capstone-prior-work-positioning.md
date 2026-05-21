# Graph-FL Gain Attribution Framework

Graph-based federated learning은 client를 독립적인 평균 대상이 아니라 관계를
가진 집합으로 본다. client마다 데이터 분포, update 방향, 모델 상태가 다르기
때문에, 이 차이를 relation graph로 표현하려는 흐름은 자연스럽다. graph는
client similarity, cluster, topology, smoothing, personalized aggregation을
한 구조 안에 담을 수 있다.

최근 Graph-FL/PFL 연구는 이 방향으로 빠르게 확장됐다. FedAMP, SFL,
pFedGraph, FedAGA, FED-PUB/GPFL 같은 계열은 client similarity나 relation
graph를 이용해 non-IID 환경에서 협력 또는 개인화를 개선하려고 한다. 더 최근
흐름은 graph attention, hypernetwork, hypergraph, functional embedding,
spectral knowledge, fairness, privacy, robustness까지 넓어졌다.

문제는 graph가 유용해 보이는 결과 안에 여러 효과가 섞여 있다는 점이다. 어떤
방법은 graph 전체를 쓰지만, 어떤 방법은 graph에서 일부 성질만 떼어 단순화한다.
예를 들어 similarity만 쓰거나, sparse kNN만 남기거나, cluster 단위로 묶거나,
graph 없이 contribution/dominance만 보정하거나, graph smoothing과 비슷한
효과만 aggregation에 넣는다. 이런 흐름은 graph가 불필요하다는 뜻이 아니라,
graph 기반 성능 향상을 설명할 후보가 여러 개라는 뜻이다.

따라서 핵심 질문은 단순히 “graph를 쓰면 좋아지는가?”가 아니다.

```text
Q1. Graph-FL의 성능 향상은 실제 client relation 때문인가?
Q2. 그렇다면 어떤 graph 특성이 그 효과를 설명하는가?
```

Q1은 graph-specific effect의 존재 여부를 묻는다. Q2는 그 효과가 homophily,
density, sparsity, degree structure, community structure, temporal stability,
spectral smoothness 같은 구체적 특성 중 어디에서 나오는지 묻는다. 다만 첫
실험의 강한 주장은 relation, smoothing, coarse community, dominance
correction을 구분하는 데 둔다. topology, representation, spectral smoothness는
같은 실행에서 함께 기록하되, 1차 결론을 보강하거나 다음 실험 방향을 정하는
보조 해석 축으로 다룬다.

이 프로젝트는 새로운 Graph-FL SOTA 알고리즘을 추가하는 것보다, graph gain을
분해해 평가하는 프레임워크를 만드는 데 초점을 둔다. 가장 작은 범위에서도
`graph_source`, `graph_mode`, `aggregation_target`, control graph,
diagnostic metric을 같은 실행 환경에서 비교할 수 있으면 의미가 있다. 이후
Graph-FL 연구에서 새로운 graph 방법이 나와도 같은 축으로 비교하고 해석할 수
있기 때문이다.

결과가 잘 나오면 더 강한 주장으로 확장된다. real graph가 matched random,
shuffled, uniform, identity, graph-free control을 넘고, alignment, LOO, DI,
N_eff 같은 mechanism metric에서도 일관된 개선을 보이면 graph-specific
effect에 대한 controlled empirical attribution을 제시할 수 있다. 반대로 real
graph와 control의 차이가 작아도 실패가 아니다. 그 경우에는 Graph-FL 성능 향상이
relation 정보보다 smoothing, dominance correction, optimizer effect, coarse
clustering으로 더 잘 설명된다는 진단 결과가 된다.

## 프레임워크 구조

이 프레임워크의 강점은 실험을 많이 나열하는 데 있지 않다. graph를 만드는 방식,
graph를 적용하는 위치, 비교군, 진단 지표를 같은 실행 구조 안에 묶어 둔다는 데
있다. 그래서 한 방법이 좋아졌을 때 단순히 “성능이 올랐다”로 끝나지 않고, 어떤
종류의 설명이 더 그럴듯한지 바로 비교할 수 있다.

특히 graph design을 조립식으로 다룬다는 점이 핵심이다. 하나의 고정된 graph
알고리즘을 넣고 끝내는 것이 아니라, client 표현, similarity 계산, edge 선택,
weight 처리, normalization, aggregation target, control 변형을 서로 바꿔 끼울
수 있는 구조로 본다. 이 때문에 새로운 선행연구가 들어와도 전체 실험을 다시
짜기보다, 해당 연구가 바꾼 조각이 어느 위치인지 먼저 놓고 비교할 수 있다.
반대로 아직 문헌에 없는 graph 알고리즘을 떠올렸을 때도, 그것이 client
representation을 바꾸는 아이디어인지, relation score를 바꾸는 아이디어인지,
topology를 바꾸는 아이디어인지, aggregation target을 바꾸는 아이디어인지 바로
분해해 넣을 수 있다.

```text
client representation
-> similarity / relation score
-> topology construction
-> edge weight / normalization
-> aggregation target
-> correction or control variant
-> shared diagnostics
```

구조는 네 층으로 볼 수 있다.

| 층 | 역할 |
|---|---|
| execution layer | 같은 seed, config, runner, output schema에서 방법과 control을 실행 |
| graph design layer | `graph_source`, `graph_mode`, `aggregation_target`, `correction_family`로 graph intervention을 분해 |
| control layer | `real_graph`, `matched_random`, `shuffled`, `uniform`, `identity`, `clustering_only`, `graphfree_dominance_reweight`를 같은 조건에서 비교 |
| diagnosis layer | accuracy/loss와 함께 real-control gap, graph-free gap, alignment, LOO, DI, N_eff, graph/spectral metric을 저장 |

이 구조가 중요한 이유는 Graph-FL 연구에서 흔히 섞이는 요인을 분리해 주기
때문이다. 새 graph 방법과 기존 방법을 서로 다른 스크립트, 다른 출력 포맷, 다른
후처리 방식으로 비교하면 성능 차이가 graph 때문인지 실험 환경 때문인지 흐려진다.
반대로 같은 runner와 result schema 안에서 방법과 control을 실행하면, 결과가
어느 설명에 가까운지 더 안정적으로 해석할 수 있다.

프레임워크는 선행연구를 알고리즘 이름 그대로 복제하는 방식보다 mechanism 단위로
흡수한다. 어떤 연구가 client similarity를 썼다면 `graph_source`와
`graph_mode`로, personalized aggregation을 썼다면 `aggregation_target`으로,
dominance나 contribution 보정에 가까운 효과라면 `correction_family`로 놓는다.
이렇게 나누면 exact reproduction이 아니어도 해당 연구가 던진 아이디어를 같은
실험판 위에서 점검할 수 있다.

이 점은 최근 흐름과도 잘 맞는다. Graph-FL/PFL 연구는 하나의 정답 graph로
수렴하기보다, similarity source, sparse topology, learned graph, functional
embedding, attention, hypernetwork, fairness correction처럼 서로 다른 조각을
계속 바꾸며 확장되고 있다. 조립식 프레임워크는 이 변화 자체를 실험 대상으로
삼는다. 그래서 특정 논문 하나를 따라가는 데 그치지 않고, graph 기반 방법들이
어떤 구성요소를 바꿀 때 실제 이득이 생기는지 비교할 수 있다.

결과 파일 하나에 여러 지표를 함께 남기는 것도 이 프레임워크의 중요한 부분이다.
accuracy만 있으면 graph가 좋은지 나쁜지만 보이지만, alignment, LOO, DI, N_eff,
smoothness가 함께 있으면 relation, influence, dominance, smoothing 중 어떤
설명이 결과와 맞는지 볼 수 있다. 즉 이 구조의 산출물은 단순한 score table이
아니라, graph 효과를 해석하기 위한 evidence bundle에 가깝다.

## 선행연구에서 이어지는 질문

FedAvg/FedOpt 계열은 graph 없이도 non-IID 환경에서 일정한 안정화 효과를 낸다.
Personalized FL 계열은 모든 client를 같은 방식으로 취급하는 것이 충분하지
않다는 문제의식을 제공한다. Graph-FL/PFL 계열은 client relation을 직접
모델링해 협력 구조를 만든다.

이 흐름 다음에는 자연스럽게 평가 문제가 나온다.

```text
client relation graph가 유효하다면, 그 유효성은 정확히 어떤 요인으로 설명되는가?
```

기존 연구의 많은 부분은 “더 좋은 graph를 만들면 더 좋은 aggregation 또는
personalization이 가능하다”는 방향에 가깝다. 이 프로젝트는 그 흐름을 받아들이되,
성능 향상의 원인을 relation, topology, smoothing, clustering, dominance
suppression, representation source로 나누어 본다.

## 연구 가설

가설은 real graph가 항상 이긴다는 전제가 아니다. 결과가 어느 쪽으로 나오든
graph 효과를 설명하기 위한 해석 축이다. 1차 범위는 H1-H4다. H5-H7은 같은
result schema에 기록하지만, 초기 결론에서는 보조 진단으로 둔다.

| 범위 | 가설 | 관찰할 증거 | 해석 |
|---|---|---|---|
| primary | H1. Relation effect | `real_graph`가 matched random, shuffled, uniform, identity보다 높고 alignment/LOO도 개선 | 실제 client relation이 단순 smoothing 이상 정보를 제공 |
| primary | H2. Coarse community effect | `clustering_only`가 real graph와 비슷함 | fine-grained edge보다 cluster/homophily가 핵심 |
| primary | H3. Generic smoothing effect | uniform 또는 matched random이 real graph와 비슷함 | graph semantics보다 평균화/smoothing 자체가 주요 원인 |
| primary | H4. Dominance correction effect | `graphfree_dominance_reweight`가 real graph와 비슷하고 DI/N_eff가 안정화됨 | graph relation보다 큰 update/client 영향력 보정이 주요 원인 |
| secondary | H5. Topology effect | 같은 relation source에서도 density, degree, entropy, sparsity 변화에 따라 결과가 달라짐 | graph의 의미뿐 아니라 topology shape가 성능을 좌우 |
| secondary | H6. Representation effect | `graph_source`를 update, weight, EMA update, classifier head로 바꿀 때 결과가 달라짐 | 어떤 client representation으로 relation을 만드는지가 중요 |
| secondary | H7. Spectral/smoothness effect | low/high-frequency energy, smoothness, H_spec 변화가 성능 변화와 연결됨 | graph signal 관점에서 유효한 frequency 성분을 설명 가능 |

이 가설들이 구분하려는 설명은 다음과 같다.

```text
fine-grained relation
coarse clustering
generic smoothing
dominance suppression
topology shape
client representation
spectral smoothness
```

## 비교 축

선행연구와 새 방법은 알고리즘 이름보다 구성요소 단위로 비교한다.

| 축 | 질문 |
|---|---|
| `graph_source` | client를 무엇으로 표현했는가? update, weight, embedding, history 등 |
| `graph_mode` | relation을 어떤 graph로 만들었는가? kNN, RBF, learned graph, QP 등 |
| `aggregation_target` | graph를 어디에 적용했는가? update, weight, personalized model 등 |
| `correction_family` | real graph인지, control graph인지, graph-free correction인지 |
| diagnostics | accuracy 외에 alignment, LOO, DI, N_eff, smoothness 등을 보는가 |

이 축을 쓰면 full graph 방법과 graph에서 일부 효과만 떼어낸 단순화 방법을 같은
실험판에서 비교할 수 있다. 어떤 경우에는 full graph가 필요할 수 있고, 어떤
경우에는 clustering-only, uniform smoothing, graph-free dominance correction
같은 단순 대안으로 충분할 수 있다. 프레임워크의 가치는 이 경계를 실험적으로
찾는 데 있다.

## Control Set And Metrics

real graph 하나만으로는 graph-specific effect를 말하기 어렵다. 비교군은
relation 정보, topology, smoothing, clustering, graph-free correction을
나누어 보도록 구성한다.

| 비교군 | 의미 |
|---|---|
| `real_graph` | 의도한 client relation을 사용한 graph |
| `matched_random` | edge 수나 weight 조건을 맞춘 random graph |
| `shuffled` | client identity를 섞어 relation 의미를 깨는 graph |
| `uniform` | relation-free smoothing |
| `identity` | graph intervention 없음 |
| `clustering_only` | coarse group 효과 |
| `graphfree_dominance_reweight` | graph 없이 contribution/dominance만 보정 |

주요 지표는 같은 result schema 안에 함께 남기는 것이 좋다.

```text
accuracy / loss
real-control gap
graph-free control gap
DI / N_eff
alignment
LOO
graph density / degree / entropy
smoothness / spectral energy
```

지표는 단순히 많이 나열하는 것이 아니라 서로 다른 설명을 분리하기 위해 둔다.

| 지표 | 연결되는 설명 | 해석 방식 |
|---|---|---|
| `real-control gap` | relation effect | real graph가 relation-destroyed control보다 나은지 본다 |
| `graph-free control gap` | dominance correction vs graph relation | graph 없이도 비슷한 개선이 나는지 본다 |
| `alignment` | relation quality | 연결된 client의 update/model 방향이 실제로 가까운지 본다 |
| `LOO` | client influence | 특정 client 제거가 graph 성능에 미치는 영향을 본다 |
| `DI` / `N_eff` | dominance suppression | 소수 client가 aggregation을 지배하는지, 유효 참여 수가 안정화되는지 본다 |
| density / degree / entropy | topology effect | relation source가 같아도 graph shape가 결과를 바꾸는지 본다 |
| smoothness / spectral energy | smoothing effect | graph 위에서 update/model signal이 얼마나 평탄해지는지 본다 |

한 실행에서는 가능한 diagnostic을 함께 저장한다. 해석에서는 primary hypothesis와
직접 연결되는 `real-control gap`, `graph-free control gap`, `alignment`, `LOO`,
`DI`, `N_eff`를 먼저 보고, topology와 spectral metric은 보조 설명으로 붙인다.

## 선행연구 흡수 방식

선행연구를 모두 exact reproduction으로 처리하지 않는다. 현재 실행 가능한 부분과
확장 대상으로 남는 부분을 분리한다.

| 분류 | 의미 |
|---|---|
| `core-supported` | 현재 framework에서 직접 실행 가능 |
| `proxy-supported` | 원 논문의 exact reproduction이 아니라 핵심 mechanism에서 영감을 받은 근사 비교 |
| `interface-target` | hook은 있지만 별도 구현이 필요한 확장 대상 |
| `out-of-scope` | 현재 aggregation-level framework 밖의 별도 시스템 |

| 연구 계열 | 처리 |
|---|---|
| FedAMP-inspired | weight/RBF relation을 이용한 graph-filtered target proxy |
| SFL-inspired | learned/server-side graph operator proxy |
| pFedGraph-inspired | QP relation, sample-size prior, graph-filtered update proxy |
| FedAGA-inspired | EMA update/history-aware graph source proxy |
| FED-PUB/GPFL | functional embedding, personalized delivery는 interface-target |
| hypernetwork/GNN server aggregation | future extension 또는 interface-target |

이 구분은 과장을 줄이면서도 선행연구의 아이디어를 프레임워크 안에 배치할 수
있게 한다. exact reproduction이 아닌 경우에도 proxy-supported path를 통해
graph effect attribution 관점의 비교는 가능하다. 따라서 표현은 “FedAMP와
비교한다”보다 “FedAMP에서 영감을 받은 weight/RBF relation proxy와 비교한다”가
더 정확하다.

## 현재 범위

현재 산출물의 중심은 완성된 단일 알고리즘이 아니라 연구형 prototype과 실행 가능한
평가 구조다.

```text
1. Graph-FL/PFL 선행연구를 공통 축으로 분해한 비교 틀
2. graph_source / graph_mode / aggregation_target / correction_family 구조
3. real graph와 control graph를 같은 조건에서 실행하는 runner
4. DI, N_eff, alignment, LOO, graph/spectral metric을 남기는 result schema
5. 대표 vision/Cora smoke 실험이 가능한 prototype
6. claim boundary와 해석 틀이 정리된 실험 프로토콜
```

확장 대상으로 남는 범위:

```text
1. 모든 선행연구의 exact reproduction
2. personalized delivery, hypernetwork, GNN server aggregation의 완전 구현
3. SOTA accuracy 달성
4. semantic gain의 순수 causal proof
```

## 표현

`Semantic Gain`과 `Smoothing Gain`은 직관적 설명으로는 유용하지만, 직접
관측되는 순수량처럼 다루기 어렵다. 이 프레임워크의 성격은 causal proof가 아니라
controlled empirical attribution에 가깝다.

안전한 표현:

```text
Semantic contribution은 real graph와 matched control 사이의 estimated
real-control gap으로 해석한다.

Smoothing contribution은 uniform, identity, matched random, shuffled,
graph-free control 등 relation-free 또는 relation-destroyed control의
행동을 통해 추정한다.
```

피해야 할 표현:

```text
Semantic Graph - Random Graph = Pure Semantic Gain
```

대신 다음 표현이 더 안전하다.

```text
real graph가 matched controls와 graph-free controls를 모두 넘고,
alignment/LOO/DI/N_eff에서도 일관된 변화가 보이면 graph-specific explanation의
강도가 높아진다.
```

## 요약

```text
핵심은 graph를 사용했다는 사실이 아니라,
graph 사용으로 얻은 성능 향상을 여러 control과 diagnostic metric으로 분해해
해석 가능하게 만드는 것이다.
```
