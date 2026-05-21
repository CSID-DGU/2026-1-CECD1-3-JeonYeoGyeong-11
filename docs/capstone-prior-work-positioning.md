# Capstone Shared Research Direction

종합설계 1 문서 작성과 프로젝트 범위 확정을 위한 기준 문서다.
핵심은 선행연구를 단순 요약하지 않고, 본 프로젝트가 무엇을 평가하려는지
분명히 잡는 것이다.

## 프로젝트 정의

본 프로젝트는 새로운 Graph-FL SOTA 알고리즘을 제안하는 것이 아니다.

Graph-FL에서 성능 향상이 발생했을 때, 그 이유가 실제 client relation
정보 때문인지, 아니면 단순 smoothing, graph density, update norm/dominance
보정, optimizer 차이 같은 요인 때문인지 분해해서 검증하는 실험 프레임워크를
만든다.

공통 설명 문장:

```text
그래프 기반 연합학습에서 성능 향상의 원인을 control graph와 diagnostic metric으로
분해해 검증하는 실험 프레임워크
```

최소 성과 기준에서도 이 프로젝트는 의미가 있다. 모든 Graph-FL 계열 알고리즘을
완성하거나 SOTA 성능을 달성하지 못하더라도, graph source, graph mode,
aggregation target, control graph, diagnostic metric을 같은 규칙으로 실행하고
비교할 수 있는 프레임워크를 제공하면 이후 그래프 기반 연합학습 연구에 계속
재사용할 수 있다.

잘 작동할 경우에는 더 강한 주장으로 확장된다. real graph가 matched random,
shuffled, uniform, identity, graph-free control을 넘고, alignment, LOO,
DI, N_eff 같은 mechanism metric에서도 일관된 개선을 보이면 단순 성능 비교를
넘어 graph-specific effect에 대한 근거를 제시할 수 있다. 반대로 real graph와
control의 차이가 작아도 실패가 아니다. 그 경우에도 기존 Graph-FL 성능 향상이
relation 정보보다 smoothing, dominance correction, optimizer effect로 설명될
수 있음을 보여주는 진단 결과가 된다.

## 선행연구를 보는 관점

기존 연구는 중요하다. FedAMP, SFL, pFedGraph, FedAGA, FED-PUB/GPFL 같은
연구들은 client similarity, relation graph, personalized aggregation이
non-IID 환경에서 유효할 수 있음을 보여준다.

그래프가 점점 많이 쓰이는 이유는 자연스럽다. 연합학습의 client들은 독립적인
평균 대상이 아니라 서로 다른 데이터 분포, update 방향, 모델 상태를 가진 집합이다.
이 관계를 명시적으로 표현하려면 graph가 좋은 언어가 된다. graph는 client 간
유사도, 협력 가능성, cluster, topology, smoothing, personalized aggregation을
한 구조 안에서 표현할 수 있다.

동시에 많은 방법은 graph 전체를 그대로 쓰기보다 일부 성질만 떼어 단순화한다.
예를 들어 client similarity만 쓰거나, sparse kNN만 남기거나, cluster 단위로
묶거나, graph 없이 contribution/dominance만 보정하거나, graph smoothing과
비슷한 효과만 aggregation에 넣는 식이다. 이 흐름은 graph가 쓸모없다는 뜻이
아니라, graph 안에 여러 효과가 섞여 있다는 뜻이다.

하지만 많은 연구는 주로 최종 성능이나 개인화 성능을 중심으로 효과를 설명한다.
본 프로젝트의 질문은 조금 다르다.

```text
graph를 써서 좋아졌는가?
```

가 아니라,

```text
좋아진 이유가 정말 graph relation 때문인가?
```

이다.

이 질문은 선행연구와 자연스럽게 연결된다. FedAvg/FedOpt 계열은 graph 없이도
non-IID 환경에서 일정한 안정화 효과를 낸다. Personalized FL 계열은 모든
client를 같은 방식으로 취급하는 것이 충분하지 않다는 문제의식을 제공한다.
Graph-FL/PFL 계열은 client relation을 직접 모델링해 협력 구조를 만든다.
여기까지 오면 다음 질문이 자연스럽게 생긴다.

```text
client relation graph가 유효하다면, 그 유효성은 정확히 어떤 요인으로 설명되는가?
```

따라서 본 주제는 선행연구와 별개의 뜬금없는 아이디어가 아니라, Graph-FL/PFL
연구가 성능 개선을 보여준 이후 반드시 따라오는 평가 문제다.

이 때문에 프레임워크 자체가 중요하다. full graph 방법과 graph에서 일부 효과만
떼어낸 단순화 방법을 같은 실험판에서 비교할 수 있어야, 어떤 수준의 graph가
필요한지 판단할 수 있다. 경우에 따라 full graph가 필요할 수도 있고, 경우에 따라
random/shuffled/uniform, clustering-only, graph-free correction 같은 단순한
대안으로 충분할 수도 있다. 본 프로젝트는 그 경계를 실험적으로 찾기 위한 구조를
제공한다.

따라서 선행연구를 쓸 때는 “어떤 알고리즘이 성능이 좋다”가 아니라 다음 축으로
분해해서 본다.

| 축 | 질문 |
|---|---|
| `graph_source` | client를 무엇으로 표현했는가? update, weight, embedding, history 등 |
| `graph_mode` | relation을 어떤 graph로 만들었는가? kNN, RBF, learned graph, QP 등 |
| `aggregation_target` | graph를 어디에 적용했는가? update, weight, personalized model 등 |
| `correction_family` | real graph인지, control graph인지, graph-free correction인지 |
| diagnostics | accuracy 외에 alignment, LOO, DI, N_eff, smoothness 등을 봤는가 |

## 차별성

본 프로젝트의 차별점은 graph를 사용한다는 사실 자체가 아니다.

차별점은 graph 효과를 여러 대조군과 지표로 분해한다는 점이다.

| 일반적인 비교 | 본 프로젝트의 비교 |
|---|---|
| FedAvg보다 높은가 | real graph가 matched random, shuffled, uniform, identity, graph-free control도 넘는가 |
| 최종 accuracy 중심 | accuracy와 함께 DI, N_eff, alignment, LOO, graph/spectral metrics를 기록 |
| 알고리즘 단위 비교 | graph source/mode/target/correction family 단위로 분해 |
| 성능 향상 보고 | 성능 향상의 원인 해석 |

중심 문장:

```text
본 과제의 차별성은 Graph-FL 알고리즘 하나를 추가하는 것이 아니라,
Graph-FL 성능 향상을 relation effect, smoothing effect, dominance correction,
graph-free correction으로 분해해 평가하는 데 있다.
```

따라서 이 프레임워크는 두 종류의 연구에 모두 쓸 수 있다. 하나는 더 정교한 graph
relation이 실제로 필요한지 확인하는 연구이고, 다른 하나는 graph에서 핵심 효과만
남긴 단순화 방법이 충분한지 확인하는 연구다. 이 점이 단순 실험 코드가 아니라
그래프 기반 연합학습 연구를 위한 공용 평가 도구로서의 가치다.

이 차별성은 두 단계로 포장한다.

```text
최소 주장: Graph-FL 연구를 위한 reusable evaluation framework를 제공한다.
확장 주장: 충분한 실험 결과가 나오면 graph-specific effect의 존재와 한계를
control/diagnostic evidence로 주장할 수 있다.
```

## 실험에서 반드시 남겨야 하는 비교

real graph만 돌리면 안 된다. 최소한 아래 비교가 있어야 graph-specific claim을
말할 수 있다.

| 비교군 | 의미 |
|---|---|
| `real_graph` | 의도한 client relation을 사용한 graph |
| `matched_random` | edge 수나 weight 조건을 맞춘 random graph |
| `shuffled` | client identity를 섞어 relation 의미를 깨는 graph |
| `uniform` | relation-free smoothing |
| `identity` | graph intervention 없음 |
| `clustering_only` | coarse group 효과 |
| `graphfree_dominance_reweight` | graph 없이 contribution/dominance만 보정 |

주요 지표:

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

한 실험은 단순히 accuracy만 남기면 안 된다. 가능한 한 위 지표들이 같은 result
schema 안에 함께 남아야 한다.

## 선행연구 흡수 방식

선행연구를 모두 exact reproduction한다고 쓰지 않는다.

| 분류 | 의미 |
|---|---|
| `core-supported` | 현재 framework에서 직접 실행 가능 |
| `proxy-supported` | 핵심 mechanism을 현재 구조로 근사해 비교 가능 |
| `interface-target` | hook은 있지만 별도 구현이 필요한 확장 대상 |
| `out-of-scope` | 현재 aggregation-level framework 밖의 별도 시스템 |

예시:

| 연구 계열 | 본 프로젝트에서의 처리 |
|---|---|
| FedAMP-like | weight/RBF/graph-filtered target proxy |
| SFL-like | learned graph/server graph operator proxy |
| pFedGraph-like | QP relation, sample-size prior, graph-filtered update proxy |
| FedAGA-like | EMA update/history-aware graph source proxy |
| FED-PUB/GPFL | functional embedding, personalized delivery는 interface-target |
| hypernetwork/GNN server aggregation | future extension 또는 interface-target |

이렇게 쓰면 선행연구를 존중하면서도, 무엇을 구현했고 무엇을 확장 대상으로
남겼는지 과장 없이 정리할 수 있다.

## 종합설계 1에서 말할 수 있는 결과물

Capstone Design I 기준 산출물은 완성 제품이 아니라 연구형 prototype이다.

말할 수 있는 것:

```text
1. Graph-FL/PFL 선행연구를 공통 축으로 분해한 비교 기준
2. graph_source / graph_mode / aggregation_target / correction_family 구조
3. real graph와 control graph를 같은 조건에서 실행하는 runner
4. DI, N_eff, alignment, LOO, graph/spectral metric을 남기는 result schema
5. 대표 vision/Cora smoke 실험이 가능한 prototype
6. claim boundary와 해석 규칙이 정리된 실험 프로토콜
```

말하면 안 되는 것:

```text
1. 모든 선행연구의 exact reproduction
2. personalized delivery, hypernetwork, GNN server aggregation의 완전 구현
3. SOTA accuracy 달성
4. semantic gain의 순수 causal proof
```

## 표현 규칙

`Semantic Gain`과 `Smoothing Gain`은 발표용 직관으로는 쓸 수 있지만, 직접
관측되는 순수량처럼 쓰면 안 된다.

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

대신:

```text
real graph가 matched controls와 graph-free controls를 모두 넘고,
alignment/LOO/DI/N_eff에서도 일관된 변화가 보이면 graph-specific explanation의
강도가 높아진다.
```

## 최종 기준 문장

모든 양식과 발표는 아래 방향에서 벗어나지 않게 쓴다.

```text
본 프로젝트의 핵심은 graph를 사용했다는 사실이 아니라,
graph 사용으로 얻은 성능 향상을 여러 control과 diagnostic metric으로 분해해
해석 가능하게 만든다는 점이다.
```
