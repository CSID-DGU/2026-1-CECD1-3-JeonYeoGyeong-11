# Capstone Shared Research Direction

이 문서는 종합설계 1 양식을 쓰기 전에 팀원들이 같은 방향을 보고 움직이기
위한 공용 기준서다. 목적은 문장을 예쁘게 만드는 것이 아니라, 우리가 무엇을
하는 프로젝트인지, 선행연구를 어떻게 받아들일 것인지, 어디를 차별점으로
잡을 것인지 오해 없이 맞추는 데 있다.

우리 프로젝트의 출발점은 연합학습에서 client들이 서로 다른 데이터 분포를
가진다는 문제다. FedAvg처럼 모든 client update를 단순 평균하면 구조는
간단하지만, 어떤 client들이 서로 비슷한 방향으로 학습하고 있는지, 어떤
client update가 전체 방향과 충돌하는지, 특정 client가 지나치게 큰 영향력을
갖는지 같은 정보는 잘 드러나지 않는다. 그래서 많은 선행연구는 client 간
유사도, personalized aggregation, graph-based collaboration 같은 방식으로
non-IID 문제를 완화하려고 했다.

이 흐름은 우리에게 중요하다. 기존 연구들이 틀렸다는 뜻이 아니다. 오히려
그 연구들은 "client 사이의 관계를 이용하면 성능이나 개인화가 좋아질 수
있다"는 가능성을 보여준다. FedAMP 계열은 model distance와 attentive
collaboration을, SFL 계열은 client relation graph와 server-side graph
operator를, pFedGraph 계열은 learned collaboration graph와 client-specific
mixture를, FED-PUB/GPFL 계열은 functional embedding 기반 개인화를 보여준다.
이들은 모두 우리가 다루는 문제의 선행 흐름이다.

하지만 우리가 잡는 질문은 조금 다르다. 기존 연구가 주로 "어떤 graph
collaboration이 성능을 높이는가"에 집중했다면, 우리는 "그 성능 향상이 정말
graph relation 때문인지 어떻게 확인할 것인가"를 묻는다. graph를 쓰면 성능이
올랐다는 결과만으로는 부족하다. 같은 graph operation 안에는 실제 의미 있는
client relation뿐 아니라 단순 smoothing, graph density 효과, update norm
억제, dominance correction, optimizer 차이 같은 요인이 섞일 수 있다. 그래서
우리 프로젝트는 graph를 쓰는 알고리즘 하나를 더 제안하는 것보다, graph 효과를
분해해서 평가하는 구조를 만드는 데 초점을 둔다.

팀 내부에서 이 프로젝트를 가장 짧게 설명하면 다음과 같다.

```text
그래프 기반 연합학습에서 성능 향상이 왜 발생했는지 분해해 검증하는
실험 프레임워크
```

조금 더 정확히 쓰면 다음 문장이 기준이다.

```text
본 과제는 Graph-FL에서 real graph가 보여주는 성능 향상이 실제 client
relation 정보 때문인지, 아니면 random/shuffled/uniform/identity graph나
graph-free dominance correction으로도 설명되는지를 동일한 실험 환경에서
비교하고, DI, N_eff, alignment, LOO, graph/spectral diagnostics로 해석하는
프레임워크를 구현한다.
```

여기서 중요한 점은 우리가 선행연구를 단순히 "정리"하는 것이 아니라는 점이다.
종합설계의 선행기술 조사에서는 선행연구가 무엇을 했는지 나열하는 데서 끝나면
안 된다. 각 연구가 어떤 client state를 보고, 어떤 relation을 만들고, 그 graph를
어디에 적용했으며, 어떤 control 없이 성능을 해석했는지를 평가해야 한다. 즉,
선행연구를 알고리즘 이름 단위로만 보는 것이 아니라 공통 구성요소로 분해해서
본다.

우리가 사용할 공통 축은 다음과 같다.

| 축 | 의미 |
|---|---|
| `graph_source` | client를 무엇으로 표현하는가: update, weight, EMA update, classifier head, functional embedding 등 |
| `graph_mode` | relation을 어떤 graph로 만드는가: kNN, RBF, learned graph, QP, random, shuffled 등 |
| `aggregation_target` | graph를 어디에 적용하는가: update, EMA update, weight, personalized model delivery 등 |
| `correction_family` | real graph인지, control graph인지, graph-free correction인지 |
| diagnostics | 성능뿐 아니라 alignment, LOO, DI, N_eff, graph metric, spectral metric을 보는가 |

이 축으로 보면 선행연구는 우리 프로젝트 안에서 자연스럽게 흡수된다. FedAMP나
SFL, pFedGraph 같은 연구를 "완전히 재현한다"고 말하지 않는다. 대신 그 연구의
핵심 mechanism을 현재 framework에서 proxy로 비교할 수 있는지, 아니면
client-specific model delivery나 hypernetwork처럼 별도 확장이 필요한
interface-target인지 구분한다. 이 구분이 중요하다. exact reproduction을
약속하지 않으면서도, 선행연구의 아이디어를 우리의 실험 프레임 안에서 비교
가능한 형태로 배치할 수 있기 때문이다.

따라서 문서에서 선행연구를 쓸 때의 톤은 다음과 같아야 한다.

```text
선행연구는 client relation graph 또는 personalized collaboration을 이용해
non-IID 환경에서 성능과 개인화를 개선해 왔다. 본 과제는 이 흐름을 이어받되,
성능 향상분이 실제 graph relation 때문인지, 단순 smoothing이나 contribution
보정 때문인지 분리해 검증하는 평가 프레임워크를 구축한다.
```

이 표현은 선행연구를 깎아내리지 않으면서도 우리 프로젝트의 위치를 분명하게
만든다. "기존 연구는 전혀 하지 않았다"라고 쓰면 과장이다. 대신 "기존 연구는
주로 성능 개선과 개인화에 초점을 두었고, 우리는 graph-specific effect의
검증과 해석에 초점을 둔다"라고 쓰는 것이 안전하고 강하다.

우리 프로젝트가 특별해 보이는 지점은 네 가지다. 첫째, claim을 보수적으로
잡는다. SOTA 정확도나 완전한 새 알고리즘을 주장하지 않고, 성능 향상의 원인을
분해하는 실험 프레임워크를 주장한다. 둘째, 비교군이 강하다. FedAvg 하나와
비교하는 것이 아니라 real graph를 matched random, shuffled, uniform, identity,
clustering-only, graph-free dominance reweighting과 비교한다. 셋째, 지표를
한 번에 남긴다. accuracy/loss와 함께 DI, N_eff, alignment, LOO, graph density,
entropy, smoothness, spectral energy를 같은 결과 schema 안에서 본다. 넷째,
선행연구를 확장 가능한 구조로 흡수한다. 어떤 연구는 proxy-supported로, 어떤
연구는 interface-target으로 분류해서 현재 구현과 미래 확장 사이의 경계를
분명히 한다.

종합설계 1에서 완성했다고 말해야 하는 것은 완전한 제품이 아니라 연구형
prototype이다. 우리가 목표로 하는 산출물은 다음과 같다.

```text
1. 선행 Graph-FL/PFL 연구를 공통 구성요소로 분해한 비교 기준
2. graph_source / graph_mode / aggregation_target / correction_family 구조
3. real graph와 control graph를 같은 조건에서 실행하는 runner
4. DI, N_eff, alignment, LOO, graph/spectral metric을 함께 남기는 result schema
5. 대표 vision/Cora smoke 실험이 가능한 prototype
6. claim boundary와 해석 규칙이 문서화된 실험 프로토콜
```

반대로 종합설계 1에서 완성했다고 쓰면 안 되는 것도 있다.

```text
1. 모든 선행연구의 exact reproduction
2. personalized delivery, hypernetwork, GNN server aggregation의 완전 구현
3. SOTA accuracy 달성
4. semantic gain의 순수 causal proof
```

특히 `Semantic Gain`과 `Smoothing Gain`은 발표용 직관으로는 쓸 수 있지만,
그 자체가 직접 관측되는 순수량이라고 쓰면 안 된다. 우리 프로젝트에서 안전한
표현은 다음과 같다.

```text
Semantic contribution은 real graph와 matched control 사이의 estimated
real-control gap으로 해석한다.

Smoothing contribution은 uniform, identity, matched random, shuffled,
graph-free control 등 relation-free 또는 relation-destroyed control의
행동을 통해 추정한다.
```

즉, "Semantic Graph - Random Graph = Pure Semantic Gain"이라고 단정하지 않는다.
대신 "real graph가 matched controls와 graph-free controls를 모두 넘어서고,
alignment/LOO/DI/N_eff 같은 mechanism metric에서도 일관된 변화가 보이면
graph-specific explanation의 강도가 높아진다"라고 쓴다.

CS양식1 선행기술 조사에서는 기술요소를 세 개로 잡는 것이 좋다.

| 기술요소 | 내용 |
|---|---|
| A. Client relation modeling | client update, weight, embedding, history 등으로 client 관계를 추정 |
| B. Graph-based collaboration | graph smoothing, learned graph, attentive aggregation, personalized aggregation |
| C. Attribution and diagnostics | matched control, graph-free control, DI/N_eff/alignment/LOO/smoothness 지표 |

이렇게 잡으면 선행연구와 우리 과제의 유사점과 차이점이 자연스럽게 정리된다.
유사점은 client relation과 graph collaboration을 다룬다는 점이다. 차이점은
우리가 final accuracy만 보지 않고, graph relation의 의미 효과와 단순 smoothing
또는 contribution correction 효과를 분리해 평가한다는 점이다.

CS양식5 결과보고서에서는 다음 문장을 중심 문장으로 삼는다.

```text
본 과제는 선행 Graph-FL/PFL 연구를 대체하는 새로운 단일 알고리즘을 제안하기보다,
해당 계열 연구들의 성능 향상을 공정하게 해석하기 위한 실험 기반 평가
프레임워크를 구현한다. 따라서 Capstone Design I의 결과물은 graph source,
graph mode, aggregation target, correction family, control graph, diagnostic
metric을 조립식으로 비교할 수 있는 prototype과 문서화된 실험 프로토콜이다.
```

조원이 새로운 아이디어를 가져오면, 그것이 우리 프로젝트 안에 들어오는 기준도
명확해야 한다. graph source, graph mode, aggregation target, correction
family, diagnostic metric 중 하나로 들어오면 현재 프로젝트에 흡수할 수 있다.
client-specific model delivery, hypernetwork, personalized local objective처럼
현재 runner/result schema를 크게 바꾸는 것은 interface-target 또는 future work로
둔다. 이렇게 해야 프로젝트가 커져도 방향이 흐려지지 않는다.

마지막으로, 모든 문서에서 결론은 같은 방향이어야 한다.

```text
우리 프로젝트의 차별성은 graph를 사용한다는 사실 자체가 아니라,
graph를 사용했을 때 얻은 성능 향상을 여러 control과 diagnostic metric으로
분해하여 해석 가능하게 만든다는 점이다.
```

이 문장이 흔들리지 않으면, 선행연구 조사, 요구분석, 상세설계, 결과보고서,
공학문제수준설명표가 모두 같은 이야기를 하게 된다.
