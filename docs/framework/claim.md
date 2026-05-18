# Project Claim And Boundary

이 저장소의 현재 주장은 명확하다.

```text
본 프로젝트는 새로운 Graph-FL 알고리즘 하나를 제안하는 것이 아니라,
Graph-FL에서 관측되는 성능 변화가 실제 graph structure에서 오는지,
아니면 dominance, norm, smoothing, optimizer 같은 더 단순한 요인으로
설명되는지를 분해해 검증하는 실험 프레임워크를 제안한다.
```

핵심 판정 질문은 다음이다.

```text
Does graph structure explain the observed gain beyond simpler confounders?
```

## What This Project Is

이 프로젝트는 graph-based federated learning method를 하나의 덩어리로 보지
않는다. 대신 FL round 안에서 다음 요소를 분리해 본다.

```text
Graph-FL gain
= client state effect
+ relation effect
+ topology effect
+ graph filtering effect
+ optimizer effect
+ low-order statistic effect
```

따라서 이 저장소의 가치는 최고 accuracy를 하나 더 만드는 데 있지 않다.

```text
Project value
= component attribution
+ diagnostic explanation
+ honest control boundary
```

## What This Project Is Not

다음은 현재 프로젝트의 주장이 아니다.

```text
우리가 만든 graph algorithm이 항상 FedAvg보다 좋다.
spectral/low-pass filtering alone이 robust하다.
Graph-FL 성능 향상은 곧 graph relation이 유효하다는 뜻이다.
SOTA 성능을 주장한다.
```

이런 문장은 control과 diagnostic chain 없이 쓰면 안 된다.

## Evidence Required

실험이 설득력을 가지려면 네 가지가 필요하다.

| 요구사항 | 봐야 하는 것 |
|---|---|
| Separation | `graph_source`, `graph_mode`, `aggregation_target`을 바꿨을 때 diagnostic behavior가 구분되는가 |
| Control | real graph 효과가 random, shuffled, uniform, identity, clustering-only, graph-free dominance correction으로 설명되는가 |
| Mechanism | accuracy 변화가 alignment, LOO, DI, N_eff, smoothness, spectral energy 중 하나 이상의 chain과 연결되는가 |
| Claim boundary | graph-specific effect가 약할 때도 어떤 더 단순한 설명이 충분한지 말할 수 있는가 |

## Primary Metrics

claim 판정에 먼저 쓰는 지표는 다음이다.

```text
real-control gap
graph-free control gap
alignment change
LOO change
DI / N_eff change
```

secondary/exploratory metric은 primary conclusion을 보조할 때만 쓴다.

```text
Secondary:
  density, degree, entropy, homophily, smoothness

Exploratory:
  spectral energy, eigengap, temporal stability
```

## Minimal Core Experiments

시간이 부족하면 아래만 수행한다. 이것은 가능한 실험 목록이 아니라 claim을
지탱하는 최소 실험이다.

```text
Preflight. Non-IID stress calibration
Core 1. Real graph vs counterfactual + graph-free controls
Core 2. Component attribution with minimal knobs
Core 3. Diagnostic mechanism chain
```

각 실험의 목적은 "우리 방법이 제일 좋다"가 아니라 다음을 보이는 것이다.

```text
이 framework를 쓰면 Graph-FL의 효과와 한계를 더 정확히 말할 수 있다.
```

## Interpretation Levels

결과는 먼저 아래 셋 중 하나로 정리한다.

| 결론 | 조건 |
|---|---|
| Strong graph-specific effect | real graph가 counterfactual graph와 graph-free controls를 모두 이기고, alignment / LOO / DI / N_eff 중 최소 두 개가 함께 개선된다 |
| Partially graph-related effect | real graph가 일부 control보다 좋지만, graph-free control이나 source/topology/smoothing confounder로 상당 부분 설명된다 |
| No necessary graph effect | real graph가 random/shuffled/uniform/graph-free control과 비슷하고, 더 단순한 설계로 충분하다는 결론이 남는다 |

이 세 단계는 성공/실패 판정이 아니다. graph-specific effect가 약하더라도,
그 사실 자체가 framework의 유용한 결론이 될 수 있다.

## Safe Report Language

좋은 문장:

```text
real graph는 matched random과 graph-free dominance control을 모두 넘었고,
alignment gain과 LOO drop이 함께 관측되었다. 이 setting에서는 graph topology가
low-order statistic 이상의 정보를 제공했을 가능성이 있다.
```

좋은 문장:

```text
real graph와 graph-free dominance reweighting이 유사한 성능과 DI/N_eff 변화를
보였다. 이 setting의 gain은 graph relation보다 dominance correction으로
상당 부분 설명된다.
```

피해야 할 문장:

```text
real graph가 FedAvg보다 좋으므로 graph relation이 유효하다.
```

피해야 할 문장:

```text
DI가 낮아졌으므로 성능 향상의 원인은 dominance suppression이다.
```

## Read Next

- [experimental-design.md](experimental-design.md): core experiment와 report table 기준
- [diagnostics.md](diagnostics.md): metric별 해석 규칙
- [interfaces.md](interfaces.md): 새 graph method를 어떤 interface로 붙이는지
- [prior-work-mapping.md](prior-work-mapping.md): 선행연구 exact/proxy/interface 경계
