# Graph-FL Experimental Design

이 문서는 현재 프로젝트의 canonical 실험 설계 요약이다. 새 작업자는 이 문서를
먼저 읽고, 세부 구현은 `interfaces.md`, metric 해석은 `diagnostics.md`로
넘어간다.

## One-Page Roadmap

```text
1. 최종 claim을 먼저 확인한다.
2. primary metric으로 claim이 판정 가능한지 본다.
3. minimal core experiment만 먼저 수행한다.
4. 결과가 애매할 때 secondary / exploratory analysis를 추가한다.
5. 마지막에 result interpretation rule로 claim boundary를 정한다.
```

최종 claim:

```text
본 프로젝트는 새로운 Graph-FL 알고리즘을 제안하는 것이 아니라,
Graph-FL에서 관측되는 성능 변화가 실제 graph structure에서 오는지,
아니면 dominance, norm, smoothing, optimizer 같은 더 단순한 요인으로
설명되는지를 분해해 검증하는 실험 프레임워크를 제안한다.
```

핵심 판정 질문:

```text
Does graph structure explain the observed gain beyond simpler confounders?
```

## Primary Metrics

claim 판정은 먼저 아래 지표로 한다.

| Metric | 질문 |
|---|---|
| `real-control gap` | real graph가 matched random, shuffled, uniform, identity, clustering-only보다 다른가 |
| `graph-free control gap` | graph-free norm/dominance correction만으로 같은 gain이 재현되는가 |
| `alignment change` | correction 후 client update와 aggregate direction이 더 일관되는가 |
| `LOO change` | single-client sensitivity가 줄었는가 |
| `DI / N_eff change` | dominance가 줄고 실질 참여 client 수가 늘었는가 |

secondary metric은 구조 설명을 위해 붙인다.

```text
density
degree
entropy
homophily
smoothness
```

exploratory metric은 mechanism candidate로만 쓴다.

```text
spectral energy
eigengap
temporal stability
```

## Minimal Core Experiment Set

### Preflight. Non-IID Stress Calibration

실험 환경이 충분히 어렵고 diagnostic metric이 stress에 반응하는지 확인한다.
이 단계만으로 graph claim을 만들지는 않는다.

필수 비교:

```text
FedAvg
FedAvgM / FedOpt family
different alpha or client-count stress
```

### Core 1. Real Graph Vs Counterfactual + Graph-Free Controls

graph topology가 단순 smoothing, edge 수, identity-free relation, dominance
correction 효과가 아닌지 확인한다.

필수 비교:

```text
real
matched_random
shuffled
uniform
identity
clustering_only
graphfree_dominance_reweight
```

이 실험이 없으면 "graph를 썼다" 이상의 주장을 하기 어렵다.

### Core 2. Component Attribution With Minimal Knobs

모든 source/mode/target을 다 돌리는 full search가 아니라, 최소 knob로 어디서
차이가 나는지 본다.

시작점:

```text
source 2개
topology 2개
aggregation_target 2개
same seed/client split
same optimizer/control setting
```

목표:

```text
source / topology / target 중 어느 component가 diagnostic behavior를 바꾸는가?
```

### Core 3. Diagnostic Mechanism Chain

accuracy 변화가 primary metric 변화와 연결되는지 확인한다.

예:

```text
classifier_head_update graph
-> label-distribution homophily 증가
-> real graph smoothness가 random보다 낮음
-> alignment 증가
-> LOO 감소
-> accuracy 안정화
```

이런 chain이 있어야 결과가 leaderboard가 아니라 explanation이 된다.

## Experiment Bank

core 결과가 나온 뒤 필요할 때만 아래 실험을 추가한다.

| Experiment | 사용 시점 |
|---|---|
| Filter strength sweep | graph filtering mechanism을 더 설명해야 할 때 |
| Frequency band importance | real-control gap이 graph-frequency band 중 어디서 오는지 봐야 할 때 |
| Harmful client detection | LOO 결과를 client-level pattern으로 해석해야 할 때 |
| Temporal stability | graph construction noise나 round-level instability가 의심될 때 |
| Full source/mode/target sweep | 최소 attribution 결과가 유망할 때 |

## Result Interpretation Rules

먼저 결론을 세 단계 중 하나로 정한다.

| Level | 판정 규칙 |
|---|---|
| Strong graph-specific effect | real graph가 counterfactual graph와 graph-free controls를 모두 이기고, alignment / LOO / DI / N_eff 중 최소 두 개가 함께 개선된다 |
| Partially graph-related effect | real graph가 일부 control보다 좋지만, graph-free control이나 source/topology confounder로 상당 부분 설명된다 |
| No necessary graph effect | real graph가 random/shuffled/uniform/graph-free control과 비슷하다 |

case별 report wording:

```text
Case A. Graph-specific claim이 강한 경우
  Graph topology provides additional diagnostic information beyond low-order update statistics.

Case B. Pairwise relation까지만 의미 있는 경우
  Client relation matters, but explicit topology assignment is not yet proven necessary.

Case C. Graph-free statistic으로 설명되는 경우
  In this setting, graph gain is largely explained by dominance or magnitude correction.

Case D. Graph-specific effect가 약한 경우
  In this setting, explicit graph structure is not necessary.
  Simpler controls or non-graph statistics explain most of the observed behavior.
```

## Recommended Report Tables

Primary table:

```text
variant
graph_source
graph_mode
aggregation_target
correction_family
control_graph_mode
accuracy_final
accuracy_best
DI_pre / DI_post
N_eff_pre / N_eff_post
alignment_pre / alignment_post
LOO_pre / LOO_post
real_control_gap
graphfree_control_gap
conclusion_level
```

Secondary structure table:

```text
variant
graph_density
graph_entropy
degree_mean / degree_max
homophily
assortativity
smoothness
```

Exploratory mechanism table:

```text
variant
low_frequency_energy_ratio
high_frequency_energy_ratio
high_to_low_energy_ratio
suppressed_energy_ratio
temporal_stability
```

## Guardrails

- accuracy만으로 graph-specific effect를 주장하지 않는다.
- spectral metric은 mechanism candidate이지 단독 증거가 아니다.
- control graph와 graph-free correction이 없으면 graph claim을 세우지 않는다.
- graph-specific effect가 약해도 실패로 쓰지 않는다. 더 단순한 설계로 충분하다는
  결론도 이 framework의 결과다.
