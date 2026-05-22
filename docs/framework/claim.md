# Project Claim And Boundary

## Claim

Purpose: report claim boundary and forbidden wording.

```text
Graph-FL gain이 실제 graph structure 때문인지,
아니면 dominance, norm, smoothing, optimizer 같은 단순 요인 때문인지
component/control/diagnostic metric으로 분해한다.
```

Core question:

```text
Does graph structure explain the observed gain beyond simpler confounders?
```

## Scope

| In scope | Out of scope |
|---|---|
| component attribution | new Graph-FL algorithm claim |
| diagnostic explanation | SOTA accuracy claim |
| graph vs graph-free boundary | “FedAvg보다 높음 = graph relation 유효” |
| control-based conclusion | spectral/low-pass robustness 단독 주장 |

## Decomposition

Accuracy gain은 아래 항목의 합성 결과로 본다. 실험은 이 항목들을 하나씩 분리한다.

```text
Graph-FL gain
= client state effect
+ relation effect
+ topology effect
+ graph filtering effect
+ optimizer effect
+ low-order statistic effect
```

## Semantic / Smoothing Vocabulary

`Semantic Gain` and `Smoothing Gain` may be used as presentation shorthand, but
they are not separate directly observed quantities in this project.

Project-safe translation:

```text
Estimated semantic contribution
= Score(real_graph) - Score(matched control)

Generic smoothing contribution
= relation-free or relation-destroyed control behavior
```

Interpretation rules:

| Term | Project meaning |
|---|---|
| `Semantic Graph` | `real_graph`: graph built from a chosen client representation and topology |
| `Semantic Gain` | estimated `real-control gap`, not a pure causal residue |
| `Smoothing Gain` | effect suggested by `uniform`, `matched_random`, `identity`, and smoothness diagnostics |
| `Random Graph` | matched counterfactual, not assumed to be pure smoothing |

Use this framing to explain intuition. Use the control matrix and diagnostics for
claim strength.

## Evidence Required

claim은 `Separation -> Control -> Mechanism -> Boundary` 순서로 세운다.

| Requirement | Required evidence |
|---|---|
| Separation | `graph_source`, `graph_mode`, `aggregation_target` 변경 시 diagnostic behavior 분리 |
| Control | random, shuffled, uniform, identity, clustering-only, graph-free dominance correction 비교 |
| Mechanism | accuracy change와 `alignment`, `LOO`, `DI`, `N_eff`, `smoothness`, spectral metric 연결 |
| Boundary | graph-specific effect가 약할 때 대체 설명 제시 |

## Primary Metrics

Primary metric은 claim 판정에 직접 쓰고, secondary/exploratory metric은 mechanism 보조 증거로만 쓴다.

```text
real-control gap
graph-free control gap
alignment change
LOO change
DI / N_eff change
```

Secondary:

```text
density
degree
entropy
homophily
smoothness
```

Exploratory:

```text
spectral energy
eigengap
temporal stability
```

## Minimal Experiments

아래 네 단계가 최소 claim path다. Optional sweep은 이 결과가 애매할 때만 추가한다.

```text
Preflight. Non-IID stress calibration
Core 1. Real graph vs counterfactual + graph-free controls
Core 2. Component attribution with minimal knobs
Core 3. Diagnostic mechanism chain
```

## Interpretation Levels

결론은 성공/실패가 아니라 claim strength로 분류한다.

| Level | Condition |
|---|---|
| Strong graph-specific effect | real graph가 counterfactual graph와 graph-free controls를 모두 이기고, `alignment` / `LOO` / `DI` / `N_eff` 중 최소 2개 개선 |
| Partially graph-related effect | 일부 control보다 좋지만 graph-free control 또는 source/topology/smoothing confounder로 상당 부분 설명 |
| No necessary graph effect | random/shuffled/uniform/graph-free control과 유사 |

## Report Language

Allowed:

```text
real graph는 matched random과 graph-free dominance control을 모두 넘었고,
alignment gain과 LOO drop이 함께 관측되었다.
```

```text
real graph와 graph-free dominance reweighting이 유사한 성능과 DI/N_eff 변화를 보였다.
```

Avoid:

```text
real graph가 FedAvg보다 좋으므로 graph relation이 유효하다.
DI가 낮아졌으므로 성능 향상의 원인은 dominance suppression이다.
```

## Read Next

- [graph_fl_experimental_design.md](graph_fl_experimental_design.md): core experiment and interpretation rules
- [graph_fl_experimental_design_appendix.md](graph_fl_experimental_design_appendix.md): metric definitions
- [interfaces.md](interfaces.md): extension interfaces
- [prior-work-mapping.md](prior-work-mapping.md): prior-work support boundary
