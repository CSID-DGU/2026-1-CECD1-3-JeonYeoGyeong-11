# Graph-FL Experimental Design

Canonical design: [graph_fl_experimental_design.md](graph_fl_experimental_design.md). Metric reference: [graph_fl_experimental_design_appendix.md](graph_fl_experimental_design_appendix.md).

## Roadmap

Use this order when planning or reviewing a run.

```text
1. Check claim.
2. Run minimal core experiments.
3. Compare real graph against counterfactual and graph-free controls.
4. Attribute source/mode/target effects.
5. Assign interpretation level.
```

## Primary Metrics

These metrics decide whether a result can support a graph-specific claim.

| Metric | Question |
|---|---|
| `real-control gap` | real graph가 matched random, shuffled, uniform, identity, clustering-only와 다른가 |
| `graph-free control gap` | graph-free norm/dominance correction이 같은 gain을 재현하는가 |
| `alignment change` | aggregate direction이 client update와 더 일관되는가 |
| `LOO change` | single-client sensitivity가 줄었는가 |
| `DI / N_eff change` | dominance가 줄고 실질 참여 client 수가 늘었는가 |

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

## Core Experiments

Core experiments are ordered by dependency: stress check, graph/control separation, component attribution, then mechanism chain.

### Preflight

| Item | Values |
|---|---|
| goal | non-IID stress calibration |
| baseline | `FedAvg`, `FedAvgM/FedOpt` |
| stress | alpha/client-count sweep |
| claim | none |

### Core 1. Real Graph Vs Controls

First claim-bearing experiment.

```text
real
matched_random
shuffled
uniform
identity
clustering_only
graphfree_dominance_reweight
```

### Core 2. Component Attribution

Run after Core 1 shows a real-control gap worth explaining.

```text
source 2개
topology 2개
aggregation_target 2개
same seed/client split
same optimizer/control setting
```

### Core 3. Diagnostic Chain

Use this to connect outcome changes to metric changes.

Example:

```text
classifier_head_update graph
-> label-distribution homophily 증가
-> real graph smoothness가 random보다 낮음
-> alignment 증가
-> LOO 감소
-> accuracy 안정화
```

## Experiment Bank

Optional experiments refine an already observed pattern.

| Experiment | Use |
|---|---|
| Filter strength sweep | graph filtering strength |
| Frequency band importance | low/mid/high band attribution |
| Harmful client detection | LOO client-level pattern |
| Temporal stability | graph construction noise |
| Full source/mode/target sweep | promising minimal attribution result |

## Interpretation Rules

Assign one level before writing report text.

| Level | Rule |
|---|---|
| Strong graph-specific effect | real graph > counterfactual graph and graph-free controls; at least 2 of `alignment`, `LOO`, `DI`, `N_eff` improve |
| Partially graph-related effect | real graph > some controls but graph-free/source/topology confounder explains much |
| No necessary graph effect | real graph ≈ random/shuffled/uniform/graph-free |

## Report Tables

Primary:

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

Structure:

```text
variant
graph_density
graph_entropy
degree_mean / degree_max
homophily
assortativity
smoothness
```

Mechanism:

```text
variant
low_frequency_energy_ratio
high_frequency_energy_ratio
high_to_low_energy_ratio
suppressed_energy_ratio
temporal_stability
```

## Guardrails

```text
No graph-specific claim from accuracy alone.
No graph claim without control graph and graph-free correction.
Spectral metrics are mechanism candidates, not standalone evidence.
Weak graph-specific effect is still a valid boundary result.
```
