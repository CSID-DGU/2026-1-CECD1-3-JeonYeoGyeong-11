# Graph-FL Experimental Design

목표: Graph-FL gain이 실제 graph structure 때문인지, 아니면 norm, dominance, smoothing, optimizer 같은 단순 요인 때문인지 분해한다.

세부 수식과 지표 해석은 [graph_fl_experimental_design_appendix.md](graph_fl_experimental_design_appendix.md)를 따른다.

## 1. Claim

핵심 질문:

```text
Does graph structure explain the observed gain beyond simpler confounders?
```

주장하지 않는 것:

```text
새 Graph-FL 알고리즘 제안
SOTA accuracy 주장
FedAvg보다 높으면 graph relation이 유효하다는 주장
```

주장하는 것:

```text
Graph-FL gain을 component, control, diagnostic metric으로 분해한다.
```

## 2. Positioning

| Work | 관련 아이디어 |
|---|---|
| FedAMP | similar client collaboration |
| SFL | client relation graph in PFL |
| pFedGraph | learned collaboration graph |
| FED-PUB / GPFL | functional embedding similarity |

## 3. Decomposition Axes

```text
graph_source
-> graph_mode
-> aggregation_target
-> correction_family
-> diagnostics
```

| Component | 질문 | 예시 |
|---|---|---|
| `graph_source` | client를 무엇으로 표현하는가? | `update`, `classifier_head_update`, `ema_update`, `weight` |
| `graph_mode` | relation을 어떤 graph로 만드는가? | `knn`, `rbf_knn`, `learned_smooth`, `pfedgraph_qp` |
| `aggregation_target` | graph를 어떤 signal에 적용하는가? | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| `correction_family` | graph, control, graph-free 중 무엇인가? | `real_graph`, `control_graph`, `graph_free` |
| `diagnostics` | 실제로 무엇이 바뀌었는가? | `DI`, `N_eff`, `alignment`, `LOO`, graph metrics |

## 4. Metric Map

### Group A. Intervention / Aggregate Behavior

| Ref | Metric | 질문 |
|---|---|---|
| `2.1` | `Real-Control Gap` | real graph가 control보다 다른가? |
| `2.2` | `Cosine Similarity / Alignment` | aggregate direction이 client update를 더 잘 대표하는가? |
| `2.3` | `Leave-One-Out Influence` | 특정 client가 round direction을 과하게 흔드는가? |

### Group B. Client Contribution Confounders

| Ref | Metric | 배제할 설명 |
|---|---|---|
| `2.4` | `Update Norm` | high-norm client 억제 |
| `2.5` | `Contribution Share` | contribution reweighting |
| `2.6` | `Dominance Index` | dominant client 완화 |
| `2.7` | `Effective Number of Clients` | 참여 균형 조정 |

### Group C. Graph Structure Confounders

| Ref | Metric | 배제할 설명 |
|---|---|---|
| `2.8` | `Graph Density` | edge 수 또는 smoothing 양 |
| `2.9` | `Degree Distribution` | hub 구조 |
| `2.10` | `Graph Entropy` | diffuse smoothing 또는 edge 집중 |

### Group D. Graph Mechanism Descriptors

| Ref | Metric | 설명 후보 |
|---|---|---|
| `2.11` | `Homophily / Assortativity` | 어떤 client 속성을 반영하는가? |
| `2.12` | `Graph Smoothness` | graph와 update signal이 맞는가? |
| `2.13` | `Spectral Energy / Frequency Band` | 어떤 graph-frequency band가 중요한가? |
| `2.14` | `Temporal Stability` | stable relation인가, round noise인가? |

## 5. Core Experiments

### Preflight. Non-IID Stress Calibration

| 항목 | 설정 |
|---|---|
| 목적 | non-IID stress calibration |
| `alpha` | `0.03`, `0.1`, `0.3`, `1.0` |
| `client_count` | `20`, `50` |
| `seed` | `3-5` values |
| baseline | `FedAvg`, `FedAvgM/FedOpt` |
| 관찰 | accuracy stability, `Update Norm`, `DI`, `N_eff`, `Alignment`, `LOO` |
| claim | 없음 |

### Core 1. Real Graph Vs Controls

| Variant | 확인하는 것 |
|---|---|
| `real_graph` | 실제 graph effect |
| `matched_random` | edge 수/weight 분포 효과 |
| `shuffled` | client identity assignment 효과 |
| `uniform` | generic smoothing 효과 |
| `identity` | graph intervention 없음 |
| `clustering_only` | coarse group 효과 |
| `graphfree_dominance_reweight` | dominance correction 효과 |

Semantic/smoothing reading:

```text
Semantic Gain is reported as an estimated real-control gap.
Smoothing Gain is inferred from relation-free or relation-destroyed controls.
Random control is a matched counterfactual, not pure smoothing by assumption.
```

Use `matched_random` for edge-count or weight-distribution effects, `shuffled`
for client identity assignment, `uniform` for relation-free averaging,
`identity` for no graph intervention, and `graphfree_dominance_reweight` for
non-graph contribution correction. A graph-specific claim needs the real graph
to survive this set, not only to beat FedAvg.

### Core 2. Component Attribution

원칙:

```text
한 번에 하나만 바꾼다.
seed, client split, optimizer, control setting은 고정한다.
```

| Axis | 최소 비교 | 해석 |
|---|---|---|
| `graph_source` | `update` vs `classifier_head_update` | representation effect |
| `graph_mode` | `default_similarity_knn` vs `matched_random/shuffled` | topology/sparsification effect |
| `aggregation_target` | `graph_filtered_update` vs `graph_filtered_ema_update` | target signal effect |

### Core 3. Diagnostic Mechanism Chain

좋은 chain:

```text
graph_source/mode 선택
-> homophily or smoothness 변화
-> alignment 증가
-> LOO 감소
-> accuracy stability 개선
```

약한 chain:

```text
accuracy만 증가
diagnostic 변화 없음
```

## 6. Interpretation Rules

| Level | 조건 | 결론 |
|---|---|---|
| Strong graph-specific effect | real graph가 counterfactual graph와 graph-free controls를 모두 이기고, `Alignment`, `LOO`, `DI`, `N_eff` 중 최소 2개 개선 | graph topology가 low-order statistic 이상의 정보를 제공했을 가능성 |
| Partially graph-related effect | 일부 control보다 좋지만 graph-free 또는 clustering-only로 상당 부분 설명 | relation은 의미 있으나 fine-grained topology 필요성은 약함 |
| No necessary graph effect | random/shuffled/uniform/graph-free와 유사하고 primary metrics 개선 없음 | explicit graph structure 없이 단순 correction으로 충분 |

Guardrails:

```text
accuracy만으로 graph-specific claim을 하지 않는다.
control graph와 graph-free correction 없이는 graph claim을 하지 않는다.
spectral metric은 단독 증거가 아니다.
graph-specific effect가 약해도 실패가 아니다.
```

## 7. Report Tables

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
temporal_stability
```

Mechanism:

```text
variant
low_frequency_energy_ratio
mid_frequency_energy_ratio
high_frequency_energy_ratio
high_to_low_ratio
suppressed_energy_ratio
band_importance
```

## 8. Sources

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)
- [Personalized Cross-Silo Federated Learning on Non-IID Data](https://arxiv.org/abs/2007.03797)
- [Graph Structured Data Viewed Through a Fourier Lens](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-209.html)
- [Deeper Insights into Graph Convolutional Networks](https://arxiv.org/abs/1801.07606)
- [Revisiting Graph Neural Networks: All We Have is Low-Pass Filters](https://arxiv.org/abs/1905.09550)
- [Beyond Homophily in Graph Neural Networks](https://arxiv.org/abs/2006.11468)
- [Mixing patterns in networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.67.026126)
- [Personalized Federated Learning with Inferred Collaboration Graphs](https://proceedings.mlr.press/v202/ye23b.html)
- [Personalized Federated Learning With Graph](https://arxiv.org/abs/2203.00829)
