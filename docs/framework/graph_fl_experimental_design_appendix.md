# Graph-FL Experimental Design Appendix

Reference for [graph_fl_experimental_design.md](graph_fl_experimental_design.md).

## 1. Notation

| Symbol | Meaning |
|---|---|
| $N$ | client 수 |
| $i,j$ | client index |
| $t$ | communication round |
| $w^t$ | global model at round $t$ |
| $w_i^t$ | local model of client $i$ |
| $g_i^t$ | raw local update |
| $\tilde g_i^t$ | corrected or graph-filtered update |
| $G^t$ | update matrix |
| $p_i^t$ | FedAvg sample-size weight |
| $q_i^t$ | contribution share |
| $A^t$ | client graph adjacency |
| $L^t$ | graph Laplacian |
| $z_i^t$ | client representation for graph construction |
| $\epsilon$ | numerical stability constant |

Basic update:

$$
g_i^t = w_i^t - w^t
$$

FedAvg aggregation:

$$
g_{\text{avg}}^t
=
\sum_i p_i^t g_i^t
$$

## 2. Metric Reference

Metric order:

```text
2.1-2.3   Intervention / aggregate behavior
2.4-2.7   Client contribution confounders
2.8-2.10  Graph structure confounders
2.11-2.14 Graph mechanism descriptors
```

### Group A. Intervention / Aggregate Behavior

#### 2.1 Real-Control Gap

$$
\Delta_{\text{real-control}}
=
Score_{\text{real}} - Score_{\text{control}}
$$

| Field | Content |
|---|---|
| use | real graph와 control graph의 counterfactual 차이 |
| Score | `accuracy`, `alignment change`, `LOO change`, `DI change`, `N_eff change` |
| `Real > matched_random` | edge 수 이상의 relation 정보 가능성 |
| `Real > shuffled` | client identity assignment가 중요 |
| `Real > uniform` | generic smoothing 이상으로 topology가 중요 |
| `Real ≈ control` | graph-specific claim 약함 |

#### 2.2 Cosine Similarity / Alignment

$$
\cos(x,y)
=
\frac{x^\top y}{\|x\|_2\|y\|_2 + \epsilon}
$$

| Field | Content |
|---|---|
| use | update 방향 유사도 |
| pairwise similarity | $x=g_i^t$, $y=g_j^t$ |
| alignment | $x=g_i^t$, $y=$ aggregated update |
| high alignment | aggregate가 client update를 잘 대표 |
| caveat | high `DI`와 함께 오르면 dominant client 효과일 수 있음 |

#### 2.3 Leave-One-Out Influence

$$
LOO_i^t
=
1 - \cos(g_{\text{all}}^t, g_{-i}^t)
$$

| Field | Content |
|---|---|
| use | client 하나를 뺐을 때 aggregate direction 변화 |
| $g_{\text{all}}^t$ | all-client aggregate |
| $g_{-i}^t$ | client $i$ 제외 aggregate |
| high `LOO_i` | client $i$ 영향 큼 |
| lower LOO after correction | single-client sensitivity 완화 가능성 |
| caveat | useful minority signal 제거 여부는 accuracy/loss와 확인 |

### Group B. Client Contribution Confounders

#### 2.4 Update Norm

$$
\|g_i^t\|_2
$$

| Field | Content |
|---|---|
| use | client update 크기 |
| high value | client $i$가 model을 크게 움직임 |
| confounder | high-norm client 억제 효과 |
| check with | `LOO`, `alignment`, `Contribution Share` |

#### 2.5 Contribution Share

$$
q_i^t
=
\frac{
p_i^t \|g_i^t\|_2
}{
\sum_j p_j^t \|g_j^t\|_2 + \epsilon
}
$$

| Field | Content |
|---|---|
| use | client별 aggregation 영향력 근사 |
| numerator | sample-size weight × update norm |
| denominator | all-client weighted update norm sum |
| high value | client $i$ 영향력 큼 |
| confounder | topology가 아니라 reweighting 효과 |

#### 2.6 Dominance Index

$$
DI^t = \max_i q_i^t
$$

| Field | Content |
|---|---|
| use | 가장 큰 contribution share |
| high `DI` | 한 client가 aggregation 지배 |
| lower `DI` | dominance 완화 |
| confounder | graph gain이 dominance suppression일 수 있음 |
| caveat | `DI` 하락만으로 좋은 correction은 아님 |

#### 2.7 Effective Number of Clients

$$
N_{\text{eff}}^t
=
\frac{1}{\sum_i (q_i^t)^2 + \epsilon}
$$

| Field | Content |
|---|---|
| use | 실질적으로 기여한 client 수 |
| low value | 소수 client에 contribution 집중 |
| high value | 여러 client가 고르게 기여 |
| good sign | `N_eff` 증가 + alignment 증가 |
| warning sign | `N_eff` 증가 + accuracy 하락 |

### Group C. Graph Structure Confounders

#### 2.8 Graph Density

$$
\rho_A^t
=
\frac{
|\{(i,j): i<j, A_{ij}^t > 0\}|
}{
\binom{N}{2}
}
$$

| Field | Content |
|---|---|
| use | 가능한 edge 중 실제 edge 비율 |
| high density | 많은 client가 섞임 |
| confounder | edge 수 또는 smoothing 양 |
| control | `matched_random` |

#### 2.9 Degree Distribution

$$
d_i^t
=
\sum_j \mathbf{1}[A_{ij}^t > 0]
$$

| Field | Content |
|---|---|
| use | client별 연결 수 |
| high degree client | graph hub |
| confounder | hub 구조가 결과를 만들 수 있음 |
| check | degree와 $q_i$ 상관 |

#### 2.10 Graph Entropy

| Field | Content |
|---|---|
| use | edge weight 분포의 균등성 |
| high entropy | diffuse smoothing |
| low entropy | 일부 edge/client pair에 집중 |
| confounder | uniform mixing 또는 strong-edge selection |
| caveat | low entropy가 항상 나쁜 것은 아님 |

### Group D. Graph Mechanism Descriptors

#### 2.11 Homophily / Assortativity

$$
h_{\text{dist}}^t
=
\frac{
\sum_{i<j} A_{ij}^t \cos(r_i,r_j)
}{
\sum_{i<j} A_{ij}^t + \epsilon
}
$$

| Field | Content |
|---|---|
| use | 연결된 client들의 속성 유사도 |
| $r_i$ | client $i$의 label distribution |
| high value | 비슷한 client끼리 강하게 연결 |
| low but useful graph | label similarity 말고 optimization similarity 가능성 |

#### 2.12 Graph Smoothness

$$
H_G^t
=
\frac{
\operatorname{tr}((G^t)^\top L^t G^t)
}{
\|G^t\|_F^2 + \epsilon
}
$$

| Field | Content |
|---|---|
| use | graph 위 update signal smoothness |
| low value | 연결된 client update가 비슷함 |
| real < random | real graph가 update relation을 더 잘 반영할 가능성 |
| caveat | smoothness 개선 + accuracy 하락은 over-smoothing 가능성 |

#### 2.13 Spectral Energy / Frequency Band

$$
G^t
=
G_{\text{low}}^t
+
G_{\text{mid}}^t
+
G_{\text{high}}^t
$$

| Field | Content |
|---|---|
| use | update signal의 graph-frequency band 분해 |
| low band | graph 위 smooth shared component |
| mid band | intermediate component |
| high band | connected clients 사이 차이가 큰 component |

Band interventions:

```text
all-band
low-only
mid-only
high-only
remove-low
remove-mid
remove-high
```

| Field | Content |
|---|---|
| rule | spectral metric은 단독 증거가 아님 |

#### 2.14 Temporal Stability

$$
J_E^t
=
\frac{
|E^t \cap E^{t+1}|
}{
|E^t \cup E^{t+1}| + \epsilon
}
$$

| Field | Content |
|---|---|
| use | round 사이 graph edge 유지율 |
| high value | stable client relation |
| low value | round-level noise 가능성 |

## 3. Control Graphs

| Control | 확인하는 것 | 반증하려는 설명 |
|---|---|---|
| `matched_random` | edge 수/weight 분포 | topology가 아니라 mixing strength |
| `shuffled` | edge identity assignment | client identity가 중요하지 않음 |
| `uniform` | 평균 smoothing | topology 없이 smoothing만으로 충분 |
| `identity` | graph 개입 없음 | graph path가 의미 없음 |
| `clustering_only` | coarse grouping | fine-grained edge 불필요 |
| `graphfree_dominance_reweight` | dominance correction | graph가 아니라 contribution correction |

## 4. Optional Experiment Bank

| Experiment | Question | Method |
|---|---|---|
| Metric Predictiveness | 어떤 metric이 다음 round failure를 설명하는가? | Spearman, regression, AUC |
| Source Attribution | 어떤 representation이 graph를 의미 있게 만드는가? | `update`, `classifier_head_update`, `ema_update`, `weight` |
| Topology Attribution | 어떤 topology가 중요한가? | `knn`, `dense`, `rbf_knn`, `magnitude_knn`, `pfedgraph_qp` |
| Aggregation Target Attribution | graph가 어디에 작용해야 하는가? | update, EMA update, weight |
| Filter Strength Sweep | smoothing 강도는 얼마가 적절한가? | `0.0`, `0.25`, `0.5`, `1.0`, `2.0` |
| Frequency Band Importance | 어떤 band가 중요한가? | low/mid/high/remove variants |
| Harmful Client Analysis | LOO가 큰 client는 무엇으로 설명되는가? | sample size, norm, $q_i$, degree, neighbor similarity |

## 5. Report Templates

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

## 6. Sources

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
- [Modularity and community structure in networks](https://pubmed.ncbi.nlm.nih.gov/16723398/)
- [Personalized Federated Learning with Inferred Collaboration Graphs](https://proceedings.mlr.press/v202/ye23b.html)
- [Personalized Federated Learning With Graph](https://arxiv.org/abs/2203.00829)
- [Personalized Subgraph Federated Learning](https://arxiv.org/abs/2206.10206)
