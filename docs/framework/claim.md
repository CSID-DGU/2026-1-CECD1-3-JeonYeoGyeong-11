# Non-IID 연합학습 성능 향상을 위한 Graph 기반 집계 효과 분해 및 진단 프레임워크

초기에는 선행연구 조사가 충분하지 않은 상태에서 client graph + spectral smoothing 방향의 novelty를 크게 봤습니다.

최근 관련 연구를 다시 확인해보니, 이 방향은 이미 많이 연구되어 있다는 점을 확인했습니다. 이 부분은 제가 처음에 조사가 부족했던 것 같습니다. 죄송합니다. 그래서 주제의 방향을 수정할 필요가 있을 것 같습니다.

다만 지금까지 진행한 구현과 실험을 바로 버려야 하는 상황은 아니라고 생각합니다. 현재 구현된 코드와 실험들 중에서 활용할 수 있는 부분을 바탕으로, 기존 방향과 겹침을 줄이면서도 이어갈 수 있는 연구 질문을 다시 생각해봤습니다.

아래는 제가 현재 코드와 실험 흐름을 바탕으로 다시 생각해본 주제입니다. 아직 확정하자는 뜻은 아니고, 방향이 괜찮아 보이면 이쪽으로 조금 더 정리해보고, 마음에 들지 않으면 다른 방향도 다시 생각해볼 수 있을 것 같습니다.

## 1. 연구 동기

처음에는 기존 연구 흐름처럼 client relation graph를 구성하고, 이를 federated aggregation correction에 활용하는 방향을 시도했다.

그러나 초기 실험을 진행하면서 중요한 의문이 생겼다.

> graph-based correction이 성능을 올렸다면, 그 이득은 정말 graph가 client relation을 잘 포착했기 때문인가?

최근 graph-based FL 연구들은 client 간 similarity, relation graph, personalized graph, attention graph 등을 활용하여 FedAvg 계열 방법 대비 성능 향상을 보인다. 그러나 성능 향상만으로는 그 이득이 실제 graph-specific relation에서 온 것인지, 또는 다른 aggregation-level effect에서 온 것인지 구분하기 어렵다.

성능 향상은 다음과 같은 다른 mechanism으로도 설명될 수 있다.

- graph correction이 세밀한 edge-level relation이 아니라, 비슷한 client를 묶는 coarse clustering 효과로 작동했기 때문일 수 있다.
- graph correction이 dominant update의 영향력을 줄였기 때문일 수 있다.
- graph 구조와 무관하게 update를 smoothing/mixing한 regularization 효과일 수 있다.
- graph topology 자체보다 update norm control 또는 contribution balancing이 핵심일 수 있다.

따라서 본 연구는 새로운 graph aggregation 알고리즘 하나를 제안하는 것보다, **graph-based aggregation gain의 원인을 분해하고 진단하는 프레임워크**를 설계하는 데 초점을 둔다.

---

## 2. 핵심 관찰: graph correction의 구조적 효과

graph-based FL 방법은 보통 client model 또는 update를 graph weight, collaboration weight, attention weight를 통해 섞는 구조를 갖는다. 따라서 graph correction은 구조적으로 **smoothing/mixing 효과**를 포함한다.

하지만 smoothing/mixing이 곧 fine-grained graph relation의 효과를 의미하지는 않는다. 같은 성능 향상은 coarse clustering, generic smoothing, 또는 aggregation influence 재분배 과정에서 나타나는 dominance suppression으로도 발생할 수 있다. 또한 dominance suppression은 graph correction의 필연적 결과가 아니라, graph topology와 weight normalization에 따라 dominant update가 줄어들 수도 있고 오히려 확산될 수도 있다.

기존 graph/similarity 기반 FL 연구들은 similarity graph, collaboration graph, attention graph 등을 통해 client relation을 모델링하며 성능 향상을 보인다. 그러나 이러한 성능 향상이 실제 fine-grained relation 때문인지, 더 단순한 clustering/smoothing/dominance control 때문인지는 별도의 control 없이는 명확히 구분하기 어렵다.

---

## 3. 핵심 질문

graph-based aggregation correction이 성능을 올렸을 때, 그 이유는 무엇인가?

1. **Fine-grained graph relation**  
   세밀한 edge weight와 topology가 aggregation correction에 유효했는가?

2. **Coarse clustering**  
   graph의 이득이 실제로는 비슷한 client를 묶는 grouping 효과로 설명되는가?

3. **Dominance suppression**  
   dominant client update의 영향력을 줄였기 때문에 성능이 향상되었는가?

4. **Generic smoothing**  
   graph 정보와 무관하게 smoothing/mixing 자체가 regularization처럼 작동했는가?

기존 graph-based FL 연구들은 주로 성능 향상을 중심으로 graph relation의 유용성을 보이지만, 그 이득이 **fine-grained graph relation, coarse clustering, dominance suppression, generic smoothing** 중 어디서 오는지를 명시적으로 분리해 검증하는 비교는 충분하지 않다.

---

## 4. 연구 목표

본 연구의 목표는 Non-IID 연합학습에서 FL 성능 향상을 위해 사용되는 graph-based aggregation correction의 작동 원인을 분석하고, 더 안정적인 aggregation 전략을 찾는 것이다.

구체적으로는 다음을 목표로 한다.

1. Non-IID FL에서 baseline aggregation의 성능과 update dominance를 진단한다.
2. real graph correction과 control graph correction을 비교하여 graph relation의 실제 효과를 확인한다.
3. cluster-only 또는 block-uniform graph control을 통해 fine-grained graph relation과 coarse clustering 효과를 분리한다.
4. graph-free correction과 비교하여 dominance suppression 효과를 분리한다.
5. 성능 향상이 graph relation, clustering, dominance suppression, generic smoothing 중 어디서 오는지 해석할 수 있는 진단 프레임워크를 구현한다.

---

## 5. 실험 및 프레임워크 설계

| 그룹 | 방법 | 목적 |
|---|---|---|
| Standard Baseline | FedAvg, FedAvgM | 기본 FL aggregation 성능 확인 |
| Existing Graph/Similarity FL | pFedGraph, FedAMP 등 | 기존 graph/similarity 기반 방법과 비교 |
| Real Graph Correction | update graph, signed conflict graph | 실제 client relation 기반 correction 확인 |
| Control Graph | random, shuffled, uniform, identity | graph relation이 깨져도 효과가 남는지 확인 |
| Clustering Control | cluster-only, block-uniform graph | 세밀한 graph relation 없이 grouping만으로 효과가 나는지 확인 |
| Graph-free Correction | norm clipping, contribution cap, dominance reweighting | dominance suppression 또는 magnitude control 효과 확인 |

### 비교 대상 재현 범위

pFedGraph, FedAMP 등 기존 graph/similarity FL 방법은 공개 코드 또는 논문 설정을 기준으로 가능한 범위에서 재현한다.  
다만 본 연구의 주된 목적은 각 방법의 최고 성능을 새로 튜닝하는 것이 아니라, **동일한 실험 환경에서 graph-specific effect와 generic correction effect를 분리해 비교하는 것**이다.

### 핵심 비교

| 결과 | 해석 |
|---|---|
| real graph > clustering control > random/uniform | fine-grained graph relation과 coarse clustering이 모두 일부 기여할 가능성 |
| real graph ≈ clustering control > random/uniform | 세밀한 edge relation보다 coarse clustering 효과가 핵심일 가능성 |
| real graph > control graph | graph relation 정보가 실제로 유효할 가능성 |
| real graph ≈ control graph | graph-specific relation보다 generic smoothing 가능성 |
| graph-free ≈ graph correction | 성능 향상이 dominance suppression으로 설명될 가능성 |
| graph-free > graph correction | 단순 dominance control이 graph correction보다 안정적일 가능성 |
| 모든 correction ≈ baseline | 해당 setting에서는 correction 효과가 제한적일 가능성 |

### Robustness 확인 범위

| 변수 | 범위 |
|---|---|
| Non-IID 정도 | $\alpha \in \{0.03, 0.1\}$ |
| Client 수 | $N \in \{5, 20\}$ |
| 데이터셋 | FashionMNIST + 1개 이상 추가 |
| Seed | 5개 이상 |

---

## 6. 진단 지표

각 round에서 client별 update와 global aggregation의 관계를 기록한다.

| 지표 | 정의 | 의미 |
|---|---|---|
| Contribution magnitude | $q_i = \frac{p_i \|g_i\|}{\sum_j p_j \|g_j\| + \epsilon}$ | client update가 전체 update norm에서 차지하는 상대적 영향 |
| Leave-one-out distortion | $d_i = 1 - \cos(\Delta, \Delta_{-i})$ | client $i$를 제거했을 때 global update 방향이 얼마나 바뀌는지 |
| Alignment | $a_i = \cos(g_i, \Delta)$ | client update와 global update의 방향 일치도 |
| Dominance Index | $DI = \max_i q_i$ | 특정 client update에 영향력이 집중되는 정도 |
| Effective Client Number | $N_{eff} = \frac{1}{\sum_i q_i^2 + \epsilon}$ | aggregation에 실질적으로 기여하는 client 수 |

여기서

$$
\Delta = \sum_i p_i g_i
$$

이고,

$$
\Delta_{-i} = \sum_{j \neq i} \tilde p_j g_j
$$

이다.

graph correction 전후로 $DI$, $N_{eff}$, $d_i$가 어떻게 변하는지를 보면, 해당 correction이 graph relation을 활용했는지, dominant update를 억제했는지, 또는 단순 smoothing으로 작동했는지를 해석할 수 있다.

또한 real graph와 cluster-only 또는 block-uniform graph를 비교하면, 성능 향상이 세밀한 edge-level relation 때문인지 단순 client grouping 효과 때문인지도 함께 확인할 수 있다.

단, 이 지표들은 단독으로 인과를 증명하지 않는다. 본 연구에서 지표는 **성능 변화와 control variant를 함께 읽을 때** mechanism evidence로 사용한다. 예를 들어 $DI$ 감소와 $N_{eff}$ 증가는 dominance suppression의 근거가 될 수 있지만, accuracy가 오르지 않으면 useful signal까지 눌린 over-smoothing일 수 있다. real graph가 shuffled/random control보다 높고, graph-free correction이 같은 개선을 재현하지 못할 때에만 fine-grained graph relation 가능성이 강해진다. 지표별 해석 규칙과 피해야 할 과잉 해석은 [`diagnostic_metric_interpretation.md`](diagnostic_metric_interpretation.md)에 둔다.

또한 새 graph algorithm을 추가할 때는 adjacency 생성 방식만 기록하지 않고, 어떤 `graph_source`에서 graph를 만들고 어떤 `aggregation_target`에 적용했는지를 함께 명시한다. 이 확장 계약은 [`graph_algorithm_extension.md`](graph_algorithm_extension.md)를 따른다.

---

## 7. Claim 구조

### Main Claim

> graph-based aggregation correction의 성능 향상은 그 자체로 세밀한 graph-specific client relation이 유효하다는 증거가 아니다.  
> 해당 이득은 fine-grained graph relation, coarse clustering, dominance suppression, generic smoothing 중 어디서 오는지 분해해서 평가해야 한다.

### Working Hypothesis

> label-skew Non-IID FL에서 naive graph correction의 이득은 fine-grained graph relation뿐 아니라 coarse clustering, dominance suppression 또는 generic smoothing으로도 설명될 수 있다.

### Engineering Goal

> Non-IID FL 성능 향상을 위해 aggregation correction의 작동 원인을 진단하고, 그 결과를 바탕으로 더 안정적인 aggregation 전략을 선택할 수 있는 실험 프레임워크를 구현한다.

---

## 8. 실험 우선순위

1. **Baseline logging**  
   FedAvg/FedAvgM에서 $q_i$, $d_i$, $DI$, $N_{eff}$를 기록한다.

2. **Control graph 비교**  
   real graph와 random/shuffled/uniform/identity graph를 비교한다.

3. **Clustering control 비교**  
   cluster-only 또는 block-uniform graph를 통해 fine-grained graph relation 없이 client grouping만으로도 성능 향상이 가능한지 확인한다.

4. **Graph-free correction 비교**  
   graph correction과 norm clipping, contribution cap, dominance reweighting을 비교한다.

5. **Setting 확장**  
   $\alpha \in \{0.03, 0.1\}$, $N \in \{5, 20\}$, 데이터셋 2개 이상으로 확장한다.

6. **기존 graph/similarity FL 방법 재현**  
   가능한 범위에서 pFedGraph, FedAMP 등과 동일 프레임워크 내에서 비교한다.

---

## 9. 실험 결과 경우의 수

### Case 1. real graph > clustering control AND real graph > graph-free

fine-grained graph relation 정보가 실제로 유효할 가능성이 높은 결과이다.

> graph-based correction은 coarse clustering을 넘어 edge-level client relation 정보를 활용해 aggregation distortion을 줄일 수 있다.

이 경우 contribution은 기존 graph FL 주장을 clustering control, control graph, graph-free correction 비교를 통해 더 엄밀하게 검증한 것이다.

---

### Case 2. real graph ≈ clustering control AND clustering control > random/uniform

세밀한 edge-level relation보다 coarse clustering이 성능 향상의 핵심일 가능성이 높다.

> 기존 연구에서 충분히 분리되지 않았던 fine-grained graph relation 효과와 coarse clustering 효과를 control experiment로 구분했을 때, 성능 향상의 주된 원인이 client grouping일 수 있음을 보인다.

이 경우 복잡한 graph correction보다 cluster-level aggregation 또는 clustered personalization 방향으로 연구를 전환할 수 있다.

---

### Case 3. real graph ≈ control graph AND graph-free > both

graph relation은 유효하지 않거나 제한적이며, dominance suppression 자체가 성능 향상의 핵심일 가능성이 높다.

> 기존 연구에서 충분히 분리되지 않았던 graph relation 효과와 dominance suppression 효과를 control experiment로 구분했을 때, 성능 향상의 주된 원인이 dominance suppression일 수 있음을 보인다.

이 경우 graph-free dominance-aware correction을 중심으로 연구 방향을 전환할 수 있다.

---

### Case 4. real graph ≈ control graph AND graph-free ≈ real graph AND both > baseline

graph-specific relation보다 generic smoothing 또는 update mixing 효과가 핵심일 가능성이 있다.

> graph correction의 성능 향상은 graph topology 자체보다 update mixing의 regularization 효과로 설명될 수 있다.

이 경우 contribution은 graph-specific claim을 검증하기 위해 control graph 비교가 필요하다는 점을 보이는 것이다.

---

### Case 5. 모두 baseline과 차이 없음

실험한 설정에서는 graph/control/graph-free correction이 baseline 대비 일관된 개선을 보이지 않는 경우이다.

> 해당 setting에서는 aggregation correction의 효과가 제한적이거나 setting-dependent할 수 있다.

이 경우에는 correction 방법의 한계를 단정하기보다, 어떤 조건에서 correction이 작동하지 않는지를 정리하고 추가적인 setting 확장을 향후 과제로 둔다.

---

## 10. 한계 및 향후 발전과제

### 10.1 설계 가이드라인 확장

현재 연구는 graph gain의 원인을 분해하는 데 초점을 맞춘다.  
향후에는 실험 결과를 바탕으로 $DI$, $N_{eff}$, $d_i$를 활용한 aggregation 설계 원칙을 도출할 수 있다.

예를 들어:

- $DI$가 높은 round에서는 dominance-aware reweighting을 적용한다.
- $N_{eff}$가 낮은 경우 contribution balancing을 강화한다.
- real graph가 control graph보다 일관되게 우수한 경우 graph-aware correction을 선택한다.

### 10.2 인과관계 분석 확장

현재 연구는 diagnostic metric과 성능 변화의 관계를 주로 상관관계 수준에서 분석한다.  
향후에는 leave-one-out evaluation, counterfactual update replacement, causal intervention 실험을 통해 인과적 해석을 강화할 수 있다.

### 10.3 이론적 분석 확장

현재 연구는 $d_i$, $DI$, $N_{eff}$와 수렴 경계(convergence bound)의 관계를 직접 증명하지 않는다.  
향후에는 dominance suppression이 update variance, client drift, convergence stability에 미치는 영향을 이론적으로 분석할 수 있다.

---

## 11. 최종 정리

본 연구는 새로운 graph aggregation 알고리즘 하나를 제안하는 것에 초점을 두지 않는다.

대신 Non-IID FL 성능 향상을 위해 graph-based aggregation correction을 사용할 때, 그 성능 향상이 fine-grained client relation 때문인지, coarse clustering 때문인지, dominant update 억제 때문인지, 또는 generic smoothing 때문인지를 분해하고 진단하는 프레임워크를 제안한다.

이를 통해 graph-based FL 연구에서 성능 향상의 원인을 더 명확하게 해석하고, 실험 결과에 따라 graph-aware correction 또는 dominance-aware correction 등 더 안정적인 aggregation 전략으로 확장할 수 있다.
