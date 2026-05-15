# Diagnostic Metric Interpretation Guide

이 문서는 `round_metrics.csv`, `client_metrics.csv`, `graph_stats.csv`에 남기는 지표가 왜 성능 분해에 쓰일 수 있는지, 그리고 어떤 해석은 할 수 없는지를 정리한다.

핵심은 지표 하나로 원인을 단정하지 않는 것이다. 성능 변화는 항상 `variant 비교 + pre/post 지표 변화 + graph 통계`를 함께 보고 해석한다.

## 1. 기본 표기

round $t$에서 client $i$의 raw update를 $g_i$, correction 이후 진단 대상 update를 $\tilde g_i$라고 둔다.

pre-update:

$$
\Delta_{pre} = \sum_i p^{pre}_i g_i
$$

post-update:

$$
\Delta_{post} = \sum_i p^{post}_i \tilde g_i
$$

여기서 $p_i$는 sample weight 또는 correction 이후 aggregation weight다. graph filtering 계열에서는 $\tilde g_i$ 자체가 바뀌고, graph-free reweighting 계열에서는 주로 $p_i$가 바뀐다. 따라서 pre/post 지표는 weight 변화와 update-vector 변화 둘 다를 반영해야 한다.

## 2. 지표별 해석 근거

| 지표 | 무엇을 측정하는가 | 성능 분해에서 말해주는 것 | 주의점 |
|---|---|---|---|
| `q_i` | $p_i\|g_i\|$가 전체 weighted update norm 규모에서 차지하는 비율 | 어떤 client가 global update에 큰 물리적 영향력을 갖는지 보여준다. 큰 $q_i$가 특정 client에 몰리면 dominance pathology 후보가 된다. | 방향 정보는 없다. 큰 update가 항상 나쁜 것은 아니므로 `alignment`, `LOO`와 같이 봐야 한다. |
| `DI = max(q_i)` | 가장 큰 client contribution share | 한 client가 round를 사실상 끌고 가는지 보는 dominance 지표다. correction 후 `DI`가 내려가고 accuracy가 오르면 dominance suppression이 성능에 기여했을 가능성이 커진다. | `DI` 하락만으로 좋은 correction은 아니다. accuracy가 오르지 않으면 useful signal까지 눌렀을 수 있다. |
| `N_eff = 1 / sum(q_i^2)` | 실질적으로 기여하는 client 수 | aggregation이 몇 명의 client에 의존하는지 보여준다. `N_eff` 상승은 contribution이 더 분산되었음을 뜻한다. | 무조건 높을수록 좋은 것은 아니다. Non-IID에서 서로 충돌하는 client를 억지로 섞으면 over-smoothing이 된다. |
| `alignment_i = cos(g_i, Delta)` | client update와 global update의 방향 일치도 | correction 후 mean alignment가 오르면 client updates가 더 일관된 방향으로 합쳐졌다는 증거다. graph relation 또는 clustering이 같은 방향 update를 묶었는지 확인하는 데 쓴다. | dominant client 하나가 global direction을 만들면 그 client alignment만 높아질 수 있다. `DI`, `LOO`로 교차 확인해야 한다. |
| `LOO_i = 1 - cos(Delta, Delta_-i)` | client 하나를 제거했을 때 global update 방향이 흔들리는 정도 | 특정 client 제거가 round 방향을 크게 바꾸면 aggregation이 취약하다. correction 후 LOO 평균/상위값이 줄면 single-client sensitivity가 완화된 것이다. | 모든 LOO가 낮아지는 것이 항상 좋지는 않다. useful minority signal까지 지워졌는지는 accuracy/loss와 봐야 한다. |
| `update_norm_raw/corrected` | raw update와 corrected update의 크기 | graph filtering, norm clipping, smoothing이 실제로 update 크기를 얼마나 줄였는지 보여준다. `q_i` 변화가 weight 때문인지 update-vector 변형 때문인지 분리할 수 있다. | norm 감소 자체는 mechanism이지 성능 근거가 아니다. alignment와 accuracy 개선이 동반되어야 유의미하다. |
| `graph_density` | 가능한 edge 중 실제 edge 비율 | control graph와 real graph의 mixing 기회를 맞추는 기준이다. density가 다르면 graph relation 효과와 단순 mixing 강도가 섞인다. | density가 같아도 edge identity가 다르면 효과는 다를 수 있다. shuffled/random control이 필요하다. |
| `graph_entropy` | edge weight 분포가 얼마나 균등한지 | low entropy는 소수 강한 edge 중심, high entropy는 diffuse smoothing에 가깝다. real graph가 high entropy control과 비슷한 성능이면 generic smoothing 가능성이 커진다. | entropy는 relation quality가 아니라 weight 분포 지표다. label/cluster alignment와 함께 봐야 한다. |
| `alpha_entropy` | aggregation weight 분포의 균등성 | correction이 client weight를 얼마나 균등하게 만들었는지 보여준다. `DI` 하락, `N_eff` 상승과 함께 dominance suppression을 판단한다. | 균등 weighting이 항상 좋은 것은 아니다. harmful client까지 키울 수 있다. |
| `accuracy/loss` | task 성능 결과 | mechanism 지표가 실제 성능 변화와 연결되는지 확인하는 최종 outcome이다. | outcome만으로 원인을 말할 수 없다. 반드시 control과 진단 지표를 같이 보고 해석한다. |

## 3. 메커니즘별 판정 규칙

### Fine-grained graph relation

가능한 증거:

- `real graph`가 `shuffled/random/uniform/identity`보다 성능이 높다.
- `graph_density`, `graph_entropy`가 비슷한 control과 비교해도 real graph의 accuracy/loss가 더 좋다.
- real graph에서 `alignment_post`가 개선되고 `LOO_post`가 안정화되지만, 그 효과가 graph-free dominance correction만으로 재현되지 않는다.
- `cluster-only`보다 real graph가 더 좋으면 coarse grouping 이상의 edge-level relation 가능성이 커진다.

말하면 안 되는 것:

- real graph 성능이 높다는 이유만으로 "graph relation이 유효하다"고 단정하면 안 된다. dominance suppression 또는 smoothing으로도 설명될 수 있다.

### Coarse clustering effect

가능한 증거:

- `cluster-only`가 real graph와 비슷하고 random/uniform보다 높다.
- `cluster_id`별로 alignment 또는 q 변화가 일관되게 나타난다.
- fine-grained edge를 쓰지 않아도 `DI`와 `LOO`가 real graph 수준으로 안정화된다.

해석:

- 성능 향상은 client 간 세밀한 edge weight보다 비슷한 client를 같은 그룹으로 묶는 효과에서 온 것일 수 있다.

### Dominance suppression

가능한 증거:

- correction 후 `DI_post < DI_pre`, `N_eff_post > N_eff_pre`.
- 상위 `q_i` client의 `q_corrected`와 `LOO_corrected`가 내려간다.
- `graph-free_normclip`, `graphfree_cap`, `graphfree_reweight`가 real graph와 비슷한 성능을 낸다.
- 성능 개선이 graph density/entropy 차이보다 `DI` drop 또는 `N_eff` gain과 더 강하게 같이 움직인다.

해석:

- graph가 client relation을 잘 잡아서라기보다, dominant update를 눌러 aggregation을 덜 취약하게 만든 효과일 수 있다.

### Generic smoothing or mixing

가능한 증거:

- `shuffled`, `random`, `uniform` control이 real graph와 비슷한 성능을 낸다.
- graph identity를 깨도 `alignment`, `LOO`, `DI` 변화가 비슷하다.
- `graph_entropy`가 높고 edge weight가 diffuse한 상태에서 성능이 오른다.

해석:

- graph-specific relation보다 update mixing 자체가 regularization처럼 작동했을 수 있다.

### Over-smoothing or under-correction

over-smoothing 가능성:

- `DI`는 낮아지고 `N_eff`는 올라가지만 accuracy/loss가 개선되지 않는다.
- `update_norm_corrected`가 크게 줄고 `alignment_post`가 떨어진다.
- real graph와 uniform graph가 같이 성능을 낮춘다.

under-correction 가능성:

- pre/post `DI`, `N_eff`, `alignment`, `LOO`가 거의 변하지 않는다.
- baseline과 성능 차이가 없다.

## 4. 보고서에서 써야 할 문장 형식

좋은 해석 문장:

> `ours_real_graph_k2`는 FedAvg보다 accuracy가 높았지만, `ours_graphfree_reweight`와 유사한 `DI` 감소와 `N_eff` 증가를 보였다. 따라서 이 setting의 이득은 fine-grained graph relation보다 dominance suppression으로 설명될 가능성이 크다.

좋은 해석 문장:

> real graph가 density/entropy가 맞춰진 shuffled control보다 높은 accuracy와 높은 `alignment_post`를 보였고, graph-free correction은 같은 성능을 재현하지 못했다. 따라서 relation-specific signal 가능성이 남는다.

피해야 할 문장:

> real graph가 FedAvg보다 높으므로 graph relation이 유효하다.

피해야 할 문장:

> `DI`가 낮아졌으므로 성능이 좋아진 원인은 dominance suppression이다.

## 5. 최소 보고 단위

성능 분해를 주장하려면 variant별로 최소한 다음을 함께 보고해야 한다.

- `mean_delta_vs_fedavg`
- `mean_di_drop = mean(DI_pre - DI_post)`
- `mean_neff_gain = mean(N_eff_post - N_eff_pre)`
- `mean_alignment_gain = mean(alignment_post - alignment_pre)`
- `mean_loo_drop = mean(loo_pre - loo_post)`
- `mean_graph_density`, `mean_graph_entropy`
- `graph-free`와 `control graph` 대비 성능 차이

이 조합을 사용하면 "성능이 올랐다"에서 멈추지 않고, 그 성능이 relation-specific effect, clustering effect, dominance suppression, generic smoothing 중 어디에 가까운지를 제한적으로 분해할 수 있다.
