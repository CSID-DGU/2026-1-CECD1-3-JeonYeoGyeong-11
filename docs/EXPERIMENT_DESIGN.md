# 실험 설계

Client update를 graph signal로 해석하는 연구 아이디어를 실험 가능한 질문과 코드 모듈로 정리한다. 아래 항목은 우선순위가 아니라, 연구 아이디어를 분해한 실험 축이다.

## 1. 중심 가설

Federated Learning에서 각 client update는 독립 벡터의 집합이 아니라, client relation graph 위에 놓인 graph signal로 볼 수 있다.

```text
node: client i
node signal: update delta g_i = w_i - w_global
compressed signal: z_i = R^T g_i
edge: relation between client updates
graph: W_t
laplacian: L_t = D_t - W_t
diagnostic: H_spec(Z | W) = Tr(Z^T L(W) Z) / ||Z||_F^2
```

실제 model update는 압축 벡터 `z_i`가 아니라 원래 full update `g_i`를 사용한다. `z_i`는 graph construction, spectral diagnosis, client interpretation을 위한 표현이다.

## 2. 실험 축

### A. Update-Space Graph Signal

질문:

- client update similarity graph가 random graph와 구분되는 구조를 보이는가?
- update graph 구조가 label skew, domain skew, client drift와 연결되는가?
- 일반 FL과 FGL 모두에서 같은 관점이 성립하는가?

관찰값:

- graph density, degree distribution, edge persistence
- low/high-frequency energy
- spectral entropy, eigengap
- per-client residual
- client label histogram과 residual/cluster의 관계

### B. Graph Construction

graph construction은 단순 구현 옵션이 아니라 frequency basis를 바꾸는 modeling choice다.

현재 구현:

- dense positive cosine graph
- kNN graph
- mutual-kNN graph
- threshold graph
- uniform graph
- random graph with density matched to kNN
- magnitude-aware graph
- global-alignment graph
- EMA temporal graph

추가 후보:

- layer-wise graph
- signed positive/negative graph
- learned graph

실험 질문:

- dense, kNN, uniform, random-matched가 서로 다른 spectral profile을 만드는가?
- kNN이 좋다면 similarity-aware라서 좋은가, sparse graph라서 좋은가?
- temporal/EMA graph가 single-round graph보다 안정적인가?
- graph construction이 accuracy뿐 아니라 spectrum, cluster, residual 해석을 바꾸는가?

### C. Spectral Diagnostics

`H_spec(Z | W)`는 absolute non-IID score가 아니라 graph-conditioned alignment diagnostic으로 해석한다.

질문:

- `H_spec(Z_t | W_t)`와 `H_spec(Z_t | W_ema)`가 어떻게 다른가?
- `H_spec`가 graph density만 따라가는가, 아니면 update dynamics를 추가로 설명하는가?
- high-frequency energy와 per-client residual이 특정 client skew와 연결되는가?
- low-frequency energy가 consensus-like behavior와 연결되는가?

필요한 decoupling:

- graph construction representation과 metric representation 분리
- current graph와 previous/EMA graph 비교
- cosine graph와 random-matched graph 비교
- projection dimension/seed sensitivity 확인

### D. Low/High-Frequency Interpretation

해석 기준:

```text
low frequency: shared/global update direction candidate
high frequency: disagreement or conflict candidate
```

주의점:

- high-frequency는 항상 제거해야 하는 noise가 아니다.
- minority distribution이나 useful heterogeneity일 수 있다.
- aggregation 이전에 client-level 분석이 필요하다.

관찰값:

- residual이 큰 client의 label histogram
- round별 residual 안정성
- high-frequency client의 local/global accuracy
- group fairness 또는 minority client 성능

### E. Clustering and Communities

Laplacian spectrum은 latent client community 탐색으로 이어질 수 있다.

질문:

- spectral cluster가 round 사이에서 안정적인가?
- cluster가 label/domain histogram과 맞물리는가?
- boundary client나 switching client가 존재하는가?
- cluster-first aggregation 또는 personalization으로 확장할 근거가 있는가?

추가 구현:

- per-round eigengap logging은 현재 포함
- cluster assignment, cluster stability, label histogram alignment 분석은 추가 필요

### F. Aggregation and Personalization

aggregation은 아이디어의 끝이 아니라 응용 축이다. 진단과 구조 검증 없이 성능 개선만 주장하면 해석이 약해진다.

현재 구현:

- FedAvg
- diagnostic-only spectral trace
- conflict residual 기반 conservative weight correction
- adaptive tau
- EMA graph
- graph source 선택: `update`, `normalized_update`, `weight`
- aggregation target 선택: `update`, `weight`

추가 후보:

- bounded spectral correction variants
- cluster-first aggregation
- signed graph aggregation
- band-pass/adaptive filter ablation
- cluster-specific head or adapter
- spectral mixture personalization

실험 순서:

1. diagnostic-only로 graph/spectrum trace 확인
2. graph construction 차이가 random-matched를 넘어서는지 확인
3. residual과 client skew의 관계 분석
4. aggregation correction 적용
5. cluster가 안정적이면 personalization으로 확장

### G. Graph Source와 Aggregation Target 분리

기본 구현은 1차 baseline 형태로 둔다.

```text
graph source: client update delta g_i = w_i - w_global
graph signal: z_i = R^T g_i
alpha source: graph-spectral residual / conflict score
aggregation target: original full update delta g_i
```

FedAvg가 원래 평균하던 client update를 그대로 사용하되, sample 수만으로 평균하지 않고 update 사이의 relation graph로 반영비를 보정한다. FL protocol 변경을 최소화하면서 graph/spectral 관점을 넣을 수 있어 baseline extension으로 적합하다.

확장 아이디어는 세 축으로 분리한다.

```text
1. relation graph를 무엇으로 만들 것인가?
2. client 반영비 alpha를 어떻게 만들 것인가?
3. 최종 global update/model을 무엇으로 합칠 것인가?
```

가능한 조합:

| variant | graph source | aggregation target | 해석 |
|---|---|---|---|
| current update graph | update delta `g_i` 또는 projection `z_i` | full update delta `g_i` | 현재 구현. FedAvg의 update space를 유지하면서 graph residual로 반영비만 조정 |
| gradient-informed graph | raw gradient 또는 local training trajectory의 pseudo-gradient | full update delta `g_i` | conflict 판단은 gradient 방향으로 하고, 실제 model update는 기존 FL 방식 유지 |
| gradient aggregation | raw gradient 또는 pseudo-gradient | gradient | FedSGD, gradient surgery, conflict-aware gradient aggregation 계열 확장 |
| weight-state graph | local model weight `w_i` | local weight `w_i` 또는 update delta `g_i` | client가 도착한 model state의 유사도로 graph 구성 |
| layer-wise graph | layer별 update/gradient/weight | layer별 update delta | layer마다 graph와 alpha를 다르게 두는 확장 |
| spectral-filtered aggregation | update graph 위의 low/mid/high-frequency component | filtered update | low-frequency shared direction은 global로 보내고, high-frequency component는 줄이거나 personalization 후보로 분리 |
| cluster-first aggregation | graph community 또는 spectral cluster | cluster별 update/model | 하나의 global model이 모든 client를 설명하지 못할 때 cluster별 aggregation 또는 personalization으로 확장 |

현재 CLI/config에서 바로 가능한 조합:

```text
--graph-source update --aggregation-target update
--graph-source normalized_update --aggregation-target update
--graph-source weight --aggregation-target update
--graph-source update --aggregation-target weight
--graph-source weight --aggregation-target weight
```

suite variant token:

```text
ours_mutual_knn, ours_mutual_knn_k{K}
ours_magnitude
ours_global_alignment
ours_weight_graph
ours_weight_graph_knn_k{K}
ours_weight_agg
```

주의점:

- `g_i = w_i - w_global`은 raw gradient가 아니라 local training 전체가 만든 update delta다. pseudo-gradient로 해석할 수 있지만, 엄밀한 gradient aggregation과는 구분한다.
- gradient 자체를 합치는 실험은 client가 gradient를 반환하도록 protocol을 바꾸거나, local step trajectory를 요약하는 규칙이 필요하다.
- weight로 graph를 만드는 실험은 "어디로 움직였는가"보다 "어디에 도착했는가"를 보는 관점이다. update graph와 다른 구조를 줄 수 있지만 scale sensitivity 점검이 필요하다.
- layer-wise, spectral-filtered, cluster-first 방식은 현재 방식보다 강한 개입이다. diagnostic-only와 graph ablation이 안정적으로 나온 뒤 확장한다.

## 3. 코드 모듈

실험 축을 바꿀 때 수정 대상이 분리되도록 모듈을 나누었다.

```text
spectral_fl/flower_app.py
  Flower ClientApp/ServerApp entrypoint
  result JSON 저장

spectral_fl/flower_runner.py
  local App runner
  flwr run 호환 helper

spectral_fl/config_io.py
  JSON config loading

spectral_fl/projection.py
  flatten_weights
  make_gaussian_projection

spectral_fl/update_graph.py
  dense_positive_cosine
  build_client_graph
  compute_graph_diagnostics
  graph variants

spectral_fl/spectral_diagnostics.py
  laplacian
  spectral_filter
  heterogeneity
  spectral_energy_diagnostics
  normalized_conflicts

spectral_fl/aggregation.py
  compute_tau
  compute_conflict_weights
  weighted_average_by_alpha
  weight entropy/effective clients/min weight floor

spectral_fl/strategy.py
  Flower strategy orchestration

spectral_fl/general_suite_variants.py
  General FL suite variant token -> 실행 옵션 변환

spectral_fl/suite_stats.py
  suite summary 통계와 result JSON helper
```

기본 script 실행은 `flower_app.py`의 App component를 사용한다. 같은 component는 `pyproject.toml`을 통해 Flower App 흐름으로도 연결할 수 있다.

## 4. 실험 Protocol

첫 실험은 현재 구현이 trace를 제대로 남기는지 확인하는 diagnostic-only run으로 둔다.

```powershell
python run_general_suite.py --config configs/general_diagnostic_smoke.json
```

확장 순서:

```text
Phase A: diagnostic-only sanity and graph/spectrum trace
Phase B: frequency decomposition smoke and kNN vs random-matched
Phase C: raw current graph vs EMA temporal graph
Phase D: projection seed/dimension sensitivity
Phase E: client residual and cluster interpretation
Phase F: aggregation correction ablation
```
