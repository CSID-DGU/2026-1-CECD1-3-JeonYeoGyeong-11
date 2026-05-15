# Phase 6. Aggregation, Delivery, State, Local Hooks

## Agent Task Card

- Start when: counterfactual diagnostics need formal aggregation/delivery/state/hook modules and `CURRENT_STATUS.md` names Phase 6 as next.
- Goal: formalize executable core/proxy aggregation, delivery, state store, and local hook paths while keeping interface-target methods explicit.
- Implement only: listed core aggregation operators, graph-free controls, `AggregationResult`, basic state store, global/previous-personalized delivery, and none/proximal hooks.
- Do not: implement full personalized training loops, FED-PUB graph forward, exact FedAMP/pFedGraph objectives, or hypernetwork/server-GCN execution.
- Required tests: aggregation, delivery, state store, local hook, counterfactual runner tests, then full unittest.
- Stop condition: `alpha` and `alpha_matrix` are distinct, core/proxy paths execute, and interface-target paths fail loudly with trace records.

## 목적

Phase 1-5는 주로 design composer, 현재 spectral strategy의 graph construction, diagnostics를 정리한다. Phase 6은 선행 graph-FL 방법들이 실제로 달라지는 나머지 지점을 연다.

핵심은 네 가지다.

```text
AggregationOperator
DeliveryPolicy
StateStore
LocalObjectiveHook
```

이 phase의 목표는 모든 personalized FL 방법을 exact하게 구현하는 것이 아니다. 다만 interface만 두고 끝내지도 않는다. pFedGraph, FedAMP, FED-PUB 같은 방법이 어느 지점에 들어와야 하는지 코드 contract를 열어두면서, 최소한 “작동하는 얇은 구현”을 포함한다.

지원 수준은 아래처럼 고정한다.

| 수준 | 의미 | 이 phase에서의 역할 |
|---|---|---|
| `core-supported` | 실제 실행 가능 | 프레임워크의 실체 |
| `proxy-supported` | 단순화 버전으로 실행 가능 | 사용 가능성 증거 |
| `interface-target` | contract와 위치만 정의 | 확장 지점 명시 |
| `out-of-scope` | 현재 범위 밖 | 과장 방지 |

## AggregationOperator

예상 파일:

```text
spectral_fl/lifecycle/aggregation.py
tests/lifecycle/test_aggregation_operators.py
```

최소 구현 operator:

| operator | 의미 | 지원 수준 |
|---|---|---|
| `global_update` | FedAvg-style update aggregation | core-supported |
| `spectral_filtered_update` | graph-filtered update aggregation | core-supported |
| `spectral_filtered_ema_update` | EMA update target | core-supported |
| `weight` | model weight aggregation | core-supported |
| `spectral_filtered_weight` | graph-filtered weight aggregation | core-supported |
| `graphfree_norm_clip` | graph 없이 norm 기반 보정 | core-supported |
| `graphfree_contribution_cap` | graph 없이 contribution cap 보정 | core-supported |
| `graphfree_dominance_reweight` | graph 없이 dominance reweight 보정 | core-supported |
| `personalized_weight` | client별 row-wise model mixture | interface-target |
| `gradient_proxy` | gradient proxy aggregation | interface-target |
| `spectral_filtered_gradient_proxy` | graph-filtered gradient proxy aggregation | interface-target |
| `cluster_wise_update` | cluster별 update aggregation | proxy-supported |
| `cluster_wise_weight` | cluster별 weight aggregation | proxy-supported |
| `directed_neighbor_weight` | directed neighbor model mixture | proxy-supported |
| `masked_weight` | mask-aware model aggregation | interface-target |
| `generated_personalized_weight` | hypernetwork or generator-produced personalized weights | interface-target or out-of-scope |

위 표에서 `core-supported`로 표시된 항목은 interface가 아니라 실제 연산 경로를 가져야 한다. Phase 5 counterfactual runner가 `weights_post`, `post_flat_updates`, pre/post metrics를 계산하려면 이 경로들이 실행 가능해야 한다.

trace:

| key | 의미 |
|---|---|
| `alpha` | aggregation weights |
| `q_i` | client contribution |
| `alpha_entropy` | weight balance |
| `pre_post_delta_norm` | aggregation 전후 update norm |
| `personalized_model_distance` | personalized target일 때 global 대비 거리 |

`AggregationResult`:

```text
AggregationResult
- aggregation_target:
    update
    weight
    gradient_proxy
    embedding
    masked_weight
    generated_weight
- global_update: np.ndarray | None
- global_weights: list[np.ndarray] | None
- per_client_weights: list[list[np.ndarray]] | None
- alpha: np.ndarray | None
- alpha_matrix: np.ndarray | None
- cluster_ids: np.ndarray | None
- masks: Any | None
- metadata: dict
```

The framework must distinguish global aggregation weights `alpha` from personalized row-wise mixture weights `alpha_matrix`.

- `alpha`: one global aggregation distribution over clients.
- `alpha_matrix`: client-specific mixture matrix, where each row defines how a target client combines other clients.

Many graph-PFL methods produce a different aggregation distribution for each client. These methods should not be forced into a single global alpha.

## DeliveryPolicy

예상 파일:

```text
spectral_fl/lifecycle/delivery.py
tests/lifecycle/test_delivery_policy.py
```

최소 구현 policy:

| policy | 의미 | 지원 수준 |
|---|---|---|
| `global` | 모든 client에 같은 global model 전달 | core-supported |
| `previous_personalized` | 이전 round client별 model 전달 | proxy-supported |
| `cloud_model` | FedAMP류 personalized cloud model 전달 | interface-target |

`previous_personalized`는 얇은 proxy를 둔다.

- `StateStore.personalized_models`에 값이 있으면 전달
- 기본 정책(default): `global_with_trace`  
  personalized state가 없으면 global model을 전달하되, 아래 trace를 반드시 남긴다.
  - `delivery_policy=previous_personalized`
  - `delivery_cold_start=global_with_trace`
  - `personalized_state_available=false`
- strict 정책(optional): `raise_on_missing_personalized_state`  
  personalized state가 없으면 `MissingPersonalizedStateError`를 발생시킨다.

`cloud_model`은 `NotImplementedError`와 trace contract를 둔다.

## StateStore

예상 파일:

```text
spectral_fl/lifecycle/state_store.py
tests/lifecycle/test_state_store.py
```

저장 대상:

| state | 용도 | 지원 수준 |
|---|---|---|
| `ema_updates` | FedAGA류 accumulated gradient proxy | core-supported |
| `ema_graph` | graph smoothing across rounds | core-supported |
| `previous_relation` | relation drift 측정 | core-supported |
| `personalized_models` | pFedGraph/FED-PUB/FedAMP 확장 | proxy-supported (basic get/set) |
| `cluster_models` | pFedGraph류 local regularization 확장 | interface-target |

`StateStore`는 최소한 `ema_updates`, `ema_graph`, `previous_relation`을 실제 저장/조회할 수 있어야 한다.

## LocalObjectiveHook

예상 파일:

```text
spectral_fl/lifecycle/local_hooks.py
tests/lifecycle/test_local_hooks.py
```

최소 구현 hook:

| hook | 의미 | 지원 수준 |
|---|---|---|
| `none` | 기본 local training | core-supported |
| `proximal_to_delivered_model` | FedAMP/FedProx류 proximal | proxy-supported |
| `cluster_model_regularization` | pFedGraph류 cluster model regularization | interface-target |
| `mask_previous_personalized` | FED-PUB류 mask/previous model regularization | interface-target |

이 phase에서 client training loop를 크게 바꾸지 않는다. 다만 `proximal_to_delivered_model`은 얇은 proxy로 실제 계산 경로를 하나 만든다.

LocalObjectiveHook implementation strategy is fixed to Option B by default.

Primary implementation (Option B):

```text
server sends hook_config via client.fit()
client applies proximal penalty when supported
if unsupported, client raises UnsupportedLocalHookError
the hook must not be silently ignored
```

Contingency path (Option A) is allowed only when Option B is structurally incompatible with the existing Flower client path.

```text
Option A (contingency only):
LocalObjectiveHook.apply(loss, current_weights, delivered_weights, config) -> loss
```

Contingency requirements:

1. Before using Option A, the implementation report must explain why Option B cannot be supported without a large client-loop rewrite.
2. Option A must not be implemented in parallel with Option B unless explicitly required by tests.
3. If Option B fails due to a local bug, fix Option B rather than switching to Option A.

## 완료 기준

- 기존 `_aggregate_target` 로직이 `AggregationOperator`로 감싸지고, 아래 operator가 실제 실행된다:  
  `global_update`, `spectral_filtered_update`, `spectral_filtered_ema_update`, `weight`, `spectral_filtered_weight`, `graphfree_norm_clip`, `graphfree_contribution_cap`, `graphfree_dominance_reweight`
- unsupported personalized operator는 조용히 실패하지 않고 명확한 에러를 낸다.
- `StateStore`가 `ema_updates`, `ema_graph`, `previous_relation`을 실제 저장/조회한다.
- `StateStore.personalized_models`는 basic get/set 수준으로 동작한다.
- delivery는 `global`이 core로 실행되고, `previous_personalized`가 얇은 proxy로 동작한다.
- local hook은 `none`이 core로 동작하고, `proximal_to_delivered_model`이 proxy로 동작한다.
- 기존 실험은 기본 policy/hook으로 동일하게 동작한다.
- 최소 1개의 extension smoke에서 delivery/local/state/aggregation 조합이 실제로 실행된다.
- interface-target 컴포넌트는 명시적 에러 또는 명시적 unsupported trace record를 남긴다. 조용한 default/global fallback은 허용하지 않는다.
- `AggregationResult` distinguishes `alpha` and `alpha_matrix`.
- At least one personalized row-wise mixture path is represented as proxy-supported or interface-target.
- Cluster-wise aggregation is represented as proxy-supported or interface-target.

## 테스트

```text
python -m unittest tests.lifecycle.test_aggregation_operators
python -m unittest tests.lifecycle.test_delivery_policy
python -m unittest tests.lifecycle.test_state_store
python -m unittest tests.lifecycle.test_local_hooks
python -m unittest tests.lifecycle.test_counterfactual_runner
python -m unittest discover -s tests
```

가능하면 아래 조합 smoke를 추가한다.

```text
Design A (core):
global delivery + none hook + update state + cosine relation + kNN topology + spectral_filtered_update

Design B (extension proxy):
global delivery + proximal_to_delivered_model + ema_updates state + rbf/cosine relation + spectral_filtered_weight
```

Design B는 최소한 아래 artifact를 생성해야 한다.

```text
module_traces.jsonl:
  delivery/local_hook/state_store trace records 포함
round_metrics.csv:
  기존 schema 유지
aggregation trace:
  alpha_entropy 포함
state_store trace:
  ema_update_norm(또는 동등 지표) 포함
```

## 하지 않는 것

- client별 personalized model training의 완전 구현
- FED-PUB proxy graph forward
- FedAMP exact local objective
- pFedGraph exact cluster-model local update

이 phase가 끝나면 프레임워크는 “실제로 실행되는 core/proxy 경로”와 “향후 exact personalized method를 붙일 interface-target”을 코드에서 명확히 구분한다.

## Files to create

- `spectral_fl/lifecycle/aggregation.py`
- `spectral_fl/lifecycle/delivery.py`
- `spectral_fl/lifecycle/state_store.py`
- `spectral_fl/lifecycle/local_hooks.py`
- `tests/lifecycle/test_aggregation_operators.py`
- `tests/lifecycle/test_delivery_policy.py`
- `tests/lifecycle/test_state_store.py`
- `tests/lifecycle/test_local_hooks.py`

## Files allowed to modify

- client fit config handling files, only as needed for Option B
- `spectral_fl/lifecycle/counterfactuals.py` if replacing `MinimalAggregationAdapter`
- `spectral_fl/lifecycle/__init__.py`

## Files not allowed to modify

- full personalized training loop
- FED-PUB proxy graph forward
- FedAMP exact local objective
- pFedGraph exact cluster-model local update
- hypernetwork/server-GCN implementation

## Step-by-step implementation order

Step 6A. Promote or replace `MinimalAggregationAdapter` with formal `AggregationOperator`.  
Step 6B. Implement all core-supported aggregation operators and define `AggregationResult` with `alpha` and `alpha_matrix`.  
Step 6C. Implement graph-free norm clip, contribution cap, and dominance reweight operators.  
Step 6D. Add proxy/interface specs for cluster-wise, directed-neighbor, masked, and generated personalized aggregation targets.  
Step 6E. Implement `StateStore` with core-supported `ema_updates`, `ema_graph`, and `previous_relation`.  
Step 6F. Implement `personalized_models` as proxy-supported basic get/set.  
Step 6G. Implement `DeliveryPolicy.global`.  
Step 6H. Implement `DeliveryPolicy.previous_personalized` with `global_with_trace` cold-start default and strict missing-state mode.  
Step 6I. Implement `LocalObjectiveHook.none`.  
Step 6J. Implement `LocalObjectiveHook.proximal_to_delivered_model` through Option B.  
Step 6K. Add extension smoke test for delivery/local/state/aggregation combination.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
