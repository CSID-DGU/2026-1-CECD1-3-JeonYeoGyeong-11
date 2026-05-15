# Phase 5. Counterfactual Diagnostic Runner

## Agent Task Card

- Start when: state/relation/topology modules are available and `CURRENT_STATUS.md` names Phase 5 as next.
- Goal: compute same-round actual/control shadow diagnostics without updating the server model on shadow paths.
- Implement only: counterfactual specs/results, minimal aggregation bridge, required topology-destroyed/coarse/graph-free variants, metrics emission, and side-effect tests.
- Do not: implement exact personalized delivery, local objective hooks, hypernetwork/server-GCN methods, or broad CLI changes.
- Required tests: counterfactual runner tests, diagnostics metric tests, then full unittest.
- Stop condition: actual metrics match existing diagnostics, shadow variants emit records, and unsupported variants are explicit rather than missing.

## 목적

네가 원하는 실험은 최종 accuracy만 보는 것이 아니다. 같은 round의 client artifacts를 가지고 “실제로 적용하지 않은 대조 경로”를 계산해서 내부 mechanism을 비교해야 한다.

이 phase에서는 actual training path와 shadow diagnostic path를 분리한다.

```text
actual path:
  real graph -> aggregation -> model update

shadow paths:
  same client states -> random graph -> diagnostics
  same client states -> shuffled graph -> diagnostics
  same client states -> uniform graph -> diagnostics
  same client states -> identity graph -> diagnostics
  same client states -> clustering-only -> diagnostics
  same client states -> graph-free correction -> diagnostics
```

## 새 module

예상 파일:

```text
spectral_fl/lifecycle/counterfactuals.py
spectral_fl/lifecycle/diagnostic_runner.py
tests/lifecycle/test_counterfactual_runner.py
```

## Implementation choice

Phase 5 uses `MinimalAggregationAdapter`.

The adapter is a temporary internal bridge for counterfactual diagnostics.
Phase 6 will promote or replace this logic with the formal `AggregationOperator`.

## Phase 6 의존성 처리

Phase 5는 counterfactual path의 `weights_post`와 `post_flat_updates`를 계산해야 하므로 최소 aggregation 연산에 의존한다. 이 phase에서는 `MinimalAggregationAdapter`를 직접 사용한다.

Phase 5에서 최소한 다음 연산은 실행 가능해야 한다.

```text
global_update
spectral_filtered_update
graphfree_dominance_reweight
```

## 핵심 타입

```text
CounterfactualSpec
- name
- correction_family
- graph_source
- relation_mode
- topology_mode
- aggregation_target
- graph_free_mode
- cluster_method

CounterfactualResult
- name
- adjacency
- weights_post
- post_flat_updates
- metrics
- trace_records
```

## 기본 counterfactual set

| name | 목적 |
|---|---|
| `actual` | 실제 training path |
| `matched_random` | edge count/density가 비슷한 random graph |
| `shuffled` | graph structure는 보존하되 client identity를 섞음 |
| `uniform` | 관계 정보 없이 균일 graph |
| `identity` | graph mixing 없음 |
| `clustering_only` | fine-grained edge 없이 cluster block만 사용 |
| `graphfree_norm_clip` | graph 없이 update norm control |
| `graphfree_contribution_cap` | graph 없이 contribution cap |
| `graphfree_dominance_reweight` | graph 없이 dominance reweight |

`graphfree_norm_clip`과 `graphfree_contribution_cap`은 full default set에는 남긴다. 단, 구현 일정상 필요하면 Phase 6에서 완료해도 된다.

## Component-aware counterfactual generation

Counterfactual generation should be component-aware. Given an actual `GraphFLDesign`, the runner should generate variants by replacing exactly one component family when possible.

### 1. Topology-destroyed variants

- `matched_random`
- `shuffled`
- `uniform`
- `identity`

These variants test whether fine-grained topology matters beyond generic graph mixing.

### 2. Coarse-structure variants

- `clustering_only`
- `block_uniform`

These variants test whether the gain comes from fine-grained edge relations or coarse grouping.

### 3. Graph-free aggregation variants

- `graphfree_norm_clip`
- `graphfree_contribution_cap`
- `graphfree_dominance_reweight`

These variants test whether the gain comes from graph structure or simpler aggregation-level dominance control.

### 4. State-source variants

When compatible alternatives exist, the runner should support changing only the client state source.

Examples:

- update vs weight
- full update vs classifier head
- raw update vs EMA update
- gradient proxy vs update

These variants test whether the gain comes from the graph algorithm itself or from a better state representation.

### 5. Relation-only variants

When compatible alternatives exist, the runner should support changing only the relation estimator.

Examples:

- cosine vs RBF
- cosine vs norm-aware cosine
- relation with sample prior vs relation without sample prior
- learned attention vs non-learned proxy relation

These variants test whether the gain comes from relation estimation rather than topology construction.

### 6. Aggregation-only variants

When compatible alternatives exist, the runner should support changing only the aggregation operator.

Examples:

- update aggregation vs weight aggregation
- graph-filtered aggregation vs cluster-wise aggregation
- graph-filtered aggregation vs graph-free dominance reweighting

These variants test whether the gain comes from graph construction or aggregation target choice.

The counterfactual runner should not only destroy topology. When the design registry exposes compatible alternatives, it should also support state-source, relation-only, and aggregation-only ablations.

## 계산해야 할 값

각 counterfactual path는 최소한 다음을 계산한다.

| metric | 해석 |
|---|---|
| `di_pre/post` | 지배 client contribution이 줄었는가 |
| `neff_pre/post` | 실질 참여 client 수가 늘었는가 |
| `align_mean_pre/post` | aggregate 방향이 client들과 더 일관되는가 |
| `loo_mean_pre/post` | 특정 client 제거에 덜 민감해졌는가 |
| `graph_density` | graph가 얼마나 연결됐는가 |
| `graph_entropy` | edge weight가 얼마나 균형적인가 |
| `alpha_entropy` | aggregation weight가 얼마나 균형적인가 |
| `alpha_matrix_entropy` | personalized row-wise mixture가 얼마나 균형적인가 |

## 기존 diagnostics와의 연결

현재 `summarize_pre_post`와 `append_round_metrics_csv`는 actual path 중심이다. 이 phase에서는 이 로직을 재사용해서 counterfactual별 metrics를 만든다.

예상 artifact:

```text
counterfactual_metrics.csv
```

column 예시:

```text
run_id, variant, seed, round, counterfactual,
di_pre, di_post, neff_pre, neff_post,
align_mean_pre, align_mean_post,
loo_mean_pre, loo_mean_post,
graph_density, graph_entropy, alpha_entropy, alpha_matrix_entropy,
graph_kind, graph_source, aggregation_target
```

## 완료 기준

- actual path 결과와 기존 round diagnostics가 일치한다.
- 같은 round에서 최소 `actual`, `matched_random`, `shuffled`, `uniform`, `identity`, `clustering_only`, `graphfree_dominance_reweight` path를 계산할 수 있다.
- shadow path는 server model을 업데이트하지 않는다.
- counterfactual metrics가 CSV 또는 round trace에 남는다.
- 위 최소 aggregation 연산 3종이 counterfactual runner 경로에서 실제 호출되어 결과를 만든다.
- Counterfactual generation supports topology-destroyed variants.
- Counterfactual generation can be extended to state-source, relation-only, and aggregation-only ablations when compatible alternatives exist.

## 규칙

If a counterfactual path cannot be computed, it must emit an explicit unsupported/error record. It must not silently disappear from the metrics.

## 테스트

```text
python -m unittest tests.lifecycle.test_counterfactual_runner
python -m unittest tests.diagnostics.test_metrics
python -m unittest discover -s tests
```

## 하지 않는 것

- personalized delivery exact 구현
- local objective hook 측정
- hypernetwork/server GCN 계열 exact 구현

이 phase가 끝나면 “real graph가 최종 accuracy를 올렸다”가 아니라, “같은 round에서 real graph가 어떤 내부 지표를 어떻게 바꾸었는가”를 볼 수 있다.

## Files to create

- `spectral_fl/lifecycle/counterfactuals.py`
- `spectral_fl/lifecycle/diagnostic_runner.py`
- `tests/lifecycle/test_counterfactual_runner.py`

## Files allowed to modify

- `spectral_fl/diagnostics/*`
- existing round metrics helper files
- `spectral_fl/lifecycle/__init__.py`

## Files not allowed to modify

- personalized delivery exact implementation
- local objective hook
- hypernetwork/server-GCN implementation
- public CLI behavior unless required for diagnostics flag wiring

## Step-by-step implementation order

Step 5A. Define `CounterfactualSpec` and `CounterfactualResult`.  
Step 5B. Add `MinimalAggregationAdapter` with at least `global_update`, `spectral_filtered_update`, and `graphfree_dominance_reweight`.  
Step 5C. Implement actual-path metric reproduction and verify it matches existing round diagnostics.  
Step 5D. Implement same-round shadow paths for `actual`, `matched_random`, `shuffled`, `uniform`, `identity`, `clustering_only`, and `graphfree_dominance_reweight`.  
Step 5E. Add extension points for state-source, relation-only, and aggregation-only ablations when compatible alternatives are registered.  
Step 5F. Ensure shadow paths never update the server model.  
Step 5G. Emit counterfactual metrics to CSV or round trace.  
Step 5H. Add tests for side-effect isolation, metric consistency, and minimum counterfactual set.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
