# Phase 4. Client State, Relation, Topology Modules

## Agent Task Card

- Start when: `GraphFLDesign` can resolve current presets and `CURRENT_STATUS.md` names Phase 4 as next.
- Goal: split graph construction into `ClientStateExtractor`, `RelationEstimator`, and `TopologyOperator` with generalized output envelopes.
- Implement only: core/proxy adapters for update/head/EMA states, cosine/RBF/sample-prior relations, and dense/kNN/control/cluster-block topologies.
- Do not: implement learned attention, graph autoencoder training, hypernetworks, mask personalization, aggregation operators, or delivery hooks.
- Required tests: client-state module tests, relation/topology module tests, boundary tests, existing graph tests, then full unittest.
- Stop condition: relation and topology are visibly separate in code and traces, while unsupported learned/interface kinds produce explicit records.

## 목적

기존 `graph_source`와 `graph_mode`를 lifecycle module로 감싼다. 이 phase가 끝나면 “그래프를 만든다”는 작업이 하나의 함수 호출이 아니라 다음 세 단계로 분리된다.

```text
ClientStateExtractor -> RelationEstimator -> TopologyOperator
```

이 분리가 있어야 어떤 graph-FL 방법이 state를 바꾼 것인지, relation estimator를 바꾼 것인지, topology를 바꾼 것인지 내부적으로 검토할 수 있다.

## 새 module

예상 파일:

```text
spectral_fl/lifecycle/client_state.py
spectral_fl/lifecycle/relation.py
spectral_fl/lifecycle/topology.py
tests/lifecycle/test_client_state_modules.py
tests/lifecycle/test_relation_topology_modules.py
```

## ClientStateExtractor

입력:

```text
RoundContext
GraphSourceConfig
```

출력:

```text
ClientStateOutput
- state_kind: str
- payload:
    vectors: list[np.ndarray] | None
    tensors: list[Mapping[str, np.ndarray]] | None
    scalar_features: np.ndarray | None
    pairwise_ready: Any | None
- per_client_meta:
    num_examples
    label_histogram optional
    loss optional
    validation_score optional
    graph_stats optional
    sample_prior optional
- source_used: str
- metadata: dict
```

Supported `state_kind` values include `weights`, `updates`, `gradients`, `pseudo_gradients`, `classifier_head`, `classifier_head_update`, `ema_update`, `layer_slice`, `embedding`, `graph_descriptor`, `hash_signature`, `mixed_moments`, `validation_utility`, `sample_prior`, and `hybrid`.

The framework must not assume that every client state is a single flat vector. Flat vectors are the default core representation, but embeddings, signatures, moment statistics, graph descriptors, validation utilities, and hybrid states must be represented through a common state envelope.

trace:

| key | 의미 |
|---|---|
| `state_norm_mean` | client state norm 평균 |
| `state_norm_min/max` | state norm 범위 |
| `state_cosine_mean/std` | state 간 cosine 분포 |
| `source_used` | 실제 사용된 source |
| `layer_start/end` | layer slice source일 때 선택 구간 |

초기 구현은 기존 `graph_vectors_for_spectral`을 감싸는 adapter로 충분하다.

## RelationEstimator

입력:

```text
ClientStateOutput.payload
relation mode 또는 graph mode의 relation 부분
```

출력:

```text
RelationOutput
- relation_matrix: np.ndarray
- relation_kind: str
- is_symmetric: bool
- is_directed: bool
- is_learned: bool
- raw_scores: np.ndarray | None
- normalized_scores: np.ndarray | None
- relation_meta:
    metric
    temperature
    prior_used
    sample_prior
    validation_split_used
    learned_module_name
    optimization_status
```

Supported `relation_kind` values include `cosine`, `euclidean`, `rbf`, `gradient_alignment`, `signed_conflict`, `norm_aware_cosine`, `validation_utility`, `hamming`, `dtw`, `learned_attention`, `qp_collaboration`, `graph_autoencoder_score`, and `hybrid`.

Relation estimation and topology construction must remain separate. A learned attention score, QP collaboration score, or validation utility matrix is still a relation output. Sparsification, row normalization, clustering, and block construction belong to `TopologyOperator`.

trace:

| key | 의미 |
|---|---|
| `relation_score_mean/std` | raw relation 분포 |
| `relation_entropy` | relation이 uniform에 가까운지 |
| `uses_sample_prior` | pFedGraph류 prior 사용 여부 |
| `sample_prior_entropy` | sample-size prior의 균형도 |

초기 adapter 단계에서도 relation/topology 경계는 섞지 않는다. 기존 `dense_positive_cosine`, `rbf`, `pfedgraph_qp` 경로는 다음처럼 분리 adapter로 감싼다.

1. `RelationEstimator`: pairwise score 또는 relation matrix까지만 계산
2. `TopologyOperator`: sparsification/normalization/graph transform만 계산

trace 분리뿐 아니라 계산 책임 분리도 동시에 만족해야 한다.

## TopologyOperator

입력:

```text
RelationOutput.relation_matrix
topology config
```

출력:

```text
TopologyOutput
- adjacency: np.ndarray
- graph_kind: str
- is_directed: bool
- is_weighted: bool
- is_dynamic: bool
- is_layerwise: bool
- cluster_ids: np.ndarray | None
- masks: Any | None
- layerwise_adjacency: Mapping[str, np.ndarray] | None
- metadata: dict
```

Supported `graph_kind` values include `dense`, `knn`, `threshold`, `directed_top_m`, `cluster_block`, `block_uniform`, `layerwise`, `dynamic`, `learned`, `identity`, `uniform`, `matched_random`, and `shuffled`.

The supported kind lists define the design-space vocabulary that the output envelopes can represent. They are not a Phase 4 implementation checklist. Phase 4 must keep implementation scope tied to support levels.

Clustering can appear either as a topology operation or as an aggregation operation.

Topology-level clustering:

- relation matrix -> cluster ids -> block graph

Aggregation-level clustering:

- cluster ids -> cluster-wise update/weight aggregation
- cluster-specific model storage
- cluster model delivery

The framework must keep these two roles separate.

## Implementation support levels

Phase 4 should separate what the schema can represent from what the first implementation must execute.

| support level | Phase 4 scope |
|---|---|
| core-supported | update/head/EMA client states, cosine/RBF relations, dense/kNN/control topology adapters |
| proxy-supported | sample-prior relation metadata, cluster/block topology adapters, graph-free control compatibility where it touches topology diagnostics |
| interface-target | learned attention, graph-autoencoder scores, learned/dynamic/layer-wise topology, hypernetwork-facing graph metadata, mask-aware graph metadata |
| out-of-scope | exact learned server graph training, exact graph autoencoder training, hypernetwork model generation, local GNN architecture rewrites |

Implementation agents must not try to implement every `state_kind`, `relation_kind`, or `graph_kind` listed above. The generalized envelopes are required so these methods can be represented and traced; executable behavior should be added only for the core/proxy items assigned to this phase.

Learned attention, graph-autoencoder relation scores, hypernetwork-specific layer-wise graphs, and mask-aware personalization should return explicit `interface-target` or `unsupported` records until a later phase adds real execution paths.

trace:

| key | 의미 |
|---|---|
| `graph_density` | edge density |
| `graph_entropy` | edge weight entropy |
| `degree_mean/min/max` | degree statistics |
| `row_entropy_mean` | row-stochastic graph일 때 row entropy |
| `connected_components` | graph connectivity |
| `topology_operator` | dense, knn, threshold, control, clustering 등 |

## 기존 코드 migration

| 기존 위치 | 새 위치 |
|---|---|
| `spectral_fl.graph.sources.spectral.graph_vectors_for_spectral` | `ClientStateExtractor` adapter |
| `spectral_fl.graph.builders._build_legacy_base_client_graph` | `RelationEstimator`와 `TopologyOperator` adapter |
| `build_relation_graph(... correction_family=control_graph)` | Phase 5 shadow runner 또는 topology control adapter |
| `compute_graph_diagnostics` | `TopologyOperator` trace helper |

전환 규칙:

- Phase 4 종료 시점까지 “relation 계산 + topology 변환”이 하나의 함수에 섞여 있는 신규 코드를 허용하지 않는다.
- 기존 legacy 함수는 compatibility 경로로만 남기고, 새 lifecycle 경로에서는 단계별 adapter를 거친다.

## 완료 기준

- 기존 `build_relation_graph` API는 유지된다.
- 내부적으로 새 module adapter를 사용할 수 있다.
- source/relation/topology 각각의 trace가 나온다.
- source/relation/topology 각각이 독립 output 타입과 독립 테스트를 가진다.
- pFedGraph QP mode는 sample prior trace를 남긴다.
- 기존 graph tests가 모두 통과한다.
- `ClientStateOutput` can represent non-flat-vector states through a common envelope.
- `RelationOutput` records whether the relation is directed, symmetric, learned, and prior-based.
- `TopologyOutput` records whether the graph is directed, weighted, dynamic, layer-wise, or cluster-based.
- core/proxy/interface scopes are documented so design-space vocabulary is not mistaken for a requirement to implement every kind immediately.

## 테스트

```text
python -m unittest tests.lifecycle.test_client_state_modules
python -m unittest tests.lifecycle.test_relation_topology_modules
python -m unittest tests.structure.test_lifecycle_boundaries
python -m unittest tests.graph.test_update_graph tests.graph.test_graph_registry tests.graph.test_graph_source_registry
python -m unittest discover -s tests
```

## 하지 않는 것

- aggregation operator 분리
- personalized model delivery
- local objective hook 구현

이 phase의 목표는 graph construction의 원인을 state, relation, topology로 분해할 수 있게 만드는 것이다.

## Warning

New lifecycle code must not combine relation estimation and topology transformation in the same new module. Compatibility wrappers may call legacy functions, but new lifecycle outputs must keep the boundary visible.

## Files to create

- `spectral_fl/lifecycle/client_state.py`
- `spectral_fl/lifecycle/relation.py`
- `spectral_fl/lifecycle/topology.py`
- `tests/lifecycle/test_client_state_modules.py`
- `tests/lifecycle/test_relation_topology_modules.py`

## Files allowed to modify

- `spectral_fl/graph/builders.py`
- `spectral_fl/graph/sources/*`
- `spectral_fl/graph/diagnostics` helpers if needed
- `spectral_fl/lifecycle/__init__.py`

## Files not allowed to modify

- aggregation operator implementation
- personalized delivery
- local objective hook
- public CLI behavior

## Step-by-step implementation order

Step 4A. Define generalized `ClientStateOutput`, `RelationOutput`, and `TopologyOutput` envelopes.  
Step 4B. Implement core `ClientStateExtractor` adapters for update/head/EMA states around the existing graph vector extraction path.  
Step 4C. Implement core/proxy relation estimators for cosine/RBF and pFedGraph-proxy sample-prior tracing where applicable.  
Step 4D. Implement core/proxy topology operators for dense, kNN, threshold/control, and clustering/block-style graph transformations where applicable.  
Step 4E. Preserve the existing `build_relation_graph` public API while routing the new lifecycle path through adapters.  
Step 4F. Add trace records for state, relation, and topology modules.  
Step 4G. Add explicit unsupported/interface-target traces for learned attention, graph autoencoder, hypernetwork, and mask-aware kinds if they are requested before implementation.  
Step 4H. Add tests for each output type and module boundary.  
Step 4I. Run existing graph registry/source tests and full test suite.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
