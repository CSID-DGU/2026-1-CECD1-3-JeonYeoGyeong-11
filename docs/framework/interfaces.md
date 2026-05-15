# Composable Graph Algorithm Interfaces

이 프로젝트에서 새 graph-FL 알고리즘을 구현한다는 것은 `strategy.py`에
분기를 계속 추가한다는 뜻이 아니다. 알고리즘을 lifecycle component로
나누고, 각 component를 좁은 인터페이스에 연결한다는 뜻이다.

## Interface Map

| Layer | Question | Interface | Location |
|---|---|---|---|
| Method design | 방법은 어떤 component 조합인가? | `GraphFLDesign`, `ComponentSpec` | `spectral_fl/designs/` |
| Client state | 각 client를 어떤 vector/state로 볼 것인가? | `register_graph_source`, `GraphSourceContext`, `GraphSourceResult` | `spectral_fl/graph/sources/`, `spectral_fl/graph/signals/` |
| Relation estimator | client state 사이 relation을 어떻게 계산할 것인가? | `register_graph_builder`, `GraphBuildContext`, `GraphBuildResult` | `spectral_fl/graph/registry.py` |
| Topology operator | relation을 dense/sparse/control graph로 어떻게 만들 것인가? | graph builder, sparsification helpers | `spectral_fl/graph/sparsification.py`, `spectral_fl/graph/controls.py` |
| Aggregation operator | graph를 update, EMA update, weight 중 어디에 적용할 것인가? | `aggregation_target` | `spectral_fl/strategies/graphfl/targets.py` |
| Runtime strategy | Flower round 안에서 component를 어떤 순서로 호출할 것인가? | `GraphFLDiagnosticStrategy` runtime | `spectral_fl/strategies/graphfl/strategy.py` |
| Diagnostics | graph correction이 무엇을 바꿨는가? | metrics and artifact writers | `spectral_fl/diagnostics/`, `spectral_fl/strategies/graphfl/diagnostics.py` |
| Suite grammar | 어떤 조합을 반복 실행 token으로 노출할 것인가? | variant parser | `spectral_fl/experiments/suites/vision/variants.py` |

## GraphFLDesign

`GraphFLDesign`은 method를 metadata-first로 설명한다. 이 객체는 논문
방법을 바로 실행하는 runtime이 아니라, method가 어떤 lifecycle slot을
차지하는지와 현재 코드에서 어떤 CLI knob로 재현되는지를 기록한다.

필수 slot:

- `client_state`: server가 client를 표현하는 state
- `relation`: state에서 relation을 추정하는 방식
- `topology`: relation을 graph/topology로 바꾸는 방식
- `aggregation`: graph가 update/model/personalization에 개입하는 방식

보조 slot:

- `delivery`: global/personalized model delivery 방식
- `local_objective`: local objective hook
- `state_store`: round 사이에 유지되는 state
- `diagnostics`: trace와 artifact protocol

Prior-work preset은 `core-supported`, `proxy-supported`,
`interface-target`, `out-of-scope` 중 하나의 support level을 가져야 한다.

## Graph Source Interface

새 client representation이 필요하면 graph source를 등록한다.

```python
from spectral_fl.graph import GraphSourceResult, register_graph_source


@register_graph_source("my_client_state")
def build_my_client_state(context):
    vectors = []
    for update in context.local_updates:
        vectors.append(my_vectorize(update))
    return GraphSourceResult(
        vectors=vectors,
        source_used="my_client_state",
        metadata={"state_kind": "my_client_state"},
    )
```

좋은 source는 다음 조건을 만족한다.

- client 순서를 바꾸지 않는다.
- 모든 vector 길이가 같다.
- NaN/Inf를 만들지 않는다.
- metadata에 method-specific 해석 정보를 담는다.
- 기존 source와 의미가 같다면 새 이름을 만들지 않는다.

## Graph Builder Interface

새 relation estimator나 topology가 필요하면 graph builder를 등록한다.

```python
import numpy as np

from spectral_fl.graph import GraphBuildContext, register_graph_builder, require_graph_context


@register_graph_builder("my_relation_graph")
def build_my_relation_graph(context: GraphBuildContext):
    require_graph_context(
        context,
        graph_sources=("classifier_head_update",),
        aggregation_targets=("graph_filtered_update",),
    )
    z = context.z_mat
    adj = np.maximum(z @ z.T, 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj, {"base_graph_kind": "my_relation_graph"}
```

Builder가 반환한 adjacency는 framework에서 shape, finite value,
non-negative weight, symmetry, zero diagonal을 검증하고 정규화한다. Directed
graph나 signed graph가 필요하면 먼저 lifecycle/topology contract를 확장해야
하며, 기존 undirected Laplacian path에 몰래 끼워 넣지 않는다.

## Aggregation Target

`aggregation_target`은 graph가 어디에 개입하는지 정의한다.

| Target | Meaning |
|---|---|
| `update` | FedAvg-style local update aggregation |
| `graph_filtered_update` | current round update matrix를 graph low-pass 후 aggregate |
| `graph_filtered_ema_update` | client update EMA를 graph low-pass 후 aggregate |
| `weight` | local model weight aggregation |
| `graph_filtered_weight` | local model weight를 graph low-pass 후 aggregate |

`spectral_filtered_*`는 기존 config와 결과 호환을 위한 old spelling이다.
새 문서와 새 명령은 `graph_filtered_*`를 우선 사용한다. 새 target이 필요하면
`spectral_fl/strategies/graphfl/targets.py`에서 데이터 흐름을 추가하고 CLI
choices, config, suite variant, diagnostics를 함께 갱신한다.

## Suite Variant Interface

Suite variant token은 연구자가 반복 실행할 조합의 public surface다. Token은
다음 조건을 만족해야 한다.

- 이름만 보고 source/mode/target을 추론할 수 있다.
- random/control counterpart를 만들 수 있다.
- expected result path가 deterministic하다.
- `tests/experiments/vision/`에 parser test가 있다.
- README보다 `extension-guide.md`에 상세 예시를 둔다.

## Diagnostics Contract

Graph algorithm은 성능 숫자만으로는 부족하다. 최소한 다음 질문에 답할 수
있어야 한다.

- graph density와 degree distribution은 어떤가?
- real graph가 random/shuffled/identity control보다 다른가?
- aggregation 전후 dominance, alignment, leave-one-out distortion이 바뀌었는가?
- graph low-pass가 실제 update/weight signal을 얼마나 바꾸었는가?
- 효과가 graph relation 때문인지, cluster effect인지, graph-free dominance
  control 때문인지 비교 가능한가?

## Implementation Checklist

1. Method profile을 문서나 preset metadata로 먼저 적는다.
2. Source가 새로 필요한지 확인한다.
3. Builder가 새로 필요한지 확인한다.
4. `require_graph_context`로 의미 없는 조합을 막는다.
5. `GraphFLDesign` preset을 추가하거나 갱신한다.
6. Suite token은 source/mode/target 경로가 실제로 존재한 뒤 추가한다.
7. Shape, determinism, metadata, diagnostics, control comparability 테스트를 추가한다.
8. 실행 전 `python -m unittest discover -s tests`와
   `python scripts/checks/diagnostic_suite_preflight.py`를 통과시킨다.
