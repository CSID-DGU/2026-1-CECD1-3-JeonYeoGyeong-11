# Phase 3. GraphFLDesign Composer And Registry

## Agent Task Card

- Start when: Phase 2 contracts exist and `CURRENT_STATUS.md` names Phase 3 as next.
- Goal: introduce `GraphFLDesign`, `ComponentSpec`, registry, presets, and mutation helpers as metadata-first design objects.
- Implement only: design dataclasses, registry, built-in/proxy presets, compatibility aliases, `to_legacy_args`, and design trace metadata.
- Do not: implement every lifecycle module, migrate the full strategy, or claim exact reproduction for proxy presets.
- Required tests: design registry/preset tests, graph method-spec tests, lifecycle contract tests, then full unittest.
- Stop condition: existing graph presets resolve through design metadata and at least one built-in design can be traced by name/support level.

## 목적

이 phase는 프레임워크의 중심 단위를 `graph_source`, `graph_mode`, `aggregation_target`의 흩어진 옵션 묶음에서 `GraphFLDesign`으로 올린다.

이 구조가 있어야 연구자가 graph-FL 설계를 조립하고, 변형하고, 새로 만들기 쉬워진다는 주장을 할 수 있다. 핵심은 선행연구 구현체를 전부 복사하는 것이 아니라, 반복적으로 등장하는 설계 결정을 하나의 design object로 표현하는 것이다.

```text
GraphFLDesign
  -> ClientStateExtractor
  -> RelationEstimator
  -> TopologyOperator
  -> AggregationOperator
  -> DeliveryPolicy
  -> StateStore
  -> LocalObjectiveHook
  -> DiagnosticProtocol
```

초기 구현은 얇게 시작한다. 곧바로 모든 module class를 요구하지 않고, 현재 CLI/config 노브를 하나의 design metadata로 묶는다.

## 왜 Phase 3인가

Phase 1-2에서 trace와 lifecycle context를 만든 뒤 바로 state/relation/topology module로 들어가면, 모듈은 생기지만 “연구자가 조립하는 설계 단위”가 약해진다.

따라서 state/relation/topology를 본격적으로 감싸기 전에 `GraphFLDesign`을 먼저 둔다.

```text
source/mode/target compatibility knobs
  -> GraphFLDesign
  -> lifecycle modules
  -> trace + counterfactual diagnostics
```

## 새 contract

예상 파일:

```text
spectral_fl/designs/design.py
spectral_fl/designs/registry.py
spectral_fl/designs/presets.py
spectral_fl/designs/prior_work.py
spectral_fl/designs/__init__.py
tests/designs/test_design_registry.py
tests/designs/test_design_presets.py
```

초기 타입:

```text
GraphFLDesign
- name: str
- client_state: ComponentSpec
- relation: ComponentSpec
- topology: ComponentSpec
- aggregation: ComponentSpec
- delivery: ComponentSpec
- local_objective: ComponentSpec
- state_store: ComponentSpec
- diagnostics: ComponentSpec
- support_level: core-supported | proxy-supported | interface-target | out-of-scope
- tags: tuple[str, ...]
- description: str
- references: tuple[str, ...]

ComponentSpec
- kind: str
- name: str
- params: Mapping[str, object]
- support_level: core-supported | proxy-supported | interface-target | out-of-scope
- is_learned: bool
- is_stateful: bool
- input_kind: tuple[str, ...]
- output_kind: str
- trace_keys: tuple[str, ...]
```

초기에는 `ComponentSpec`이 실제 class instance일 필요는 없다. config/metadata wrapper로 시작해서 이후 phase에서 실제 lifecycle module로 resolve한다.

단, metadata-only 상태를 오래 유지하지 않는다. 이 phase 완료 시점에는 최소 1개의 built-in design이 실제 lifecycle module wiring으로 resolve되어야 한다.

## Built-in design examples

| design | 구성 | 의미 |
|---|---|---|
| `head_knn_filtered_update` | classifier head update + cosine + kNN + spectral filtered update | 현재 diagnostic real graph 기본형 |
| `ema_magnitude_knn_filtered` | EMA update + magnitude-aware kNN + spectral filtered EMA update | FedAGA류 proxy |
| `pfedgraph_proxy` | update + sample prior + cosine-difference QP + filtered update proxy | pFedGraph-inspired proxy design |
| `fedamp_proxy` | weight + RBF relation + spectral filtered weight | FedAMP-inspired proxy design |
| `sfl_proxy` | weight + learned smooth graph + spectral filtered weight | SFL-inspired proxy design |
| `graphfree_dominance_reweight` | update + no graph + dominance reweight | graph-free control design |

prior-work 계열 preset은 기본적으로 `support_level=proxy-supported`를 사용한다. 문서, trace, CLI-visible 설명 어디에서도 exact reproduction처럼 보이게 표현하지 않는다.
Prior-work proxy presets must never be described as exact reproductions.

## Intended design families

Prior-work-inspired designs must expose their support level explicitly. The framework should never describe proxy presets as exact reproductions. For example, `fedamp_proxy`, `pfedgraph_proxy`, and `sfl_proxy` are lifecycle approximations that preserve the design pattern, not faithful reproduction of the full original algorithms.

| design family | example methods | expected support |
|---|---|---|
| `model_distance_dense_weight` | FedAMP-style | proxy-supported |
| `validation_utility_directed_weight` | FedFomo-style | proxy-supported |
| `client_relation_graph_weight` | SFL / pFedGraph-style | proxy-supported |
| `cluster_only_update` | GCFL / FedCCH-style | proxy-supported |
| `embedding_threshold_weight` | GPFedRec / FED-PUB-style | proxy-supported or interface-target |
| `learned_attention_graph` | pFedGAT-style | interface-target |
| `hypernetwork_generated_weight` | FedAGHN / FedSheafHN-style | interface-target or out-of-scope |
| `masked_graph_personalization` | ADPFedGNN / FED-PUB-style | interface-target |

## 기존 preset과의 관계

현재 있는 파일:

```text
spectral_fl/graph/presets.py
spectral_fl/graph/method_specs.py
```

은 바로 삭제하지 않는다. 이 phase에서는 다음처럼 흡수한다.

| 기존 | 새 위치 |
|---|---|
| `graph_preset_names()` | `design_names()` compatibility wrapper |
| `resolve_graph_preset_spec()` | `resolve_design(...).to_legacy_args()` |
| `GraphFLMethodSpec` | `GraphFLDesign` metadata 또는 `prior_work.py` |
| `fedamp_like`, `pfedgraph_like`, `sfl_like` | 호환 alias로 유지하되 내부적으로 `fedamp_proxy`, `pfedgraph_proxy`, `sfl_proxy`로 resolve |

기존 CLI의 `--graph-preset`은 당장은 유지하되 내부적으로 design preset을 resolve하도록 바꾼다. 새 CLI 옵션은 나중에 `--graph-design`으로 추가할 수 있다.

## Design 변형

이 phase에서 중요한 것은 design을 쉽게 변형하는 API다.

예상 helper:

```text
design.with_client_state(...)
design.with_relation(...)
design.with_topology(...)
design.with_aggregation(...)
design.with_diagnostics(...)
```

이 helper가 있어야 counterfactual runner가 다음 변형을 자동 생성할 수 있다.

```text
actual design
same state + random matched topology
same state + shuffled topology
same state + clustering-only topology
same aggregation + graph-free correction
```

이 변형 API가 있어야 연구자가 “하나의 축만 바꾼 실험”을 빠르게 만든다. 즉 이 phase의 성공 기준은 preset 개수가 아니라, 조립/변형 연산의 일관성이다.

## Trace 연결

`GraphFLDesign`은 최소한 아래 metadata를 trace와 결과 JSON에 남긴다.

| key | 의미 |
|---|---|
| `design_name` | design 이름 |
| `client_state.name` | state extractor 이름 |
| `relation.name` | relation estimator 이름 |
| `topology.name` | topology operator 이름 |
| `aggregation.name` | aggregation operator 이름 |
| `support_level` | core-supported/proxy-supported/interface-target/out-of-scope 구분 |
| `design_tags` | prior-work, control, graph-free 등 |

## 완료 기준

- `GraphFLDesign`과 `ComponentSpec` dataclass가 있다.
- design registry에 built-in design을 등록/조회할 수 있다.
- 기존 `graph_preset`이 design preset으로 resolve될 수 있다.
- `GraphFLDesign.to_legacy_args()`가 기존 strategy 실행 노브를 만든다.
- 최소 1개 built-in design이 실제 lifecycle module spec으로 resolve되고 round trace에 module 이름이 남는다.
- design metadata가 meta 또는 diagnostics trace에 남을 준비가 된다.
- 기존 graph preset tests가 통과한다.
- `ComponentSpec` records whether the component is learned, stateful, and what input/output kinds it expects.
- prior-work-inspired design families are represented as proxy or interface presets, not exact reproductions.

## 테스트

```text
python -m unittest tests.designs.test_design_registry
python -m unittest tests.designs.test_design_presets
python -m unittest tests.graph.test_graph_method_specs
python -m unittest tests.lifecycle.test_module_contracts
python -m unittest discover -s tests
```

## 하지 않는 것

- 모든 lifecycle module class 구현
- strategy 전체 migration
- counterfactual runner 구현
- personalized aggregation exact 구현

이 phase는 연구자가 조립하는 “설계 단위”를 먼저 코드에 만드는 단계다. 이 단위가 있어야 이후 lifecycle modules와 diagnostics가 하나의 프레임워크로 묶인다.

## Files to create

- `spectral_fl/designs/design.py`
- `spectral_fl/designs/registry.py`
- `spectral_fl/designs/presets.py`
- `spectral_fl/designs/prior_work.py`
- `spectral_fl/designs/__init__.py`
- `tests/designs/test_design_registry.py`
- `tests/designs/test_design_presets.py`

## Files allowed to modify

- `spectral_fl/graph/presets.py`
- `spectral_fl/graph/method_specs.py`
- `spectral_fl/lifecycle/__init__.py` if needed

## Files not allowed to modify

- full strategy migration
- counterfactual runner
- personalized aggregation exact implementation

## Step-by-step implementation order

Step 3A. Add `ComponentSpec` and `GraphFLDesign` dataclasses, including support level, learned/stateful flags, input/output kinds, and trace keys.  
Step 3B. Add design registry.  
Step 3C. Add built-in design presets.  
Step 3D. Add compatibility mapping from legacy graph presets to design presets.  
Step 3E. Add `to_legacy_args()` so current strategy execution can still run.  
Step 3F. Add `with_client_state`, `with_relation`, `with_topology`, `with_aggregation`, and `with_diagnostics` helpers.  
Step 3G. Add trace metadata for design name, component names, support level, and tags.  
Step 3H. Add tests for registry, presets, compatibility aliases, and design mutation.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
