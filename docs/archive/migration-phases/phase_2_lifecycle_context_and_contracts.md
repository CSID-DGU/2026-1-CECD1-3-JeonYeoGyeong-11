# Phase 2. Lifecycle Context And Module Contracts

## Agent Task Card

- Start when: Phase 1 trace schema is stable and `CURRENT_STATUS.md` names Phase 2 as next.
- Goal: define lifecycle contexts, module protocols, and `ModuleResult` without migrating graph logic yet.
- Implement only: context dataclasses, module interfaces, result contract, boundary tests, and minimal wiring helpers.
- Do not: move graph source/builders, implement counterfactuals, or add personalized delivery.
- Required tests: `tests.lifecycle.test_module_contracts`, `tests.structure.test_lifecycle_boundaries`, relevant graph registry tests, then full unittest.
- Stop condition: modules can return explicit `ok`, `unsupported`, or `error` results with traces, and boundaries are protected by tests.

## 목적

trace schema 다음에는 lifecycle module이 공통으로 받을 context와 반환 contract를 만든다. 이 phase는 아직 기존 strategy를 대규모로 옮기지 않는다. 먼저 interface를 고정하고, 기존 `graph_source`, `graph_mode`, `aggregation_target`을 이 interface 위에 올릴 준비를 한다.

## 새 contract

예상 파일:

```text
spectral_fl/lifecycle/context.py
spectral_fl/lifecycle/modules.py
tests/lifecycle/test_module_contracts.py
tests/structure/test_lifecycle_boundaries.py
```

핵심 context:

```text
RoundContext (공통 최소 필드만 보유)
- server_round
- cids
- rng
- config
- state_store

StateExtractionContext
- round_context
- global_weights
- local_weights
- local_updates
- num_examples
- client_metrics

RelationContext
- round_context
- client_state_output

TopologyContext
- round_context
- relation_output

AggregationContext
- round_context
- topology_output
- local_updates
- num_examples

ModuleResult
- status: ok | unsupported | error
- support_level: core-supported | proxy-supported | interface-target | out-of-scope
- output
- trace_records
- metadata
- error_type: str | None
- error_message: str | None
```

`unsupported` is a first-class result, not a silent fallback. A module that only exposes an interface for future learned graph modules, hypernetworks, generated personalized models, or local architecture changes must return `status=unsupported` with the appropriate `support_level` instead of substituting a global/default path.

핵심 protocol:

```text
DeliveryPolicy.run(context) -> ModuleResult
LocalObjectiveHook.run(context) -> ModuleResult
ClientStateExtractor.run(context) -> ModuleResult
RelationEstimator.run(context) -> ModuleResult
TopologyOperator.run(context) -> ModuleResult
AggregationOperator.run(context) -> ModuleResult
```

## 구현 범위

1. `RoundContext`와 단계별 세부 context dataclass를 추가한다.
2. module protocol 또는 abstract base class를 추가한다.
3. 기존 strategy 내부 변수들을 “단계별 context”로 옮길 수 있는 helper를 만든다.
4. 아직 실제 aggregation 로직은 기존 경로를 유지한다.
5. module result가 trace bundle과 함께 합쳐질 수 있음을 테스트한다.
6. module 간 import boundary를 테스트로 고정한다.

## 기존 코드와의 연결

현재 strategy의 주요 값은 다음처럼 mapping된다.

| 기존 값 | 새 context 위치 |
|---|---|
| `server_round` | `RoundContext.server_round` |
| `cids` | `RoundContext.cids` |
| `self._current_global` | `RoundContext.global_weights` |
| `local_weights` | `RoundContext.local_weights` |
| `local_updates` | `RoundContext.local_updates` |
| `n_examples_arr` | `RoundContext.num_examples` |
| `client_metrics` | `RoundContext.client_metrics` |
| `graph_rng` | `RoundContext.rng` |

## 완료 기준

- lifecycle package가 생긴다.
- 공통 context와 단계별 context가 분리되어 테스트된다.
- 모듈이 자기 단계 context 외의 내부 구조에 직접 의존하지 않는다.
- 기존 strategy behavior는 바뀌지 않는다.
- 기존 graph registry와 source registry는 그대로 동작한다.
- `ModuleResult` records `status`, `support_level`, and typed error fields.
- unsupported or interface-target components are represented explicitly in traces and tests.

## 테스트

```text
python -m unittest tests.lifecycle.test_module_contracts
python -m unittest tests.structure.test_lifecycle_boundaries
python -m unittest tests.graph.test_graph_registry tests.graph.test_graph_source_registry
python -m unittest discover -s tests
```

## 하지 않는 것

- graph source/builders를 실제 module로 이동
- shadow counterfactual 계산
- delivery/personalized model 구현

이 phase는 “어디에 무엇을 꽂을 것인가”의 코드상 틀만 만든다.

## 경계 규칙

Phase 2부터 아래 규칙을 도입한다.

1. `relation` 모듈은 `topology` 내부 구현을 import하지 않는다.
2. `topology` 모듈은 `aggregation` 내부 구현을 import하지 않는다.
3. 각 단계는 이전 단계의 공개 output 타입만 소비한다.
4. strategy orchestration layer만 단계 간 wiring을 담당한다.
5. unsupported/interface-target behavior must not degrade silently to global/FedAvg/default behavior.

## Files to create

- `spectral_fl/lifecycle/context.py`
- `spectral_fl/lifecycle/modules.py`
- `tests/lifecycle/test_module_contracts.py`
- `tests/structure/test_lifecycle_boundaries.py`

## Files allowed to modify

- `spectral_fl/lifecycle/__init__.py`
- existing strategy tracing/wiring files only if needed for minimal context wiring

## Files not allowed to modify

- graph source/builders migration
- counterfactual runner
- personalized delivery/model code
- public CLI wrappers

## Step-by-step implementation order

Step 2A. Define `RoundContext` and phase-specific context dataclasses.  
Step 2B. Define `ModuleResult` with status, support level, trace records, metadata, and typed error fields.  
Step 2C. Define lifecycle module protocols or abstract base classes.  
Step 2D. Add helpers that convert current strategy values into lifecycle contexts.  
Step 2E. Add tests for context construction and ModuleResult trace merging.  
Step 2F. Add or update import-boundary tests.  
Step 2G. Run phase-specific tests and full test suite.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
