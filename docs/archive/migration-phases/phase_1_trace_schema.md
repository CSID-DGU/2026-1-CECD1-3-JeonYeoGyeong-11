# Phase 1. Standard Trace Schema

## Agent Task Card

- Start when: `CURRENT_STATUS.md` says Phase 1 is active, or a later phase needs trace compatibility work.
- Goal: keep the trace core small and JSON-safe while allowing standardized design-space keys inside `TraceRecord.values`.
- Implement only: trace container/schema helpers, JSON normalization, minimal strategy wiring, and trace tests.
- Do not: promote design-space metadata to top-level dataclass fields, rewrite diagnostics CSV broadly, or implement counterfactuals.
- Required tests: `python -m unittest tests.lifecycle.test_traces`, relevant tracing tests, then `python -m unittest discover -s tests`.
- Stop condition: existing behavior is unchanged, schema version is visible, and later phases can attach metadata through `values`.

## 목적

가장 먼저 표준 trace schema를 만든다. 이 phase의 핵심은 새 알고리즘을 구현하는 것이 아니라, 이후 모든 lifecycle module이 내부 값을 같은 형식으로 남길 수 있게 하는 것이다.

현재 문제는 diagnostics가 strategy 내부 변수와 CSV writer에 강하게 묶여 있다는 점이다. 프레임워크가 되려면 module output과 trace를 분리해야 한다.

```text
module_output, module_trace = module.run(context)
diagnostic_collector.add(module_trace)
```

## 새 contract

예상 파일:

```text
spectral_fl/lifecycle/traces.py
spectral_fl/lifecycle/__init__.py
tests/lifecycle/test_traces.py
```

핵심 타입:

```text
TraceRecord
- phase: str
- module: str
- name: str
- round: int | None
- values: Mapping[str, object]

Standardized design-space keys in `values`
- variant: actual | matched_random | shuffled | uniform | identity | clustering_only | graphfree_* | str
- support_level: core-supported | proxy-supported | interface-target | out-of-scope
- status: ok | unsupported | error
- design_name: str | None
- component_kind: str | None
- component_name: str | None
- input_kind: str | tuple[str, ...] | None
- output_kind: str | None
- is_learned: bool | None
- is_stateful: bool | None
- is_directed: bool | None
- is_symmetric: bool | None
- is_weighted: bool | None
- is_dynamic: bool | None
- is_layerwise: bool | None
- diagnostics: Mapping[str, object]

These design-space keys are standardized entries inside `TraceRecord.values`, not required top-level dataclass fields. Keep the top-level `TraceRecord` shape small and stable so later design-space expansion does not break trace serialization or older diagnostics readers.

RoundTraceBundle
- records: list[TraceRecord]
- add(record)
- extend(records)
- by_phase(phase)
- to_flat_dict(prefix=True)
```

초기 phase name:

| phase | 의미 |
|---|---|
| `delivery` | client에게 내려간 state |
| `local_objective` | local loss hook |
| `client_state` | graph source/state extraction |
| `relation` | raw relation estimation |
| `topology` | graph/topology construction |
| `aggregation` | aggregation weights와 pre/post update |
| `state_store` | round 간 state |
| `diagnostic` | DI, N_eff, LOO 등 공통 지표 |
| `counterfactual` | shadow path 결과 |

## 구현 범위

1. trace dataclass와 bundle container를 추가한다.
2. numpy scalar, ndarray summary, list, dict를 CSV/JSON 가능한 값으로 정규화하는 helper를 둔다.
3. 기존 diagnostics writer는 바로 갈아엎지 않는다.
4. strategy 내부에서 최소 1개 trace bundle을 생성하고, 기존 round trace에 `trace_schema_version` 정도만 넣어 wiring을 확인한다.

## 완료 기준

- 기존 실험 behavior가 바뀌지 않는다.
- `TraceRecord`와 `RoundTraceBundle` 단위 테스트가 있다.
- trace가 JSON 직렬화 가능하다.
- 기존 `round_trace` 또는 diagnostics artifact에 schema version이 남는다.
- trace values, not top-level dataclass fields, can carry design identity, component identity, support level, status, input/output kinds, learned/stateful flags, graph flags, and diagnostics.
- diagnostics include graph density/entropy, `alpha_entropy`, `alpha_matrix_entropy`, DI, N_eff, alignment, and LOO pre/post fields when the producing module has those values.

## 현재 구현 상태

구현 완료:

```text
spectral_fl/lifecycle/traces.py
spectral_fl/lifecycle/__init__.py
tests/lifecycle/test_traces.py
tests/strategies/spectral/test_tracing.py
```

현재 `TraceRecord`와 `RoundTraceBundle`은 numpy scalar, ndarray summary, dict/list/set, non-finite float를 strict JSON 가능한 값으로 정규화한다. 기존 spectral strategy의 `round_trace` payload에는 `trace_schema_version`과 최소 lifecycle trace record가 추가되어, 이후 Phase 2 lifecycle module contract가 같은 schema 위에 붙을 수 있다.

검증:

```text
python -m py_compile spectral_fl/lifecycle/traces.py spectral_fl/strategies/spectral/tracing.py
python -m unittest tests.lifecycle.test_traces tests.strategies.spectral.test_tracing
python -m unittest discover -s tests
```

결과:

```text
trace schema unit tests: 6 tests passed
full unittest suite: 93 tests passed
```

## 테스트

```text
python -m unittest tests.lifecycle.test_traces
python -m unittest discover -s tests
```

## 하지 않는 것

- shadow diagnostic runner 구현
- graph source/builders refactor
- personalized aggregation 구현
- CSV schema 대규모 변경

이 phase는 이후 작업의 계측 기반만 놓는 단계다.

## Agent Note

Phase 1 is treated as completed unless tests fail after later migration.
Future agents should not rewrite the trace schema without an explicit reason.

## Files to create

- none, unless missing from the repository

## Files allowed to modify

- `spectral_fl/lifecycle/traces.py`
- `spectral_fl/lifecycle/__init__.py`
- `tests/lifecycle/test_traces.py`
- `tests/strategies/spectral/test_tracing.py`

## Files not allowed to modify

- diagnostics CSV schema broadly
- counterfactual runner
- graph builders
- personalized aggregation

## Step-by-step implementation order

1. Validate existing Phase 1 behavior first.
2. Edit trace schema files only when a concrete failing test or migration dependency requires it.
3. Keep `trace_schema_version` wiring stable.
4. Update/extend trace tests in the same change, including standardized `values` keys for design-space metadata when they become required by later phases.
5. Run phase-specific tests and full tests.
6. Update `CURRENT_STATUS.md` if any Phase 1 compatibility action was required.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
