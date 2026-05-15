# Phase 7. Migration, Validation, And Docs

## Agent Task Card

- Start when: lifecycle modules are implemented through Phase 6 and `CURRENT_STATUS.md` names Phase 7 as next.
- Goal: migrate the default strategy path, standardize artifacts, validate behavior parity, and align docs with actual support levels.
- Implement only: strategy wiring, artifact writer alignment, config/help/docs updates, parity checks, and smoke validation.
- Do not: add new algorithm families, rewrite public CLI flags without documentation, or change unrelated experiment suites.
- Required tests: full unittest, boundary tests, CLI help commands, and a diagnostic smoke run when feasible.
- Stop condition: artifacts are produced, docs match code support levels, actual path parity is checked, and `CURRENT_STATUS.md` records remaining limitations.

## 목적

마지막 phase는 새 lifecycle framework가 기존 실험 시스템과 문서에 자연스럽게 연결되는지 확인한다. 이 단계에서는 기능을 크게 늘리기보다, 이전 phase에서 만든 contract가 실제 연구 workflow로 쓸 수 있는지 정리한다.

## Migration 대상

| 대상 | 작업 |
|---|---|
| `SpectralConflictAwareStrategy` | lifecycle context, modules, trace bundle을 기본 경로로 사용 |
| diagnostics CSV | round/client/graph/counterfactual artifact를 새 schema 기준으로 정리 |
| config files | diagnostic suite variant가 lifecycle module 이름과 맞는지 점검 |
| CLI help | graph source/mode/target이 lifecycle module의 하위 노브임을 설명 |
| docs | exact/proxy/interface/out-of-scope 기준을 최신 코드와 동기화 |
| tests | old behavior compatibility와 new module trace를 모두 검증 |

## Artifact 목표

최소 artifact:

```text
round_metrics.csv
client_metrics.csv
graph_stats.csv
counterfactual_metrics.csv
module_traces.jsonl
```

각 artifact 역할:

| artifact | 역할 |
|---|---|
| `round_metrics.csv` | actual path의 round-level 성능과 core diagnostics |
| `client_metrics.csv` | client별 q, norm, alignment, LOO, cluster id |
| `graph_stats.csv` | actual graph의 topology stats |
| `counterfactual_metrics.csv` | shadow path별 diagnostics |
| `module_traces.jsonl` | 모든 lifecycle module의 raw trace |

## Validation 질문

코드 migration이 끝나면 다음 질문에 답할 수 있어야 한다.

1. 같은 seed에서 actual path 결과가 migration 전과 같은가
2. actual path의 기존 diagnostics와 새 trace diagnostics가 일치하는가
3. control graph와 graph-free counterfactual이 같은 round artifacts를 사용했는가
4. pFedGraph-inspired proxy preset(`pfedgraph_proxy`)이 sample prior trace를 남기는가
5. unsupported personalized target이 명확히 interface error를 내는가
6. docs의 지원 범위가 코드의 실제 지원 범위와 맞는가
7. 연구자가 `GraphFLDesign`에서 state/relation/topology/aggregation 중 하나만 교체해도 실행 경로가 유지되는가
8. module boundary(import contract)가 테스트로 보호되는가

우선순위:

- 졸업 프로젝트 최소 주장에 필요한 핵심 질문: 1~3
- 확장 주장(선행연구형 proxy, sweep, trace 해석 확장): 4~8

## 테스트 세트

최소 테스트:

```text
python -m unittest discover -s tests
python -m unittest tests.structure.test_lifecycle_boundaries
python run_general_experiment.py --help
python run_general_suite.py --help
python run_experiment.py --help
```

가능하면 smoke:

```text
python run_general_experiment.py \
  --method ours \
  --dataset fashionmnist \
  --num-clients 5 \
  --rounds 2 \
  --train-subset-size 1000 \
  --test-subset-size 500 \
  --diagnostics-enable true \
  --loo-enabled true
```

## 완료 기준

- 전체 unittest 통과
- help command 통과
- diagnostic smoke run에서 artifact가 생성된다
- `counterfactual_metrics.csv`가 diagnostics enabled일 때 최소 `actual`, `matched_random`, `shuffled`, `clustering_only`, `graphfree_dominance_reweight` record를 포함한다
- docs의 phase plan, lifecycle framework, extension guide가 서로 모순되지 않는다
- design 변형 smoke에서 “한 축만 교체한 design”이 최소 1개 이상 정상 실행된다

## 논문 관점 산출물

이 phase가 끝나면 method section은 다음 구조로 쓸 수 있어야 한다.

1. Lifecycle decomposition
2. Module contracts
3. Standard trace schema
4. Counterfactual diagnostic runner
5. Supported design space and limitations

실험 section은 다음 질문을 따라간다.

1. Real graph vs matched controls
2. Real graph vs clustering-only
3. Real graph vs graph-free correction
4. Prior-work-inspired presets
5. Non-IID severity sweep
6. Internal trace analysis before final accuracy

권장 발표 우선순위:

- 필수(core): 1~3
- 확장(extension): 4~6

이 phase는 구현의 끝이 아니라, 연구 주장과 코드가 서로 맞물리는지 확인하는 마감 단계다.

## Files to create

- none required unless adding docs such as extension guide

## Files allowed to modify

- `SpectralConflictAwareStrategy` implementation files
- diagnostics artifact writer files
- config files under diagnostic/general scope
- CLI help text
- docs
- `CURRENT_STATUS.md`

## Files not allowed to modify

- old legacy reports except links/locations
- unrelated experiment suites
- public CLI flags unless explicitly documented

## Step-by-step implementation order

Step 7A. Migrate `SpectralConflictAwareStrategy` to use lifecycle context/modules/trace bundle as the default path.  
Step 7B. Verify actual-path behavior against pre-migration behavior on the same seed.  
Step 7C. Standardize artifact outputs.  
Step 7D. Verify `counterfactual_metrics.csv` contains actual/control/cluster/graph-free paths.  
Step 7E. Verify docs support levels match code support levels.  
Step 7F. Verify CLI help still works.  
Step 7G. Run diagnostic smoke.  
Step 7H. Update docs and `CURRENT_STATUS.md`.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers
