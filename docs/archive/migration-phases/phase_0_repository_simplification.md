# Phase 0. Repository Simplification

## Agent Task Card

- Start when: `CURRENT_STATUS.md` says Phase 0 cleanup is active or a later phase is blocked by legacy file layout.
- Goal: separate legacy reports/scripts from the current lifecycle framework path without changing behavior.
- Implement only: archive/move/classify legacy artifacts and preserve compatibility facades.
- Do not: delete protected facades, rewrite core strategy logic, or start lifecycle module implementation.
- Required tests: structure/import tests affected by moves, then `python -m unittest discover -s tests`.
- Stop condition: moved paths are documented, compatibility imports still work, and `CURRENT_STATUS.md` records the cleanup result.

## 목적

lifecycle framework refactor에 들어가기 전에 레포지터리에서 당장 판단을 흐리는 코드를 치워둔다. 이 정리는 코드가 단순히 지저분해서 하는 일이 아니다. 프로젝트 주제가 이전의 semantic/head/EMA graph 실험과 phase별 graph_smooth 분석에서, **graph-FL lifecycle intervention과 mechanism diagnostic framework**로 바뀌었기 때문에 필요하다.

목표는 무조건 삭제가 아니라, 이전 주제의 실험 산출물과 현재 주제의 framework core를 분리하는 것이다. 현재 연구 주제와 직접 관련 없는 오래된 실험 경로는 quarantine/deprecate해서 이후 phase 작업의 표면적을 줄인다.

이 phase는 기능 개발이 아니다. 정리 작업이다.

## 왜 먼저 하는가

프로젝트 주제가 바뀌면서 레포에는 서로 다른 연구 질문의 산출물이 같이 남아 있다.

- root의 `PHASE*.md` 실험 리포트
- `scripts/analysis/phase*` 배치/분석 스크립트
- `graph_smooth`, `dominance_aware` 같은 이전 단계 baseline/실험 경로
- Cora/FGL ablation 경로
- backward-compatible facade 파일
- 새 lifecycle/diagnostic 문서와 코드

이전 산출물은 당시에는 의미가 있었지만, 지금은 새 주장의 중심이 아니다. 이 상태로 lifecycle module refactor를 시작하면, 어떤 경로를 유지해야 하는지 계속 헷갈린다. 그래서 먼저 정리 기준을 잡는다.

## 정리 원칙

1. 바로 삭제하지 말고 먼저 `keep`, `archive`, `deprecate`, `remove`로 나눈다.
2. import boundary 테스트가 보호하는 facade는 삭제하지 않는다.
3. 현재 diagnostic suite와 직접 연결된 코드는 Phase 5 counterfactual runner가 대체 경로를 갖출 때까지 유지한다.
4. 이전 실험 재현에만 필요한 스크립트는 `scripts/legacy/` 또는 `docs/previous/`로 옮긴다.
5. 이동 후에도 전체 테스트를 통과해야 한다.
6. 삭제는 한 번에 하지 않고, 최소 한 phase 동안 deprecated 상태를 둔다.

## 분류 기준

| 등급 | 의미 | 처리 |
|---|---|---|
| `keep` | 현재 lifecycle framework 또는 diagnostic suite에 직접 필요 | 그대로 유지 |
| `archive` | 연구 맥락상 참고는 되지만 실행 경로는 아님 | `docs/previous` 또는 `docs/archive`로 이동 |
| `deprecate` | 아직 import/비교 경로가 남아 있지만 새 구조에서는 대체 예정 | warning/doc marker 추가, 후속 phase에서 제거 |
| `remove` | 테스트/문서/실험에서 참조되지 않고 결과물도 아님 | 삭제 후보 |

## 1차 후보

### Archive 후보

루트의 과거 phase 리포트들은 현재 `docs/current`/`docs/previous` 체계 밖에 있어 시야를 흐린다.

```text
PHASE1_DIAGNOSTICS_REPORT.md
PHASE2_GRAPH_INFORMATIVENESS_REPORT.md
PHASE2_GRAPH_INFORMATIVENESS_*.md
PHASE2_GRAPH_SOURCE_SANITY_REPORT_BATCH1.md
PHASE2_5_SMOOTHING_FAILURE_ANALYSIS.md
PHASE3_DOMINANCE_AWARE_REPORT.md
```

처리안:

```text
docs/archive/legacy-phase-reports/
```

로 이동한다. 내용은 보존하되 현재 claim의 기준 문서로 보지 않는다.

현재 처리 상태:

```text
docs/archive/legacy-phase-reports/
```

에 루트 `PHASE*.md` 리포트를 보존 이동했다.

### Deprecate 후보

이전 실험 경로지만 아직 비교/분석에서 쓰일 수 있는 코드다. 아래 파일들은 원래 `scripts/analysis/`에 있었고, 현재는 `scripts/archive/legacy-analysis/`로 이동했다.

```text
scripts/archive/legacy-analysis/phase1_diagnostics_report.py
scripts/archive/legacy-analysis/phase2_graph_informativeness.py
scripts/archive/legacy-analysis/phase2_graph_source_sanity_suite.py
scripts/archive/legacy-analysis/phase2_5_smoothing_failure.py
scripts/archive/legacy-analysis/phase3_dominance_aware.py
scripts/archive/legacy-analysis/graph_preset_smoke_test.py
scripts/archive/legacy-analysis/pathology_graph_case_*.py
```

이 스크립트들은 현재 lifecycle framework의 main path가 아니므로 새 코드가 의존하면 안 된다.

현재 처리 상태:

```text
scripts/archive/legacy-analysis/
```

로 phase별 이전 주제 분석 스크립트와 pathology case study 스크립트를 이동했다. 이동한 스크립트의 `PROJECT_ROOT` 계산은 새 경로 기준으로 보정했다. 현재 `scripts/analysis/`에는 새 구조 전환 중에도 참고할 수 있는 일반 분석 helper만 남긴다.

### Keep 후보

당장 지우면 안 되는 파일들이다.

```text
run_*.py
spectral_fl/strategy.py
spectral_fl/update_graph.py
spectral_fl/aggregation.py
spectral_fl/client.py
spectral_fl/general_*.py
spectral_fl/spectral_diagnostics.py
```

이 파일들은 thin compatibility facade다. 구조 테스트와 기존 실행 경로 때문에 유지한다. 단, 새 로직을 넣으면 안 된다.

현재 diagnostic/lifecycle 작업에 필요한 경로도 유지한다.

```text
spectral_fl/strategies/spectral/
spectral_fl/graph/
spectral_fl/diagnostics/
spectral_fl/corrections/
spectral_fl/experiments/general/
spectral_fl/experiments/suites/general/
configs/general/diagnostic/
tests/
```

### 검토 후보

현재 연구 주제에서 중심은 아니지만 완전히 버릴지는 확인이 필요하다.

```text
spectral_fl/strategies/baselines/graph_smooth.py
spectral_fl/strategies/baselines/dominance_aware.py
spectral_fl/experiments/cora/
spectral_fl/cli/graph_ablation.py
spectral_fl/experiments/graph_ablation.py
configs/cora/
configs/general/probes/
configs/general/stress/
configs/general/sweeps/
```

처리안:

- graph_smooth/dominance_aware는 Phase 5 counterfactual runner가 들어가기 전까지 비교 기준으로 유지한다.
- Phase 5 이후 기능이 중복되면 deprecated로 내린다.
- Cora/FGL 경로는 논문 scope에 graph-structured data 확장을 넣을지 결정한 뒤 유지/분리한다.

현재 분류:

| 경로 | 현재 등급 | 이유 | 후속 처리 |
|---|---|---|---|
| `spectral_fl/strategies/baselines/graph_smooth.py` | `keep-transitional` | 이전 주제의 baseline이지만 아직 비교 기준과 Cora 경로에서 사용된다. | Phase 5 counterfactual runner가 baseline/control 역할을 흡수하면 deprecated로 내린다. |
| `spectral_fl/strategies/baselines/dominance_aware.py` | `keep-transitional` | 이전 실험 산출물이지만 새 diagnostic 주장과 비교할 수 있는 대조군 역할이 남아 있다. | 새 lifecycle core는 이 구현에 의존하지 않는다. Phase 5 이후 중복 여부를 다시 본다. |
| `spectral_fl/experiments/cora/`, `configs/cora/`, `run_experiment.py`, `run_graph_ablation.py` | `extension-scope keep` | 현재 core는 general lifecycle framework지만, graph-structured data 확장 가능성을 보여줄 수 있는 경로다. 구조 테스트도 이 경로를 compatibility facade로 보호한다. | Phase 7에서 core example인지 extension example인지 문서상 위치를 확정한다. |
| `configs/general/probes/`, `configs/general/stress/`, `configs/general/sweeps/` | `compat/archive keep` | 현재 core 구현의 직접 대상은 아니지만 기존 진단 실험 재현과 회귀 확인에 쓰일 수 있다. | Phase 7 migration review에서 current/previous 위치를 다시 나눈다. |

따라서 Phase 0에서는 위 경로를 삭제하지 않는다. 대신 새 핵심 구현은 `spectral_fl/lifecycle`과 이후 `spectral_fl/designs`에 두어, 이전 baseline 경로가 새 framework의 중심이 되지 않게 한다.

## 안전한 작업 순서

### Step 0A. Inventory

참조 여부를 확인한다.

```text
rg "PHASE1_DIAGNOSTICS_REPORT|phase2_graph_informativeness|TracingGraphSmoothFedAvgM|dominance_aware"
python -m unittest discover -s tests
```

### Step 0B. 문서 이동

가장 안전한 루트 `PHASE*.md`부터 `docs/archive/legacy-phase-reports/`로 옮긴다.

완료 기준:

- `docs/README.md`에 legacy reports 위치가 기록됨
- 테스트 영향 없음

### Step 0C. Legacy script quarantine

`scripts/analysis/phase*` 계열을 `scripts/archive/legacy-analysis/`로 옮긴 상태를 검증한다.

완료 기준:

- 현재 README/문서에서 새 framework 경로가 legacy script를 참조하지 않음
- script tests가 있으면 path 수정

### Step 0D. Baseline 경로 검토

`graph_smooth`, `dominance_aware`는 아직 삭제하지 않는다. lifecycle counterfactual runner가 이 역할을 대체할 때 deprecate한다.

완료 기준:

- 새 phase 문서에 “대체 예정” 표시
- current configs가 여전히 실행 가능

## 이 phase에서 하지 않는 것

- core module 삭제
- run wrapper 삭제
- facade 삭제
- graph_smooth/dominance_aware 즉시 삭제
- Cora/FGL 경로 즉시 삭제

## 처음부터 다시 만들지 않는 이유

주제가 바뀌었기 때문에 새 core는 거의 처음부터 만드는 것이 맞다. 다만 레포 전체를 새로 만드는 것은 아니다. 이미 다음 자산이 유효하다.

- Flower 실행 wrapper
- vision/Cora data loading
- 기존 spectral strategy의 working aggregation path
- graph utilities와 diagnostics helpers
- configs와 unittest 구조

따라서 Phase 0의 정리 목적은 “전부 버리고 새로 시작”이 아니라, 이전 주제 산출물을 분리한 뒤 `spectral_fl/designs`와 `spectral_fl/lifecycle`에 새 core를 만들 공간을 확보하는 것이다.

## 완료 기준

- 루트 디렉터리가 현재 연구 문서와 실행 wrapper 중심으로 단순화된다.
- 오래된 리포트와 phase script가 legacy 영역으로 분리된다.
- 전체 테스트가 통과한다.
- lifecycle phase 1 작업을 시작할 때 어떤 코드를 건드려야 하는지 명확하다.

## Files to create

- `docs/archive/legacy-phase-reports/` if missing
- `scripts/archive/legacy-analysis/` if missing

## Files allowed to modify

- `docs/README.md`
- moved legacy report/script paths only

## Files not allowed to modify

- `run_*.py`
- `spectral_fl/strategy.py`
- `spectral_fl/update_graph.py`
- `spectral_fl/aggregation.py`
- `spectral_fl/client.py`
- `spectral_fl/general_*.py`
- `spectral_fl/spectral_diagnostics.py`
- `spectral_fl/strategies/spectral/*`

## Step-by-step implementation order

1. Run inventory checks and identify archive/deprecate targets.
2. Move legacy phase reports into `docs/archive/legacy-phase-reports/`.
3. Move legacy analysis scripts into `scripts/archive/legacy-analysis/`.
4. Fix moved script path assumptions only when required by tests/imports.
5. Update `docs/README.md` links to new legacy locations.
6. Run phase-specific checks and full tests.
7. Update `CURRENT_STATUS.md`.

## Phase completion report checklist

- Summary
- Files changed
- Tests run
- Behavior changes (should be none)
- Known limitations
- Next phase blockers
