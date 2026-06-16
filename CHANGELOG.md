# 변경 기록

## Unreleased

| 영역 | 변경 |
|---|---|
| extension API | source, builder, aggregation target, design 등록 surface 통합 |
| aggregation | custom target 단일 평가 결과를 실제 aggregation과 diagnostics가 공유 |
| CLI | `graphfl component`, `graphfl design`, `graphfl run`과 `--dry-run` 추가 |
| scaffold | poster session workspace, contract test, validation report 생성 |
| demo | canonical capability manifest, Submission V2, Mock DB v2, stale/restore 적용 |
| documentation | root 실행 가이드와 config routing 정리 |
| verification | runtime, CLI, scaffold, manifest, demo contract를 전체 test suite에 포함 |

## 1.0.0 - 2026-05-22

| 영역 | 변경 |
|---|---|
| package | canonical package를 `graphfl_lab`로 통일 |
| Flower app | Flower app path를 `graphfl_lab` 기준으로 정리 |
| legacy surface | `spectral_fl`, `run_general_*`, `general_*` facade 정리 |
| vision artifact | 결과 이름을 `result_vision_*`, `vision_suite_*`로 표준화 |
| compatibility | 과거 JSON/config 입력을 위한 read-only alias 유지 |
| unified runner | `run_experiment.py --track vision|cora` surface 추가 |
