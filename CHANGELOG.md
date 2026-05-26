# 변경 기록

## 1.0.0 - 2026-05-22

| 영역 | 변경 |
|---|---|
| package | canonical package를 `graphfl_lab`으로 통일 |
| Flower app | Flower app path를 `graphfl_lab` 기준으로 정리 |
| legacy surface | `spectral_fl`, `run_general_*`, `general_*` facade 제거 |
| vision artifact | 새 결과명을 `result_vision_*`, `vision_suite_*`로 표준화 |
| compatibility | 과거 JSON/config 입력을 위한 read-only alias 유지 |
| unified runner | `run_experiment.py --track vision|cora` surface 추가 |
