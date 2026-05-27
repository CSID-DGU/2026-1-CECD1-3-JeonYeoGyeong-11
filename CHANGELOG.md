# 변경 기록

## 1.0.0 - 2026-05-22

| 영역 | 변경 |
|---|---|
| package | canonical package를 `graphfl_lab`로 통일 |
| Flower app | Flower app path를 `graphfl_lab` 기준으로 정리 |
| legacy surface | `spectral_fl`, `run_general_*`, `general_*` facade 정리 |
| vision artifact | 결과 이름을 `result_vision_*`, `vision_suite_*`로 표준화 |
| compatibility | 과거 JSON/config 입력을 위한 read-only alias 유지 |
| unified runner | `run_experiment.py --track vision|cora` surface 추가 |
