# Cleanup Status

## 현재 상태

| Field | Value |
|---|---|
| status | closed |
| completed gate | Gate 6 |
| release | `1.0.0` |
| package | `graphfl_lab` |
| canonical runner | `run_vision_*`, `run_experiment.py --track vision|cora` |
| canonical output | `result_vision_*`, `vision_suite_*` |

## Gate 6

Gate 6 hard cleanup complete.

| Surface | Status |
|---|---|
| `spectral_fl` package shim | removed |
| `run_general_*` | removed |
| `graphfl_lab/experiments/general/` | removed |
| `graphfl_lab/experiments/suites/general/` | removed |
| `general_suite_*` / `result_general_*` readers/writers | removed |
| `spectral_filtered_*` CLI choices | removed |

## Canonical

현재 migration, compatibility, gate-check contract는 `docs/maintenance/migration-and-compatibility.md`에서 관리한다.
