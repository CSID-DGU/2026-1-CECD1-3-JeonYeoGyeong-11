# GraphFL Lab Migration

`graphfl_lab`가 canonical package다. Gate 6 cleanup과 public rename은 release `1.0.0`에서 완료되었다.

## Canonical Surface

| 영역 | Canonical |
|---|---|
| package | `graphfl_lab` |
| vision runner | `run_vision_*` |
| unified runner | `run_experiment.py --track vision|cora` |
| vision result | `result_vision_*` |
| vision suite | `vision_suite_*` |
| aggregation target | `graph_filtered_*` |
| filter strength | `graph_filter_strength` |

## Compatibility Alias

| Legacy | Current |
|---|---|
| `configs/general/...` | `configs/vision/...` path alias |
| `spectral_filter_strength` | JSON read alias for `graph_filter_strength` |
| `spectral_filtered_*` | aggregation input alias |
| `ours_spectral_filtered_*` | historical reporting tag |

## Removed Surface

| Removed | Replacement |
|---|---|
| `spectral_fl` package shim | `graphfl_lab` |
| `run_general_*` | `run_vision_*` |
| `graphfl_lab/experiments/general/` | `graphfl_lab/experiments/vision/` |
| `graphfl_lab/experiments/suites/general/` | `graphfl_lab/experiments/suites/vision/` |
| `general_suite_*` artifact readers/writers | `vision_suite_*` |
| `result_general_*` artifact readers/writers | `result_vision_*` |

## Canonical

세부 migration, compatibility, gate-check contract는 `docs/maintenance/migration-and-compatibility.md`에서 관리한다.
