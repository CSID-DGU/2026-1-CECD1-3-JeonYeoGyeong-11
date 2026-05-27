# GraphFL Lab Migration

`graphfl_lab`가 canonical package다. Public rename과 cleanup은 release `1.0.0` 기준으로 정리되었다.

## Canonical Surface

| Area | Canonical |
|---|---|
| package | `graphfl_lab` |
| Flower app | `graphfl_lab.flower_app:server_app`, `graphfl_lab.flower_app:client_app` |
| vision runner | `run_vision_*` |
| unified runner | `run_experiment.py --track vision|cora` |
| vision result | `result_vision_*` |
| vision suite | `vision_suite_*` |
| aggregation target | `graph_filtered_*` |
| filter strength | `graph_filter_strength` |

## Compatibility Alias

| Legacy | Current | Role |
|---|---|---|
| `configs/general/...` | `configs/vision/...` path alias | old config path read support |
| `spectral_filter_strength` | JSON read alias for `graph_filter_strength` | old config key read support |
| `spectral_filtered_*` | aggregation input alias | old target naming read support |
| `ours_spectral_filtered_*` | historical reporting tag | old artifact/report parse support |
| diagnostic trace key `spectral_filter_gain_*` | metric field name | old diagnostic field parse support |

## Removed Surface

| Removed | Replacement |
|---|---|
| `spectral_fl` package shim | `graphfl_lab` |
| `run_general_*` root entrypoints | `run_vision_*` |
| `graphfl_lab/experiments/general/` | `graphfl_lab/experiments/vision/` |
| `graphfl_lab/experiments/suites/general/` | `graphfl_lab/experiments/suites/vision/` |
| `graphfl_lab/strategies/spectral/` facade | `graphfl_lab/strategies/graphfl/` |
| `graphfl_lab/general_*`, `graphfl_lab/cli/general_*` | vision modules and unified runner |
| `general_suite_*` artifact readers/writers | `vision_suite_*` |
| `result_general_*` artifact readers/writers | `result_vision_*` |
| CLI `spectral_filtered_*` choices | `graph_filtered_*` |
| `--spectral-filter-strength` | `--graph-filter-strength` |

## Canonical Reference

상세 migration, compatibility, removed-surface, golden baseline, serialized asset policy는 `docs/maintenance.md`에서 관리한다.
