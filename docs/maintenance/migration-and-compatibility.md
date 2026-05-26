# Migration And Compatibility

## Canonical Surface

| Concept | Canonical |
|---|---|
| package | `graphfl_lab` |
| Flower app | `graphfl_lab.flower_app:server_app`, `graphfl_lab.flower_app:client_app` |
| vision runner | `run_vision_*` |
| unified runner | `run_experiment.py --track vision|cora` |
| vision result | `result_vision_*` |
| vision suite | `vision_suite_*` |
| Graph-FL runtime | `graphfl_lab/strategies/graphfl/` |
| aggregation target | `graph_filtered_*` |
| filter strength | `graph_filter_strength` |

## Compatibility Alias

| Legacy | Current |
|---|---|
| `configs/general/...` | `configs/vision/...` path alias |
| `spectral_filter_strength` | JSON read alias for `graph_filter_strength` |
| `spectral_filtered_*` | aggregation input alias |
| `ours_spectral_filtered_*` | historical reporting tag |
| diagnostic trace key `spectral_filter_gain_*` | metric field name |

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

## Gate Check Contract

```text
python scripts/dev/run.py gate-check <gate>
```

| Gate | Contract |
|---|---|
| Gate 0 | cleanup contract files, canonical docs, golden policy 존재 |
| Gate 1 | rename inventory, protected paths, serialized asset policy |
| Gate 2 | result schema와 compatibility metadata |
| Gate 3 | canonical package root와 legacy import 제거 |
| Gate 4a | unified experiment dispatcher |
| Gate 4b | named track helper |
| Gate 4c | CI, nightly, golden baseline contract |
| Gate 5a-prep | artifact writer extraction contract |
| Gate 5b-prep | suite execution and feature extraction contract |
| Gate 5c-prep | variant parser modularization contract |
| Gate 5d-prep | Graph-FL strategy module contract |
| Gate 6 | release cleanup and public surface contract |

Gate prep status:

| Gate | Status |
|---|---|
| Gate 5a-prep | complete |
| Gate 5b-prep | complete |
| Gate 5c-prep | complete |
| Gate 5d-prep | complete |
| Gate 6 | complete |

## Golden Baseline

Normalized Volatile Field:

```text
timestamp
started_at
completed_at
finished_at
wall_time_sec
total_wall_time_sec
duration_seconds
run_wall_time_sec
seconds_per_round
absolute_path
output_path
canonical_output_path
compatibility_output_path
out_dir
base_dir
diagnostics_dir
plots_dir
reports_dir
snapshots_dir
logs_dir
run_id
host
hostname
python_version
cuda_available
device
```

Schema 비교는 exact match를 사용한다. Value 비교는 volatile field 제거 후 수행한다.

## Line Budget

Measurement:

```text
git diff --numstat <gate-start-sha> -- <protected-paths>
```

Rule:

```text
added - removed <= 0
```

Protected Paths:

```text
graphfl_lab/experiments/suites/vision/variants.py
graphfl_lab/strategies/graphfl/strategy.py
graphfl_lab/experiments/vision/suite.py
```

## Serialized Asset Policy

| Pattern | Policy |
|---|---|
| `*.pkl`, `*.pickle`, `*.pt`, `*.pth` | tracked asset이면 gate에서 명시 분류 |
| `data/Cora/processed/data.pt` | generated cache로 관리 |
| `pickle/checkpoint` import path | canonical package 기준으로 재생성 또는 migration |

## Release And Tombstone Anchor

| Anchor | Status | 의미 |
|---|---|---|
| `pre-graphfl-rename` | created | Gate 0 baseline |
| `1.0.0` | released | post-Gate-6 cleanup release |
| `e647da931bb3a78cc228ac2ad31103537b5ed640` | tombstone anchor | removed-materials comparison anchor |
