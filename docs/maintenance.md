# Maintenance And Compatibility

이 문서는 현재 이름, 호환 alias, 제거된 경로, golden baseline, 생성 asset 기준을 정리한다. 기준 package는 `graphfl_lab`이다.

## Current Status

| Field | Value |
|---|---|
| release | `1.0.0` |
| package | `graphfl_lab` |
| canonical runner | `graphfl run single|suite|ablation|stress|client-count` |
| canonical output naming | `result_vision_*`, `vision_suite_*` |
| framework surface | lifecycle + graph + diagnostics + designs + validation |

## Canonical Surface

| Concept | Canonical |
|---|---|
| package | `graphfl_lab` |
| Flower app | `graphfl_lab.flower_app:server_app`, `graphfl_lab.flower_app:client_app` |
| authoring API | `graphfl_lab.extensions` |
| authoring CLI | `graphfl component`, `graphfl design`, `graphfl run` |
| module fallback | `python -m graphfl_lab.cli.main` |
| vision runner | `run_vision_*` |
| compatibility wrappers | `run_experiment.py`, `run_vision_*`, `run_graph_ablation.py` |
| vision result | `result_vision_*` |
| vision suite | `vision_suite_*` |
| Graph-FL runtime | `graphfl_lab/strategies/graphfl/` |
| aggregation target | `graph_filtered_*` |
| filter strength | `graph_filter_strength` |

문서와 새 예시는 canonical 이름을 사용한다. 예전 이름은 old config/result를 읽기 위한 compatibility path로만 남긴다.

## Compatibility Alias

| Legacy | Current | Compatibility Role |
|---|---|---|
| `configs/general/...` | `configs/vision/...` path alias | 예전 config path 읽기 |
| `spectral_filter_strength` | JSON read alias for `graph_filter_strength` | 예전 config key 읽기 |
| `spectral_filtered_*` | aggregation input alias | 예전 target 이름 읽기 |
| `ours_spectral_filtered_*` | historical reporting tag | 예전 artifact/report 읽기 |
| diagnostic trace key `spectral_filter_gain_*` | metric field name | 예전 diagnostic field 읽기 |

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

## Config Tree

```text
configs/
├── vision/
│   ├── diagnostic/
│   │   ├── smoke/
│   │   ├── core/
│   │   └── extend/
│   ├── probes/
│   ├── smoke/
│   ├── stress/
│   └── sweeps/
└── cora/
    └── ablations/
```

| Folder | Purpose |
|---|---|
| `configs/vision/diagnostic/smoke/` | 짧은 diagnostic 확인 |
| `configs/vision/diagnostic/core/` | 기본 diagnostic suite |
| `configs/vision/diagnostic/extend/` | 확장 diagnostic config |
| `configs/vision/probes/` | graph/source/target probes |
| `configs/vision/stress/` | Non-IID stress configs |
| `configs/vision/sweeps/` | sweep configs |
| `configs/cora/ablations/` | Cora graph ablation |

## Golden Baseline

`tests/golden/`은 normalized smoke output을 보관한다. Schema comparison은 exact match이고, value comparison은 아래 volatile fields만 제외한다.

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

| Golden Change | Required Record |
|---|---|
| schema change | changed field, reason, affected fixture |
| value expectation change | metric/result reason, affected fixture |
| volatile-field change | added/removed field and comparison effect |

## Line Budget

Measurement:

```text
git diff --numstat <baseline-sha> -- <protected-paths>
```

Rule:

```text
added - removed <= 0
```

Protected paths:

```text
graphfl_lab/experiments/suites/vision/variants.py
graphfl_lab/strategies/graphfl/strategy.py
graphfl_lab/experiments/vision/suite.py
```

Exception format:

```text
path | expires_on | reason
```

Current active exception: none.

## Serialized Asset Policy

| Pattern | Policy |
|---|---|
| `*.pkl`, `*.pickle`, `*.pt`, `*.pth` | tracked asset이면 release 전에 역할과 regeneration path를 명시 |
| `data/Cora/processed/data.pt` | generated cache |
| `pickle/checkpoint` import path | canonical package path로 regenerate 또는 migrate |

## Release And Tombstone Anchors

| Anchor | Status | Meaning |
|---|---|---|
| `pre-graphfl-rename` | created | rename baseline |
| `1.0.0` | released | post-cleanup release |
| `e647da931bb3a78cc228ac2ad31103537b5ed640` | tombstone anchor | removed-materials comparison anchor |

## Cleanup Sequence

| Step | Surface | Result |
|---:|---|---|
| 1 | serialized asset check | generated/cache asset boundary recorded |
| 2 | duplicate artifact writers | writer surface consolidated |
| 3 | `run_general_*` wrappers | vision runner surface adopted |
| 4 | `experiments/general/` facade | vision experiment modules adopted |
| 5 | `strategies/spectral/` facade | Graph-FL strategy modules adopted |
| 6 | `spectral_fl` package shim | `graphfl_lab` package surface adopted |
| 7 | legacy CLI choices and suite tokens | canonical CLI/config/result names adopted |
| 8 | cleanup status closure | release surface recorded |

Current validation 기준은 unit tests, CLI help/import checks, validation report command다.
