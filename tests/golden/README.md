# Golden Baseline Policy

## 목적

`tests/golden/`은 smoke output의 normalized baseline을 보관한다.
Refactor 이후 output schema와 주요 값이 유지되는지 비교하는 기준이다.

## Update Policy

Golden baseline 변경은 별도 PR에서 수행한다. PR에는 변경 이유와 영향 범위를 기록한다.

## Normalized Volatile Field

Golden 비교에서 아래 field는 run 환경에 따라 달라질 수 있어 제거한다.

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

Schema 비교는 exact match를 사용한다. Value 비교만 volatile field 제거 후 수행한다.

Canonical:

- `docs/maintenance/migration-and-compatibility.md`
