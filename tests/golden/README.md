# Golden Baseline Policy

`tests/golden/`은 normalized smoke output baseline을 보관한다. Baseline은 refactor 이후 output schema와 중요한 value가 유지되는지 확인하는 contract다.

## Update Policy

| Change Type | Required Record |
|---|---|
| schema change | changed field, reason, affected fixture |
| value expectation change | metric/result reason, affected fixture |
| volatile-field change | added/removed field and comparison effect |

Golden baseline 변경은 focused change로 다룬다. 변경 기록에는 baseline이 이동한 이유와 behavior 또는 schema 변화를 함께 남긴다.

## Normalized Volatile Fields

아래 fields는 runtime environment에 따라 달라질 수 있으므로 value comparison에서 제거한다.

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

Schema comparison은 exact match다. Value comparison은 위 volatile fields만 제외한다. Canonical policy는 `docs/maintenance.md`에서 관리한다.
