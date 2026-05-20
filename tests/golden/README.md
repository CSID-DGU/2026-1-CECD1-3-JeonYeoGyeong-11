# Golden Baseline Policy

Gate 4c captures smoke outputs under this directory. Gate 5 compares normalized
outputs against those baselines to prove behavior-preserving refactors.

Golden updates are allowed only in a separate PR with a clear reason and impact
scope.

## Volatile Fields

Normalized golden comparison removes fields that are expected to vary across
runs or machines:

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

Schema comparison is exact. Only normalized value comparison removes volatile
fields.
