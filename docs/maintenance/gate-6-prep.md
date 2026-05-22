# Gate 6 Prep — Hard Cleanup Checklist

Gate 6 removes compatibility surfaces listed in `docs/framework/naming-and-compatibility.md`.
Do not start removal until the entry criteria in `docs/maintenance/cleanup-status.md`
are satisfied.

## Entry Criteria (pragmatic)

Required (already met on `main` as of 2026-05-22):

```text
Gate 4c: one nightly green on main (see docs/maintenance/last_nightly_run.json)
Gate 3–5: graphfl_lab move, modularization, C5 public-surface alignment
Remote CI green on main after merge
Tag pre-graphfl-rename on origin
```

Recommended before deleting shims (pick one, not all seven calendar days):

```text
Re-run nightly or CI after the last Gate 6-prep doc/code change and confirm green
OR wait for 2–3 scheduled nightly successes on main if the repo is idle
```

Not required for this project pace:

```text
Seven consecutive nightly greens — conservative default for long-running teams;
replace with the recommended check above unless you explicitly want a week-long watch period
```

## Removal Order (when entry criteria are met)

1. [x] Confirm serialized assets: `python scripts/dev/migrate_serialized_objects.py` (tracked Cora cache `.pt` only; no `spectral_fl` pickle paths).
2. [x] Remove compatibility **writers** of duplicate artifacts (`general_suite_*`, `result_general_*` mirrors); readers still accept legacy paths.
3. [x] Remove `run_general_*` root wrappers and `plot_general_*` / `merge_general_*` / `deep_dive_general` script wrappers.
4. [x] Remove `graphfl_lab/experiments/general/` and `graphfl_lab/experiments/suites/general/` import facades.
5. [x] Remove `graphfl_lab/strategies/spectral/` wrappers (keep real spectral math names in operators).
6. [x] Remove `spectral_fl` package shim last, after grep shows no remaining imports outside tests explicitly checking deprecation.
7. Remove legacy CLI choices (`spectral_filtered_*` inputs) and old suite token spellings only after suite/history policy is frozen.
8. Update `docs/maintenance/cleanup-status.md` to `closed` and link from `docs/removed-materials.md`.

## Verify After Each Batch

```text
python scripts/dev/run.py gate-check 5d-prep
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
```

Optional remote check:

```text
gh workflow run nightly.yml --ref main
```
