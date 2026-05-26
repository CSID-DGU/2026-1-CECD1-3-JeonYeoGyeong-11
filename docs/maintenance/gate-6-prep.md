# Gate 6 Prep

## 목적

Gate 6 hard cleanup의 entry criteria, removal order, verification command를 기록한다.

## Entry Criteria

| Criteria | Status |
|---|---|
| `graphfl_lab` package move | met |
| modularization | met |
| public surface alignment | met |
| CI green on main | met |
| `pre-graphfl-rename` tag | met |

## Removal Order

| Step | Surface | Status |
|---:|---|---|
| 1 | serialized asset check | done |
| 2 | duplicate artifact writers | done |
| 3 | `run_general_*` wrappers | done |
| 4 | `experiments/general/` facade | done |
| 5 | `strategies/spectral/` facade | done |
| 6 | `spectral_fl` package shim | done |
| 7 | legacy CLI choices and suite tokens | done |
| 8 | cleanup status closure | done |

## 검증

```text
python scripts/dev/run.py gate-check 6
python scripts/dev/run.py gate-check 5d-prep
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
```

Canonical:

- `docs/maintenance/migration-and-compatibility.md`
