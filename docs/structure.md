# Repository Structure

## 목적

Code, script, config, test의 위치를 빠르게 찾기 위한 repository map이다.

## Change Routing

| 변경 요청 | 주요 위치 | Test 위치 |
|---|---|---|
| `graph_source` 추가/수정 | `graphfl_lab/graph/signals/`, `graphfl_lab/graph/sources/` | `tests/graph/` |
| `graph_mode` 추가/수정 | `graphfl_lab/graph/similarity/`, `graphfl_lab/graph/sparsification.py`, `graphfl_lab/graph/builders.py` | `tests/graph/` |
| graph diagnostics | `graphfl_lab/graph/diagnostics.py` | `tests/graph/` |
| diagnostics artifact | `graphfl_lab/diagnostics/` | `tests/diagnostics/` |
| `GraphFLDesign` preset | `graphfl_lab/designs/` | `tests/designs/` |
| lifecycle behavior | `graphfl_lab/lifecycle/` | `tests/lifecycle/` |
| graph filtering math | `graphfl_lab/strategies/graphfl/filtering.py` | `tests/strategies/graphfl/` |
| aggregation weights | `graphfl_lab/strategies/graphfl/aggregation.py` | `tests/strategies/graphfl/` |
| `aggregation_target` | `graphfl_lab/strategies/graphfl/targets.py` | `tests/strategies/graphfl/` |
| baseline strategy | `graphfl_lab/strategies/baselines/` | `tests/strategies/baselines/` |
| vision single run | `graphfl_lab/experiments/vision/single_run.py` | `tests/experiments/vision/` |
| vision suite | `graphfl_lab/experiments/vision/suite.py` | `tests/experiments/vision/` |
| Cora run | `graphfl_lab/experiments/cora/` | `tests/experiments/cora/` |
| suite reporting | `graphfl_lab/experiments/suites/vision/` | `tests/experiments/suites/vision/` |
| CLI argument | `graphfl_lab/cli/` | CLI help/import tests |
| config | `configs/` | JSON validation |
| evidence pack | `graphfl_lab/validation/`, `scripts/validation/` | `tests/validation/` |

## Source Layout

```text
graphfl_lab/
├── app/
├── cli/
├── clients/
├── data/
├── designs/
├── diagnostics/
├── graph/
├── lifecycle/
├── models/
├── strategies/
└── experiments/
```

## Script Layout

| Folder | 역할 |
|---|---|
| `scripts/checks/` | preflight, evidence bundle, parity check |
| `scripts/reports/` | plot and dashboard report |
| `scripts/analysis/` | deep dive and result merge |
| `scripts/validation/` | Graph-FL evidence pack |
| `scripts/smoke/` | smoke checks |
| `scripts/dev/` | maintenance helper |

## Documentation Layout

| 문서 | 역할 |
|---|---|
| `docs/framework/overview.md` | claim and experiment design |
| `docs/framework/metrics.md` | metric reference |
| `docs/framework/components.md` | component and extension map |
| `docs/framework/evidence.md` | framework-quality evidence |
| `docs/maintenance/migration-and-compatibility.md` | repository contract |
| `docs/archive/README.md` | archive map |
