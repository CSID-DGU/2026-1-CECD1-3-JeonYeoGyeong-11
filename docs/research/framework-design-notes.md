# Framework Design Notes

## Core Direction

Graph-FL method를 component composition으로 표현한다.

```text
client state extraction
-> relation estimation
-> topology construction
-> aggregation target
-> delivery / personalization
-> local objective hook
-> state store
-> diagnostics and controls
```

## Canonical Path

| Responsibility | Path |
|---|---|
| method metadata | `graphfl_lab/designs/` |
| graph source/signal | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| graph builder/registry | `graphfl_lab/graph/builders.py`, `graphfl_lab/graph/registry.py` |
| controls/clustering | `graphfl_lab/graph/controls.py`, `graphfl_lab/graph/clustering.py` |
| Graph-FL runtime | `graphfl_lab/strategies/graphfl/` |
| baselines | `graphfl_lab/strategies/baselines/` |
| lifecycle | `graphfl_lab/lifecycle/` |
| diagnostics | `graphfl_lab/diagnostics/` |
| vision orchestration | `graphfl_lab/experiments/vision/` |
| vision suite/reporting | `graphfl_lab/experiments/suites/vision/` |
| configs | `configs/vision/`, `configs/cora/` |

## Compatibility

| Legacy | Current |
|---|---|
| `configs/general/...` | `configs/vision/...` |
| `spectral_filter_strength` | `graph_filter_strength` |
| `spectral_filtered_*` | `graph_filtered_*` input alias |
| `ours_spectral_filtered_*` | historical reporting tag |

## Design Rule

| Rule | 기준 |
|---|---|
| source reuse | client representation이 같으면 기존 `graph_source` 재사용 |
| builder reuse | relation/topology가 같으면 기존 `graph_mode` 재사용 |
| target reuse | graph application 위치가 같으면 기존 `aggregation_target` 재사용 |
| diagnostics | 새 mechanism은 trace와 artifact field를 함께 추가 |
