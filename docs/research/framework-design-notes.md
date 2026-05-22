# Framework Design Notes

Active design note for the Graph-FL Design Lab implementation shape.

## Core Direction

Graph-FL methods are component compositions:

```text
client state extraction
relation estimation
topology construction
aggregation target
delivery/personalization semantics
local objective hooks
state carried across rounds
diagnostics and controls
```

## Canonical Paths

| Responsibility | Canonical path |
|---|---|
| Method metadata | `graphfl_lab/designs/` |
| Graph source/signal extraction | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| Builders and registry | `graphfl_lab/graph/builders.py`, `graphfl_lab/graph/registry.py` |
| Controls/clustering | `graphfl_lab/graph/controls.py`, `graphfl_lab/graph/clustering.py` |
| Graph-FL runtime | `graphfl_lab/strategies/graphfl/` |
| Baselines | `graphfl_lab/strategies/baselines/` |
| Lifecycle/counterfactuals | `graphfl_lab/lifecycle/` |
| Metrics/writers | `graphfl_lab/diagnostics/` |
| Vision orchestration | `graphfl_lab/experiments/vision/` |
| Vision suite/reporting | `graphfl_lab/experiments/suites/vision/` |
| Configs | `configs/vision/` |

Compatibility paths:

```text
run_general_*.py
configs/general/...
result_general_*
general_suite_summary.*
graphfl_lab/strategies/spectral/
```

## Runtime Flow

```mermaid
flowchart TD
    CFG[config or CLI args] --> DESIGN[GraphFLDesign or explicit knobs]
    DESIGN --> SRC[graph_source]
    SRC --> BUILDER[graph_mode / graph builder]
    BUILDER --> CORR[correction family]
    CORR --> TARGET[aggregation_target]
    TARGET --> STRAT[GraphFLDiagnosticStrategy]
    STRAT --> DIAG[round/client diagnostics]
    DIAG --> REPORT[suite rows, summary, plots]
```

Ownership:

| Logic | Location |
|---|---|
| relation/topology | `graph/` |
| aggregation object selection | `strategies/graphfl/targets.py` |
| artifact fields | `diagnostics/` and suite reporting |
| orchestration | `experiments/vision/` |

## Naming Policy

| Use | Name |
|---|---|
| strategy package | `graphfl_lab.strategies.graphfl` |
| runtime class | `GraphFLDiagnosticStrategy` |
| aggregation targets | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| filter key | `graph_filter_strength` |
| suite family | `ours_graph_filtered_*` |
| filter-only suffix | `_graph_filter_only` |

Removed public debt (Gate 6 + Phase 2 on `main`):

```text
spectral_fl package shim
graphfl_lab.strategies.spectral facades
run_general_* / general_* artifact readers
```

Remaining read-only compatibility (see docs/removed-materials.md):

```text
spectral_filtered_* input aliases
spectral_filter_strength JSON key alias
ours_spectral_filtered_* reporting result tags
configs/general/ path alias
```

## Experiment Philosophy

Mechanism questions:

```text
real graph vs matched random/shuffled/identity/uniform
clustering-only sufficiency
graph-free norm/cap/reweight sufficiency
measurable update/weight perturbation
effective clients, entropy, non-dominance
```

## Engineering Rules

```text
CLI modules parser-only
experiment modules orchestration-only
graph construction independent of Flower strategies
graphfl strategy package remains the single runtime surface
new component tests cover shape, determinism, metadata, compatibility aliases
```

Checks:

```text
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
```

## Known Debt

Optional breaking removals only (policy decision). Active code already prefers
canonical names.

| Debt | Reason |
|---|---|
| `spectral_filtered_*` / `spectral_filter_strength` input aliases | old JSON configs |
| `ours_spectral_filtered_*` reporting pairs | old result tags in CSVs |
| `configs/general/...` path alias | old user command paths |

Tracking:

```text
docs/framework/cleanup-plan.md
docs/framework/naming-and-compatibility.md
```
