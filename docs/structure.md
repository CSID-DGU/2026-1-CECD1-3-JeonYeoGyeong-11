# Repository Structure

Source code is organized by responsibility, not experiment name. Use the narrowest module that owns the requested change.

## Change Routing

| Request | Primary edit location | Test neighborhood |
|---|---|---|
| Add/change `graph_source` | `spectral_fl/graph/signals/`, `spectral_fl/graph/sources/` | `tests/graph/` |
| Add/change `graph_mode` | `spectral_fl/graph/similarity/`, `spectral_fl/graph/sparsification.py`, `spectral_fl/graph/builders.py` | `tests/graph/` |
| Add graph diagnostics | `spectral_fl/graph/diagnostics.py` | `tests/graph/` |
| Add diagnostics artifact fields/writers | `spectral_fl/diagnostics/` | `tests/diagnostics/` |
| Add/change `GraphFLDesign` preset | `spectral_fl/designs/` | `tests/designs/` |
| Add lifecycle behavior | `spectral_fl/lifecycle/` | `tests/lifecycle/` |
| Change graph filtering math | `spectral_fl/strategies/graphfl/filtering.py` | `tests/strategies/graphfl/` |
| Change client/conflict weights | `spectral_fl/strategies/graphfl/aggregation.py` | `tests/strategies/graphfl/` |
| Add `aggregation_target` | `spectral_fl/strategies/graphfl/targets.py` | `tests/strategies/graphfl/` |
| Change server momentum/optimizer interaction | `spectral_fl/strategies/graphfl/momentum.py` | `tests/strategies/graphfl/` |
| Add baseline | `spectral_fl/strategies/baselines/` | `tests/strategies/baselines/` |
| Change vision single run | `spectral_fl/experiments/vision/single_run.py` | `tests/experiments/vision/` |
| Change vision suite orchestration | `spectral_fl/experiments/vision/suite.py` | `tests/experiments/vision/` |
| Change stress/client-count sweep | `spectral_fl/experiments/vision/stress_grid.py`, `spectral_fl/experiments/vision/client_count_sweep.py` | `tests/experiments/vision/` |
| Change Cora single run | `spectral_fl/experiments/cora/single_run.py` | `tests/experiments/cora/` |
| Change Cora graph ablation | `spectral_fl/experiments/cora/graph_ablation.py` | `tests/experiments/cora/` |
| Change suite variant grammar | `spectral_fl/experiments/suites/vision/variants.py` | `tests/experiments/vision/` |
| Change suite reporting | `spectral_fl/experiments/suites/vision/reporting.py` | `tests/experiments/vision/` |
| Change CLI arguments | `spectral_fl/cli/` | CLI help/import tests |
| Add/edit config | `configs/` | JSON validation |

## Source Layout

```text
spectral_fl/
  app/                       Flower app config/runtime glue
  cli/                       argparse only
  clients/                   Flower client implementations
  data/                      dataset loading and partitioning
  designs/                   GraphFLDesign composer/registry/presets
  diagnostics/               schemas, metrics, CSV/JSONL writers
  graph/                     client relation graph construction
    builders.py
    diagnostics.py
    method_specs.py
    registry.py
    signals/
    similarity/
    sources/
    sparsification.py
  lifecycle/                 contracts, traces, counterfactual runner
  strategies/
    baselines/
    graphfl/
      aggregation.py
      config.py
      diagnostics.py
      filtering.py
      momentum.py
      strategy.py
      targets.py
      tracing.py
    spectral/                compatibility wrappers
  experiments/
    vision/
    general/                 compatibility wrappers
    cora/
    suites/
      vision/
      general/               compatibility wrappers
```

## Compatibility Facades

Keep these thin. Add new logic in scoped modules, then re-export only if compatibility requires it.

```text
spectral_fl/aggregation.py
spectral_fl/client.py
spectral_fl/general_client.py
spectral_fl/general_data.py
spectral_fl/general_models.py
spectral_fl/general_suite_variants.py
spectral_fl/cli/general_experiment.py
spectral_fl/cli/general_suite.py
spectral_fl/cli/general_client_count_sweep.py
spectral_fl/cli/general_stress_grid.py
spectral_fl/model.py
spectral_fl/spectral_diagnostics.py
spectral_fl/strategy.py
spectral_fl/suite_stats.py
spectral_fl/update_graph.py
spectral_fl/strategies/spectral/
spectral_fl/experiments/general/
spectral_fl/experiments/suites/general/
spectral_fl/experiments/suite.py
spectral_fl/experiments/stress_grid.py
spectral_fl/experiments/client_count_sweep.py
spectral_fl/experiments/graph_ablation.py
scripts/analysis/deep_dive_general.py
scripts/analysis/merge_general_fedavg_ours.py
scripts/reports/plot_general_convergence.py
run_general_*.py
```

## Boundary Rules

| Area | Owns | Must not own |
|---|---|---|
| `spectral_fl/cli/` | argument parsing | datasets, models, strategies, graphs, reports |
| `spectral_fl/experiments/` | orchestration, subprocesses, metadata, output files | graph math, strategy internals |
| `spectral_fl/graph/` | relation graph construction | Flower strategies, experiment runners |
| `spectral_fl/strategies/graphfl/` | server-side aggregation behavior | config paths, suite output layouts |
| `spectral_fl/strategies/spectral/` | compatibility wrapper | new logic |
| `spectral_fl/strategies/baselines/` | baseline strategies, tracing helpers | graph builder internals |
| `spectral_fl/designs/` | method composition metadata | experiment execution |
| `spectral_fl/lifecycle/` | contracts, traces, state store, side-effect-free diagnostics | runtime graph builders |
| `spectral_fl/diagnostics/` | schemas, metrics, writers | suite-specific configs |
| `spectral_fl/app/` | Flower App glue | CLI parsing |

## Config Layout

```text
configs/
  cora/
    ablations/
      graph/
  vision/
    baselines/
    diagnostic/
    smoke/
    probes/
      frequency/
      graph_source/
      structure/
      tau/
    stress/
      client_count/
      fedavg_collapse/
    sweeps/
      client_count/
```

Generated results:

```text
experiments_current/
```

## Adding New Code

| New work | Location |
|---|---|
| graph signal | `graph/signals/`, then `graph/sources/` |
| graph similarity | `graph/similarity/`, then wire in `graph/builders.py` |
| sparsification rule | `graph/sparsification.py` |
| aggregation target | `strategies/graphfl/targets.py` |
| server optimizer interaction | `strategies/graphfl/momentum.py` |
| suite-consumed diagnostics | owner module emits; experiment reporting aggregates |
| experiment workflow | `experiments/vision/` or `experiments/cora/` |
| config | narrowest `configs/<track>/<question>/` folder |

## Guard Tests

```text
tests/structure/
```

Purpose:

```text
import boundaries
facade thinness
no logic drift into compatibility wrappers
```
