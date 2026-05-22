# Repository Structure

Source code is organized by responsibility, not experiment name. Use the narrowest module that owns the requested change.

## Change Routing

| Request | Primary edit location | Test neighborhood |
|---|---|---|
| Add/change `graph_source` | `graphfl_lab/graph/signals/`, `graphfl_lab/graph/sources/` | `tests/graph/` |
| Add/change `graph_mode` | `graphfl_lab/graph/similarity/`, `graphfl_lab/graph/sparsification.py`, `graphfl_lab/graph/builders.py` | `tests/graph/` |
| Add graph diagnostics | `graphfl_lab/graph/diagnostics.py` | `tests/graph/` |
| Add diagnostics artifact fields/writers | `graphfl_lab/diagnostics/` | `tests/diagnostics/` |
| Add/change `GraphFLDesign` preset | `graphfl_lab/designs/` | `tests/designs/` |
| Add lifecycle behavior | `graphfl_lab/lifecycle/` | `tests/lifecycle/` |
| Change graph filtering math | `graphfl_lab/strategies/graphfl/filtering.py` | `tests/strategies/graphfl/` |
| Change client/conflict weights | `graphfl_lab/strategies/graphfl/aggregation.py` | `tests/strategies/graphfl/` |
| Add `aggregation_target` | `graphfl_lab/strategies/graphfl/targets.py` | `tests/strategies/graphfl/` |
| Change server momentum/optimizer interaction | `graphfl_lab/strategies/graphfl/momentum.py` | `tests/strategies/graphfl/` |
| Add baseline | `graphfl_lab/strategies/baselines/` | `tests/strategies/baselines/` |
| Change vision single run | `graphfl_lab/experiments/vision/single_run.py` | `tests/experiments/vision/` |
| Change vision suite orchestration | `graphfl_lab/experiments/vision/suite.py` | `tests/experiments/vision/` |
| Change stress/client-count sweep | `graphfl_lab/experiments/vision/stress_grid.py`, `graphfl_lab/experiments/vision/client_count_sweep.py` | `tests/experiments/vision/` |
| Change Cora single run | `graphfl_lab/experiments/cora/single_run.py` | `tests/experiments/cora/` |
| Change Cora graph ablation | `graphfl_lab/experiments/cora/graph_ablation.py` | `tests/experiments/cora/` |
| Change suite variant grammar | `graphfl_lab/experiments/suites/vision/variants.py` | `tests/experiments/vision/` |
| Change suite reporting | `graphfl_lab/experiments/suites/vision/reporting.py` | `tests/experiments/vision/` |
| Change CLI arguments | `graphfl_lab/cli/` | CLI help/import tests |
| Add/edit config | `configs/` | JSON validation |

## Source Layout

```text
graphfl_lab/
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
spectral_fl/__init__.py
graphfl_lab/aggregation.py
graphfl_lab/client.py
graphfl_lab/general_client.py
graphfl_lab/general_data.py
graphfl_lab/general_models.py
graphfl_lab/general_suite_variants.py
graphfl_lab/cli/general_experiment.py
graphfl_lab/cli/general_suite.py
graphfl_lab/cli/general_client_count_sweep.py
graphfl_lab/cli/general_stress_grid.py
graphfl_lab/model.py
graphfl_lab/spectral_diagnostics.py
graphfl_lab/strategy.py
graphfl_lab/suite_stats.py
graphfl_lab/update_graph.py
graphfl_lab/strategies/spectral/
graphfl_lab/experiments/general/
graphfl_lab/experiments/suites/general/
graphfl_lab/experiments/suite.py
graphfl_lab/experiments/stress_grid.py
graphfl_lab/experiments/client_count_sweep.py
graphfl_lab/experiments/graph_ablation.py
scripts/analysis/deep_dive_general.py
scripts/analysis/merge_general_fedavg_ours.py
scripts/reports/plot_general_convergence.py
run_general_*.py
```

## Boundary Rules

| Area | Owns | Must not own |
|---|---|---|
| `graphfl_lab/cli/` | argument parsing | datasets, models, strategies, graphs, reports |
| `graphfl_lab/experiments/` | orchestration, subprocesses, metadata, output files | graph math, strategy internals |
| `graphfl_lab/graph/` | relation graph construction | Flower strategies, experiment runners |
| `graphfl_lab/strategies/graphfl/` | server-side aggregation behavior | config paths, suite output layouts |
| `graphfl_lab/strategies/spectral/` | compatibility wrapper | new logic |
| `graphfl_lab/strategies/baselines/` | baseline strategies, tracing helpers | graph builder internals |
| `graphfl_lab/designs/` | method composition metadata | experiment execution |
| `graphfl_lab/lifecycle/` | contracts, traces, state store, side-effect-free diagnostics | runtime graph builders |
| `graphfl_lab/diagnostics/` | schemas, metrics, writers | suite-specific configs |
| `graphfl_lab/app/` | Flower App glue | CLI parsing |

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
