# Repository Structure

This repository intentionally uses a deep, responsibility-oriented tree.  The
goal is to make each change land in the smallest possible set of files, so both
humans and AI coding agents can infer where a modification belongs before
opening half the repository.

## Design Rule

Source code is organized by reason to change, not by experiment name.

- Code lives under responsibility boundaries: graph construction, strategy
  behavior, clients, models, data, experiment orchestration, and CLI parsing.
- Configs and generated outputs live under experiment-question boundaries:
  smoke, baselines, probes, stress, and sweeps.
- Compatibility files stay thin.  They can re-export old import paths, but new
  logic belongs in the scoped package.

## Change Routing

Use the narrowest module that matches the request.

| Request | Primary edit location | Test neighborhood |
|---|---|---|
| Add or change a `graph_source` | `spectral_fl/graph/signals/`, `spectral_fl/graph/sources/` | `tests/graph/` |
| Add or change a `graph_mode` | `spectral_fl/graph/similarity/`, `spectral_fl/graph/sparsification.py`, `spectral_fl/graph/builders.py` | `tests/graph/` |
| Add graph diagnostics | `spectral_fl/graph/diagnostics.py` | `tests/graph/` |
| Change spectral filtering math | `spectral_fl/strategies/spectral/filtering.py` | `tests/strategies/spectral/` or `tests/strategies/` |
| Change client weights or conflict weights | `spectral_fl/strategies/spectral/aggregation.py` | `tests/strategies/spectral/` or `tests/strategies/` |
| Add an `aggregation_target` | `spectral_fl/strategies/spectral/targets.py` | `tests/strategies/spectral/` or `tests/strategies/` |
| Change server momentum/optimizer interaction | `spectral_fl/strategies/spectral/momentum.py` | `tests/strategies/spectral/` or `tests/strategies/` |
| Add a baseline | `spectral_fl/strategies/baselines/` | `tests/strategies/baselines/` or `tests/strategies/` |
| Change one General FL run | `spectral_fl/experiments/general/single_run.py` | `tests/experiments/general/` |
| Change General suite orchestration | `spectral_fl/experiments/general/suite.py` | `tests/experiments/general/` |
| Change stress grid or client-count sweep behavior | `spectral_fl/experiments/general/stress_grid.py`, `spectral_fl/experiments/general/client_count_sweep.py` | `tests/experiments/general/` |
| Change Cora single-run behavior | `spectral_fl/experiments/cora/single_run.py` | `tests/experiments/cora/` |
| Change Cora graph-ablation orchestration | `spectral_fl/experiments/cora/graph_ablation.py` | `tests/experiments/cora/` |
| Change General suite variant grammar | `spectral_fl/experiments/suites/general/variants.py` | `tests/experiments/general/` |
| Change General suite reporting | `spectral_fl/experiments/suites/general/reporting.py` | `tests/experiments/general/` |
| Change CLI arguments | `spectral_fl/cli/` only | CLI help / import tests |
| Add or edit an experiment config | `configs/` | JSON validation |

## Source Layout

```text
spectral_fl/
  app/                       # Flower app config and runtime glue
  cli/                       # argparse only
  clients/                   # Flower client implementations
  data/                      # dataset loading and partitioning
  experiments/               # run orchestration, no algorithmic graph math
    general/
      single_run.py
      suite.py
      stress_grid.py
      client_count_sweep.py
    cora/
      single_run.py
      graph_ablation.py
    suites/
      stats.py
      general/
        variants.py
        reporting.py
  graph/                     # client relation graph construction
    builders.py              # assembles signal, similarity, sparsification
    diagnostics.py
    signals/                 # node feature extraction
    similarity/              # pairwise relation scores
    sources/                 # graph_source option selection/config
    sparsification.py        # dense/knn/random/threshold/uniform rules
  models/
  strategies/
    baselines/
    spectral/
      aggregation.py
      config.py
      diagnostics.py
      filtering.py
      momentum.py
      strategy.py
      targets.py
      tracing.py
```

## Compatibility Facades

These files are intentionally thin and should remain re-export wrappers.

```text
spectral_fl/aggregation.py
spectral_fl/client.py
spectral_fl/general_client.py
spectral_fl/general_data.py
spectral_fl/general_models.py
spectral_fl/general_suite_variants.py
spectral_fl/model.py
spectral_fl/spectral_diagnostics.py
spectral_fl/strategy.py
spectral_fl/suite_stats.py
spectral_fl/update_graph.py
spectral_fl/experiments/suite.py
spectral_fl/experiments/stress_grid.py
spectral_fl/experiments/client_count_sweep.py
spectral_fl/experiments/graph_ablation.py
run_*.py
```

Do not add new algorithmic logic to these files.  Add the logic in the scoped
module, then re-export from the facade only when backward compatibility needs it.

## Boundary Rules

- `spectral_fl/cli/` parses arguments and calls experiment functions.  It should
  not construct datasets, models, clients, strategies, graphs, or reports.
- `spectral_fl/experiments/` orchestrates runs, subprocesses, metadata, and
  output files.  It should not implement graph math or strategy internals.
- `spectral_fl/graph/` builds client relation graphs.  It should not import
  Flower strategies or experiment runners.
- `spectral_fl/strategies/spectral/` owns server-side aggregation behavior.  It
  should not know about config file paths or suite output layouts.
- `spectral_fl/strategies/baselines/` owns baseline strategy implementations and
  shared tracing helpers.
- `spectral_fl/app/` is Flower App glue.  Keep app config/default wiring there,
  not in CLI files.

## Config Layout

Config files are organized by experiment question, because configs are the
research-facing layer.

```text
configs/
  cora/
    ablations/
      graph/
  general/
    baselines/
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

Generated results stay under `experiments_current/` and are not tracked.  New
result directories should mirror the config question when practical, but old
generated outputs do not need to be moved.

## Adding New Code

- New graph signal: add extraction logic under `graph/signals/`, then register
  option handling under `graph/sources/`.
- New graph similarity: add a module under `graph/similarity/`; only touch
  `graph/builders.py` to wire the mode.
- New sparsification rule: use `graph/sparsification.py`.
- New aggregation target: add it to `strategies/spectral/targets.py`.
- New server optimizer interaction: use `strategies/spectral/momentum.py`.
- New diagnostics consumed by suites: emit them from the strategy/graph module
  that owns the measurement, then aggregate them in experiment reporting.
- New experiment workflow: add orchestration under `experiments/general/` or
  `experiments/cora/` and keep the matching CLI file parser-only.
- New config: add it under the narrowest `configs/<track>/<question>/` folder.

## Guard Tests

Structural tests live under `tests/structure/`.  They check import boundaries
and facade thinness so future AI-assisted edits do not quietly collapse the
tree back into a few large files.
