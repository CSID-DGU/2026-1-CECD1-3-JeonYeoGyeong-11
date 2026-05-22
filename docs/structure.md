# Repository Structure

Source code is organized by responsibility, not experiment name. Use the narrowest
module that owns the requested change.

The root [README.md](../README.md) shows an **abbreviated** layout tree. This file
is the **detailed** map: package modules, scripts, tests, configs, and which
capabilities remain after Gate 6 (see [removed-materials.md](removed-materials.md)).

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
| Change suite artifact discovery | `graphfl_lab/experiments/suites/vision/artifacts.py` | `tests/experiments/suites/vision/` |
| Change CLI arguments | `graphfl_lab/cli/` | CLI help/import tests |
| Add/edit config | `configs/` | JSON validation |

## Source Layout

```text
graphfl_lab/
├── app/                              Flower App config and runtime glue
├── cli/                              argparse only (vision, cora, suite, sweep)
├── clients/                          Flower client train/eval implementations
├── data/                             dataset loading and Dirichlet partition
├── designs/                          GraphFLDesign composer, registry, presets
├── diagnostics/                      result schemas, metrics, evidence bundle
├── graph/                            client relation graph construction
│   ├── builders.py                   register builders; compose relation + topology
│   ├── registry.py                   graph_mode lookup and plugin dispatch
│   ├── method_specs.py               runnable graph_method profiles
│   ├── presets.py                    named graph presets
│   ├── signals/                      raw client signals (update, head, …)
│   ├── similarity/                   cosine, magnitude, RBF helpers
│   ├── sources/                      graph_source adapters (graphfl, fedsim, …)
│   └── sparsification.py             kNN and sparsification utilities
├── lifecycle/                        contracts, traces, counterfactual runner
├── models/                           CNN/MLP and Cora model wrappers
├── strategies/
│   ├── graphfl/                      Graph-FL server strategy
│   │   ├── strategy.py               round loop and orchestration
│   │   ├── filtering.py              Laplacian / graph filter on client matrices
│   │   ├── aggregation.py            weighted combine and dominance hooks
│   │   ├── targets.py                aggregation_target dispatch
│   │   ├── diagnostics.py            per-round graph/control metrics
│   │   ├── tracing.py                round trace hooks
│   │   ├── momentum.py               server optimizer / FedAvgM interaction
│   │   ├── config_context.py         per-round config snapshot for filtering
│   │   └── …                         round_*, ema, projection, artifact_rows, …
│   └── baselines/                    fedavgm, fedsim, graph_smooth, dominance_aware, …
├── flower_app.py                     Flower App entry
├── flower_runner.py                  subprocess / engine dispatch for experiments
├── config_io.py                      JSON load; legacy key aliases
└── experiments/
    ├── vision/
    │   ├── single_run.py             one vision experiment + result JSON
    │   ├── suite.py                  multi-variant suite orchestration
    │   ├── stress_grid.py            stress-grid runner
    │   └── client_count_sweep.py     client-count sweep runner
    ├── cora/
    │   ├── single_run.py             Cora single run
    │   └── graph_ablation.py         graph ablation orchestration
    └── suites/vision/
        ├── variants.py               suite token grammar entry
        ├── variant_core.py           core variant parsing
        ├── variant_targets.py        graph_filtered target tokens
        ├── variant_sources.py        graph_source tokens
        ├── variant_families.py       family grouping for reporting
        ├── variant_commands.py       CLI flag emission per variant
        ├── variant_suffixes.py       _graph_filter_only and legacy suffixes
        ├── variant_diagnostics.py    diagnostic variant tokens
        ├── variant_legacy.py         historical token spellings (read/parse)
        ├── artifacts.py              result/suite path discovery
        ├── reporting.py              summaries, dashboard, interpretation
        ├── summary.py                suite summary row aggregation
        ├── features.py               suite feature columns for CSV/summary
        ├── metadata.py               suite-level metadata helpers
        └── variant_helpers.py        per-variant result path resolution
```

## Scripts Layout

```text
scripts/
├── checks/
│   ├── diagnostic_suite_preflight.py   suite variant sanity before long runs
│   ├── result_evidence_bundle.py       validate single-run result JSON bundle
│   └── prior_work_proxy_parity.py      prior-work proxy summary parity check
├── smoke/
│   └── prior_work_proxy.py             short prior-work proxy smoke workflow
├── reports/
│   ├── plot_vision_convergence.py      suite convergence plots (canonical)
│   ├── generate_dashboard_mockup.py    dashboard mockup from suite dir
│   ├── generate_diagnostic_plots.py    diagnostic plot helpers
│   ├── graph_label_alignment_report.py label–graph alignment report
│   └── spectral_decomposition_report.py  spectral decomposition analysis (math)
├── analysis/
│   ├── deep_dive_vision.py             per-variant deep dive (canonical)
│   ├── merge_vision_fedavg_ours.py     merge fedavg/ours tables
│   ├── deep_dive_seed.py               seed-level deep dive helper
│   └── aggregate_graph_ablation.py     Cora graph ablation aggregation
├── dev/
│   ├── run.py                          gate-check and dev orchestration
│   ├── golden.py                       golden / regression helpers
│   └── migrate_serialized_objects.py   pickle module-path migration utility
├── util/
│   └── print_round_table.py            print round table from result JSON
└── archive/legacy-analysis/            frozen phase-1/2/3 analysis scripts (historical)
```

Removed script names (Gate 6): `plot_general_convergence.py`, `deep_dive_general.py`,
`merge_general_fedavg_ours.py` — use vision-named scripts above.

## Tests Layout

```text
tests/
├── structure/                        import boundaries, facade thinness
├── graph/                            graph builders, sources, registry
├── strategies/graphfl/               filtering, targets, rounds, diagnostics
├── strategies/baselines/             baseline strategies
├── lifecycle/                        aggregation, counterfactuals, contracts
├── designs/                          GraphFLDesign presets
├── diagnostics/                      result schema and evidence
├── experiments/vision/               suite variants, single-run helpers
├── experiments/cora/                 Cora experiment helpers
├── experiments/suites/vision/        artifact discovery, reporting
├── cli/                              CLI parsing and choices
├── clients/                          Flower client behavior
├── core/                             config_io, package imports, runner
├── scripts/                          report script smoke tests
├── dev/                              gate-check entrypoint tests
└── golden/                           golden-file comparisons
```

## Root Launchers

```text
run_vision_experiment.py              vision single run
run_vision_suite.py                   vision suite
run_vision_client_count_sweep.py      client-count sweep
run_vision_stress_grid.py             stress grid
run_graph_ablation.py                 Cora graph ablation
run_experiment.py                     unified dispatcher (track argument)
```

## Capability Checklist (post-Gate-6)

All rows are **present** on `main` unless marked removed. Verification:
`python -m unittest discover -s tests`, `diagnostic_suite_preflight.py`.

| Capability | Entry / module | Status |
|---|---|---|
| Vision single run | `run_vision_experiment.py` → `experiments/vision/single_run.py` | active |
| Vision suite | `run_vision_suite.py` → `experiments/vision/suite.py` | active |
| Stress grid | `run_vision_stress_grid.py` | active |
| Client-count sweep | `run_vision_client_count_sweep.py` | active |
| Cora graph ablation | `run_graph_ablation.py` | active |
| Graph-FL strategy | `strategies/graphfl/strategy.py` | active |
| Baselines (FedAvgM, FedSim, …) | `strategies/baselines/` | active |
| Graph builders / sources | `graph/builders.py`, `graph/sources/` | active |
| Aggregation targets `graph_filtered_*` | `strategies/graphfl/targets.py` | active |
| Legacy target JSON alias `spectral_filtered_*` | `canonical_aggregation_target()` | read-only input |
| Diagnostic suite preflight | `scripts/checks/diagnostic_suite_preflight.py` | active |
| Suite reporting / artifacts | `suites/vision/reporting.py`, `artifacts.py` | active |
| Convergence plots | `scripts/reports/plot_vision_convergence.py` | active |
| `run_general_*` launchers | — | **removed** |
| `result_general_*` / `general_suite_*` readers | — | **removed** |
| `spectral_fl` import shim | — | **removed** |
| Suite launch `ours_spectral_filtered_*` | — | **removed** (reporting tag pairing retained) |

## Compatibility Facades

Keep these thin. Add new logic in scoped modules, then re-export only if a stable import path requires it.

```text
graphfl_lab/aggregation.py            re-export lifecycle aggregation helpers
graphfl_lab/client.py                 re-export Flower client entry
graphfl_lab/model.py                  re-export model factory
graphfl_lab/strategy.py               re-export strategy factory
graphfl_lab/suite_stats.py            re-export suite statistics helpers
graphfl_lab/update_graph.py           re-export graph update utilities
graphfl_lab/experiments/suite.py      re-export vision suite runner
graphfl_lab/experiments/stress_grid.py re-export stress grid runner
graphfl_lab/experiments/client_count_sweep.py re-export client-count sweep
graphfl_lab/experiments/graph_ablation.py re-export Cora ablation runner
run_vision_*.py                       root thin launchers (canonical public CLI)
```

## Boundary Rules

| Area | Owns | Must not own |
|---|---|---|
| `graphfl_lab/cli/` | argument parsing | datasets, models, strategies, graphs, reports |
| `graphfl_lab/experiments/` | orchestration, subprocesses, metadata, output files | graph math, strategy internals |
| `graphfl_lab/graph/` | relation graph construction | Flower strategies, experiment runners |
| `graphfl_lab/strategies/graphfl/` | server-side aggregation behavior | config paths, suite output layouts |
| `graphfl_lab/strategies/baselines/` | baseline strategies, tracing helpers | graph builder internals |
| `graphfl_lab/designs/` | method composition metadata | experiment execution |
| `graphfl_lab/lifecycle/` | contracts, traces, state store, side-effect-free diagnostics | runtime graph builders |
| `graphfl_lab/diagnostics/` | schemas, metrics, writers | suite-specific configs |
| `graphfl_lab/app/` | Flower App glue | CLI parsing |

## Config Layout

```text
configs/
├── vision/                           Fashion-MNIST / vision track
│   ├── baselines/                    FedAvg, FedOpt, dominance baselines
│   ├── diagnostic/                   mechanism and diagnostic suites
│   ├── smoke/                        shortest runnable configs
│   ├── probes/                       single-knob attribution probes
│   │   ├── frequency/                warmup / frequency ablations
│   │   ├── graph_source/             client-state source comparisons
│   │   ├── structure/                topology / relation structure
│   │   └── tau/                      tau and filter-strength sweeps
│   ├── stress/                       Non-IID stress and collapse checks
│   │   ├── client_count/             scaling num_clients
│   │   └── fedavg_collapse/          FedAvg failure modes
│   └── sweeps/                       multi-config sweep grids
│       └── client_count/
└── cora/                             Cora / FGL track
    └── ablations/
        └── graph/                    graph construction ablation smoke/full
```

Generated results (gitignored output root, name from `--out-dir` / config):

```text
<output-dir>/                         per-run JSON, suite summaries, plots
```

## Suite Output Artifacts

| Artifact | Role |
|---|---|
| `vision_suite_summary.json` / `.csv` / `.md` | canonical suite summary |
| `vision_suite_rows.json` | per-run suite rows |
| `result_vision_<method>_seed<seed>_<tag>.json` | canonical single-run result |
| `suite_summary.*`, `suite_rows.json` | short read-only aliases when present |

Use `write_suite_summary_artifacts()` in `reporting.py` for suite-level files.
Use `resolve_result_path_for_variant()` or `discover_result_json_paths()` when
reading prior runs. Do not read `general_*` or `result_general_*` paths.

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
