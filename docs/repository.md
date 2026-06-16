# Repository Layout

이 문서는 code, config, script, test, docs 위치를 빠르게 찾기 위한 map이다. 생성 output과 개인 실행 환경은 포함하지 않는다.

## Top-Level Map

```text
.
├── README.md                         프로젝트 목적, 실행, 검증 요약
├── MIGRATION.md                      이름 변경과 compatibility 요약
├── CHANGELOG.md                      변경 기록
├── pyproject.toml                    package metadata, Flower app config
├── requirements.txt                  runtime/test dependencies
├── run_experiment.py                 unified runner: --track vision|cora
├── run_vision_experiment.py          vision single-run wrapper
├── run_vision_suite.py               vision suite wrapper
├── run_vision_stress_grid.py         vision stress-grid wrapper
├── run_vision_client_count_sweep.py  vision client-count sweep wrapper
├── run_graph_ablation.py             Cora graph ablation wrapper
├── graphfl_lab/                      canonical Python package
├── configs/                          tracked experiment config presets
├── scripts/                          validation, checks, reports, smoke helpers
├── tests/                            unit and contract tests
└── docs/                             세부 문서와 보조 demo
```

## Package Layout

```text
graphfl_lab/
├── app/                              Flower app config and runtime setup
├── cli/                              command-line dispatch and wrappers
├── clients/                          Flower client implementations
├── corrections/                      correction and control helpers
├── data/                             dataset loading and partition logic
├── designs/                          GraphFLDesign registry and presets
├── diagnostics/                      result schema and artifact writers
├── extensions/                       extension API, runtime preparation, validation
├── graph/                            graph source, builder, control, diagnostics
├── lifecycle/                        lifecycle contracts, context, traces
├── models/                           model definitions
├── presentation/                     demo에서 쓰는 capability manifest
├── strategies/
│   ├── baselines/                    FedAvg-like and graph-free baselines
│   └── graphfl/                      Graph-FL runtime strategy modules
├── experiments/
│   ├── cora/                         Cora single run and graph ablation
│   ├── suites/
│   │   └── vision/                   vision suite features, variants, reporting
│   └── vision/                       vision single run, suite, stress, sweeps
└── validation/                       검증 report 생성 로직
```

## Graph-FL Strategy Layout

```text
graphfl_lab/strategies/graphfl/
├── strategy.py                       orchestration entry point
├── config.py                         strategy config defaults
├── config_context.py                 round config projection
├── fit_results.py                    client fit result collection
├── update_space.py                   local update arrays and norms
├── projection.py                     cached projection helpers
├── round_context.py                  round-level context helpers
├── round_graph.py                    graph construction for a round
├── graph_state.py                    previous/current graph selection
├── round_weights.py                  aggregation and correction weights
├── aggregation.py                    graph-filtered aggregation math
├── filtering.py                      graph filtering operator
├── targets.py                        `aggregation_target` dispatch
├── client_metrics.py                 per-client metric extraction
├── conflict_metrics.py               conflict and dominance metrics
├── spectral_metrics.py               spectral/graph diagnostic metrics
├── diagnostic_targets.py             diagnostic target flattening
├── diagnostic_artifacts.py           diagnostic artifact writing
├── counterfactual_artifacts.py       real/control shadow rows
├── artifact_rows.py                  row builders for artifacts
├── round_outputs.py                  strategy output assembly
├── round_projection.py               graph-space projection assembly
├── trace_context.py                  run/round trace values
├── tracing.py                        trace helpers
├── ema.py                            EMA update state
├── momentum.py                       momentum state
└── graph_metadata.py                 graph metadata extraction
```

## Graph Package Layout

```text
graphfl_lab/graph/
├── sources/                          source implementations
├── signals/                          client representation extraction
├── builders.py                       graph builder registry bridge
├── registry.py                       named graph source/mode registry
├── controls.py                       random, shuffled, uniform, identity controls
├── clustering.py                     cluster/block graph helpers
├── diagnostics.py                    graph-level metric calculation
├── similarity.py                     relation similarity functions
└── sparsification.py                 dense, kNN, top-k, threshold topology
```

| Concept | Main Surface | Meaning |
|---|---|---|
| `graph_source` | `sources/`, `signals/` | client state를 graph input vector로 바꾸는 단계 |
| `graph_mode` | `builders.py`, `similarity.py`, `sparsification.py` | relation과 topology를 adjacency로 만드는 단계 |
| control graph | `controls.py` | real graph와 비교할 matched counterfactual |
| graph diagnostics | `diagnostics.py` | density, degree, entropy, spectral metric 계산 |

## Config Layout

```text
configs/
├── README.md                         config naming and usage policy
├── vision/
│   ├── baselines/                    baseline config presets
│   ├── diagnostic/
│   │   ├── smoke/                    quick diagnostic smoke configs
│   │   ├── core/                     core diagnostic suite configs
│   │   └── extend/                   extension diagnostic configs
│   ├── probes/                       graph/source/target probes
│   ├── smoke/                        minimal vision smoke configs
│   ├── stress/                       Non-IID stress configs
│   └── sweeps/                       sweep configs
└── cora/
    └── ablations/                    Cora graph ablation configs
```

## Script Layout

```text
scripts/
├── analysis/                         deep dives and result merge helpers
├── archive/                          historical analysis scripts
├── checks/                           preflight, validation bundle, parity checks
├── dev/                              golden and serialized-object maintenance
├── reports/                          plot and dashboard report helpers
├── smoke/                            smoke command wrappers
├── util/                             shared script utilities
└── validation/                       Graph-FL Evidence-pack entry points
```

| Script Area | Main Use |
|---|---|
| `scripts/validation/` | graph/control/diagnostic 검증 report 생성 |
| `scripts/checks/` | lightweight repository and artifact preflight |
| `scripts/reports/` | plot/table/report generation from experiment outputs |
| `scripts/dev/` | golden fixture and serialized asset maintenance |

## Test Layout

```text
tests/
├── cli/                              CLI dispatch and help/import tests
├── clients/                          client behavior tests
├── core/                             package identity and public surface tests
├── corrections/                      correction/control behavior tests
├── data/                             dataset and partition tests
├── designs/                          GraphFLDesign registry tests
├── diagnostics/                      schema and artifact tests
├── experiments/                      vision and Cora orchestration tests
├── extensions/                       custom aggregation and extension contract tests
├── golden/                           normalized baseline policy and fixtures
├── graph/                            graph source/builder/control tests
├── lifecycle/                        lifecycle context and trace tests
├── presentation/                     demo manifest drift와 static contract tests
├── scripts/                          script-level tests
├── strategies/                       baseline and Graph-FL strategy tests
├── structure/                        repository structure tests
├── suites/                           suite parser/reporting tests
└── validation/                       Evidence-pack validation tests
```

## Documentation Layout

```text
docs/
├── README.md                         documentation index
├── framework.md                      component 구조, lifecycle, metrics
├── evidence.md                       검증 결과와 한계
├── research.md                       prior work and design pattern survey
├── repository.md                     repository layout and change routing
├── maintenance.md                    compatibility and hygiene policy
├── history.md                        legacy experiments and migration phases
└── demos/
    ├── graphfl-assembly-scratch.html 보조 HTML demo
    └── graphfl-authoring-capabilities.js demo capability manifest
```

## Change Routing

| Change Request | Main Location | Test Location |
|---|---|---|
| add or modify `graph_source` | `graphfl_lab/graph/signals/`, `graphfl_lab/graph/sources/` | `tests/graph/` |
| add or modify `graph_mode` | `graphfl_lab/graph/similarity.py`, `graphfl_lab/graph/sparsification.py`, `graphfl_lab/graph/builders.py` | `tests/graph/` |
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
| CLI argument | `graphfl_lab/cli/` | `tests/cli/` |
| extension authoring/scaffold | `graphfl_lab/extensions/`, `graphfl_lab/cli/authoring.py` | `tests/extensions/`, `tests/cli/` |
| demo capability manifest | `graphfl_lab/presentation/`, `docs/demos/` | `tests/presentation/` |
| config | `configs/` | JSON validation |
| validation report | `graphfl_lab/validation/`, `scripts/validation/` | `tests/validation/` |
