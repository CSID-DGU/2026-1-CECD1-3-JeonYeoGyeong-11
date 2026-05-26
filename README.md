# Graph-FL Design Lab

Client update를 이용해 프레임워크 내부에서 다양한 client graph를 구성·교체하고, control graph ablation을 통해 Graph-FL gain이 실제 client relation structure 때문인지 아니면 dominance, norm, smoothing, optimizer 효과 때문인지 분해하는 실험 프레임워크.

Graph-FL Design Lab의 중심은 graph를 만드는 방식과 graph를 검증하는 방식을 같은
실험대 위에 올리는 것이다. 많은 Graph-FL 방법은 client representation, relation
score, topology, edge weight, aggregation target 중 일부를 바꾸며 발전한다. 이
저장소는 그 차이를 공통 부품으로 표현하고, 같은 control과 diagnostic metric으로
비교할 수 있게 만든다.

따라서 결과 해석은 단순한 `FedAvg` 대비 성능 비교에서 멈추지 않는다. real graph,
matched control, graph-free correction, mechanism metric을 함께 읽어 graph gain이
relation 정보에서 왔는지, smoothing이나 dominance correction으로도 설명되는지,
어떤 graph 구성요소가 민감한지 확인한다.

## Naming

| Area | Canonical | Notes |
|---|---|---|
| experiment track | `vision` | configs under `configs/vision/` |
| strategy/runtime | `graphfl` | `graphfl_lab/strategies/graphfl/` |
| aggregation | `graph_filtered_*` | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| filter strength | `graph_filter_strength` | CLI: `--graph-filter-strength` |
| historical JSON | `spectral_*`, `configs/general/` | read-only aliases; see [Compatibility](#compatibility) |

## Project Summary

| Topic | Content |
|---|---|
| Claim | 새 Graph-FL 알고리즘 제안이 아니라 graph gain의 원인 분해/진단 framework |
| Core question | Does graph structure explain the observed gain beyond simpler confounders? |
| Primary evidence | real-control gap, graph-free control gap, alignment, LOO, DI / N_eff |
| Minimum experiments | Non-IID stress preflight; real vs counterfactual + graph-free controls; source/mode/target attribution; diagnostic mechanism chain |

## How To Read This Project

| Layer | What It Answers |
|---|---|
| Graph authoring | 어떤 client signal로 graph를 만들고, 어떤 topology와 aggregation target에 연결하는가 |
| Controls | 같은 조건에서 relation 의미, smoothing, coarse community, dominance correction을 어떻게 분리하는가 |
| Diagnostics | accuracy/loss 변화와 alignment, DI, N_eff, LOO, graph metrics가 함께 움직이는가 |
| Prior-work mapping | 기존 Graph-FL/PFL 방법을 exact reproduction, proxy, interface target 중 어디에 놓을 수 있는가 |

| Document | Link |
|---|---|
| Experimental design | [docs/framework/graph_fl_experimental_design.md](docs/framework/graph_fl_experimental_design.md) |
| Metric reference | [docs/framework/graph_fl_experimental_design_appendix.md](docs/framework/graph_fl_experimental_design_appendix.md) |
| Docs index | [docs/README.md](docs/README.md) |
| Visual demo | [docs/demos/graphfl-assembly-scratch.html](docs/demos/graphfl-assembly-scratch.html) |

## Quick Start

Run from repository root. Python 3.11 is the reference version. Commands below
use `python`; replace it with the interpreter from your own virtual environment
when needed. No absolute local path is required.

### Environment

| Step | Command |
|---|---|
| Create venv | `python3.11 -m venv .venv` |
| Install deps | `python -m pip install -r requirements.txt` |
| Editable install | `python -m pip install -e .` |

Datasets are loaded through the repository data helpers and cached under the
project data directory. Experiment outputs are written under the configured
`out_dir`, usually an ignored experiment-output directory. Keep generated
results out of source commits unless a document explicitly asks for a summary
or fixture.

### Verify

| Check | Command |
|---|---|
| Unit tests | `python -m unittest discover -s tests` |
| Vision CLI | `python run_vision_experiment.py --help` |
| Suite CLI | `python run_vision_suite.py --help` |

### Run Paths

| Path | Use When | Entrypoint |
|---|---|---|
| Single vision run | one assembled graph design or one baseline should be checked quickly | `python run_vision_experiment.py --config <config.json>` |
| Vision diagnostic suite | real graph, controls, graph-free correction, and diagnostics should be compared together | `python run_vision_suite.py --config <config.json>` |
| Cora graph ablation | graph-structured input path and summary writer should be checked | `python run_graph_ablation.py --config <config.json>` |
| Evidence check | a produced result JSON should be checked for schema and attribution fields | `python scripts/checks/result_evidence_bundle.py <result.json> --kind single-run` |

### Smallest Graph Smoke

| Step | Command |
|---|---|
| Run | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| Output | `--out-dir` from config (gitignored experiment output root) |

| Metadata field | Expected |
|---|---|
| `graph_method` | `default_similarity_knn` |
| `graph_design` | `default_similarity_knn` |
| `graph_source` | `update` |
| `graph_mode` | `rbf_knn` |
| `aggregation_target` | `graph_filtered_update` |
| `graph_empty` | `false` |

### Suite Preflight And Smoke

| Step | Command |
|---|---|
| Preflight | `python scripts/checks/diagnostic_suite_preflight.py` |
| Suite | `python run_vision_suite.py --config configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json` |

## Assembly Model

| Stage | Role |
|---|---|
| method profile | selects component assembly |
| client_state | `--graph-source` |
| relation_estimator + topology_operator | `--graph-mode`, builders |
| aggregation_operator | `--aggregation-target` |
| diagnostics | traces and mechanism metrics |

Graph-FL methods are component assemblies, not strategy branches.

### CLI Knobs

| Component | CLI knob | Examples |
|---|---|---|
| client state | `--graph-source` | `update`, `ema_update`, `classifier_head_update`, `weight` |
| relation/topology | `--graph-mode` | `knn`, `rbf_knn`, `pfedgraph_qp`, custom builder |
| aggregation | `--aggregation-target` | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| method profile | `--graph-method` | `default_similarity_knn`, `pfedgraph`, custom profile |
| exact preset | `--graph-preset` | registered `GraphFLDesign` or compatibility alias |

| Default method | Composition |
|---|---|
| `default_similarity_knn` | update source + RBF similarity + kNN topology + `graph_filtered_update` |

## Main Interfaces

| Interface | Role | Location |
|---|---|---|
| `GraphFLDesign` | lifecycle component profile | `graphfl_lab/designs/` |
| `graph_method` | runnable method/profile selection | `graphfl_lab/graph/method_specs.py`, `graphfl_lab/graph/presets.py` |
| `graph_source` | client representation | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| `graph_mode` | relation/topology construction | `graphfl_lab/graph/registry.py`, `graphfl_lab/graph/builders.py` |
| `aggregation_target` | graph application target | `graphfl_lab/strategies/graphfl/targets.py` |
| diagnostics | graph/control metrics | `graphfl_lab/diagnostics/`, `graphfl_lab/strategies/graphfl/diagnostics.py` |
| suite grammar | repeatable variant tokens | `graphfl_lab/experiments/suites/vision/variants.py` |

## Add A Graph Algorithm

| Step | Action |
|---|---|
| 1 | Write method profile: client_state, relation, topology, aggregation, delivery, local_objective, state_store, diagnostics |
| 2 | Assign support level: `core-supported`, `proxy-supported`, `interface-target`, `out-of-scope` |
| 3 | Add `graph_source` only for a new client representation |
| 4 | Add `graph_mode` or builder only for new relation/topology |
| 5 | Expose combinations through `GraphFLDesign` and `--graph-method` |
| 6 | Add suite tokens/configs after source/mode/target path is verified |
| 7 | Test shape, determinism, metadata, diagnostics, control comparability |

Builder sketch:

```python
from graphfl_lab.graph import GraphBuildContext, register_graph_builder


@register_graph_builder("my_relation_graph")
def build_my_relation_graph(context: GraphBuildContext):
    z = context.z_mat
    adj = z @ z.T
    adj[adj < 0.0] = 0.0
    return adj, {"base_graph_kind": "my_relation_graph"}
```

Guide: [docs/framework/extension-guide.md](docs/framework/extension-guide.md)

## Repository Layout

```text
.
├── configs/                          JSON configs (track, smoke, suite, stress)
│   ├── vision/                       Fashion-MNIST vision track
│   └── cora/                         Cora / FGL graph-ablation track
├── graphfl_lab/                      installable package (Graph-FL runtime)
│   ├── app/                          Flower App config and server glue
│   ├── cli/                          argparse-only CLI modules
│   ├── clients/                      Flower client implementations
│   ├── data/                         dataset load and partition helpers
│   ├── designs/                      GraphFLDesign registry and presets
│   ├── diagnostics/                  result schemas, metrics, evidence writers
│   ├── graph/                        client relation graph (source, builder, registry)
│   ├── lifecycle/                    contracts, traces, counterfactual runner
│   ├── models/                       vision / Cora model definitions
│   ├── strategies/
│   │   ├── graphfl/                  graph-FL server strategy (filter, aggregate, trace)
│   │   └── baselines/                FedAvgM, FedSim, dominance-aware, graph_smooth, …
│   └── experiments/
│       ├── vision/                   single-run and suite orchestration
│       ├── cora/                       Cora single-run and ablation helpers
│       └── suites/vision/            variant grammar, artifacts, reporting
├── scripts/
│   ├── checks/                       validation without full training
│   ├── smoke/                        short executable smoke workflows
│   ├── reports/                      convergence plots and dashboard helpers
│   ├── analysis/                     suite deep-dive and merge helpers
│   └── dev/                          gate-check, migration utilities
├── docs/                             active and archived documentation
├── tests/                            unit, graph, strategy, suite, structure tests
├── data/                             dataset cache (gitignored)
├── run_vision_experiment.py          thin launcher → vision single run
├── run_vision_suite.py               thin launcher → vision suite
├── run_vision_client_count_sweep.py  thin launcher → client-count sweep
├── run_vision_stress_grid.py         thin launcher → stress grid
└── run_graph_ablation.py             thin launcher → Cora graph ablation
```

**Detailed map** (per-file trees, scripts, tests, capability checklist):
[docs/structure.md](docs/structure.md). **Docs index:**
[docs/README.md](docs/README.md).

## Documents

| Document | Purpose |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | release notes |
| [docs/README.md](docs/README.md) | docs index |
| [docs/demos/graphfl-assembly-scratch.html](docs/demos/graphfl-assembly-scratch.html) | graph_source, graph_mode, aggregation_target, controls, diagnostics를 실제 실행 JSON 흐름으로 조립하는 발표용 데모 |
| [docs/structure.md](docs/structure.md) | edit routing and responsibility boundaries |
| [docs/framework/claim.md](docs/framework/claim.md) | claim boundary |
| [docs/framework/graph_fl_experimental_design.md](docs/framework/graph_fl_experimental_design.md) | current experimental design |
| [docs/framework/graph_fl_experimental_design_appendix.md](docs/framework/graph_fl_experimental_design_appendix.md) | metric definitions |
| [docs/framework/interfaces.md](docs/framework/interfaces.md) | implementation interfaces |
| [docs/framework/extension-guide.md](docs/framework/extension-guide.md) | source/builder extension workflow |
| [docs/framework/prior-work-mapping.md](docs/framework/prior-work-mapping.md) | exact/proxy/interface boundary |
| [docs/framework/diagnostics.md](docs/framework/diagnostics.md) | diagnostic interpretation |
| [docs/framework/naming-and-compatibility.md](docs/framework/naming-and-compatibility.md) | compatibility policy |

## Execution Flow

| Step | Module |
|---|---|
| 1 | `run_vision_suite.py` |
| 2 | `run_vision_experiment.py` |
| 3 | `graphfl_lab/flower_runner.py` |
| 4 | `graphfl_lab/flower_app.py` |
| 5 | `graphfl_lab/strategies/graphfl/strategy.py` |
| 6 | graph source / builder / filtering / aggregation / diagnostics |

| Entrypoint | Purpose |
|---|---|
| `run_vision_experiment.py` | single vision run |
| `run_vision_suite.py` | suite orchestration |
| `run_vision_client_count_sweep.py` | client-count sweep |
| `run_vision_stress_grid.py` | stress grid |
| `run_graph_ablation.py` | Cora graph ablation |

## Smoke And Verification

| Step | Command |
|---|---|
| Config smoke | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| Result bundle | `python scripts/checks/result_evidence_bundle.py <result.json> --kind single-run` |
| Suite preflight | `python scripts/checks/diagnostic_suite_preflight.py` |
| Diagnostic suite | `python run_vision_suite.py --config configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json` |
| Prior-work proxy | `python scripts/smoke/prior_work_proxy.py` |
| Full unit tests | `python -m unittest discover -s tests` |
| Proxy parity | `python scripts/checks/prior_work_proxy_parity.py --summary <summary.json>` |

## Reporting And Analysis

| Tool | Command pattern |
|---|---|
| Convergence plot | `python scripts/reports/plot_vision_convergence.py --suite-dir <suite-dir>` |
| Dashboard mockup | `python scripts/reports/generate_dashboard_mockup.py --suite-dir <suite-dir>` |
| Deep dive | `python scripts/analysis/deep_dive_vision.py --suite-dir <dir> --suite-tag <tag> --variant <v> --seed <n>` |
| Merge fedavg/ours | `python scripts/analysis/merge_vision_fedavg_ours.py --help` |

Suite outputs: `vision_suite_*`, `result_vision_*`. Short `suite_*` names are read-only when present. Pre-rename `general_*` artifact names are not loaded by current code.

## Compatibility

New code and new runs use `graphfl_lab`, `vision`, `graph_filtered_*`, and `graph_filter_strength`. Gate 6 removed `run_general_*`, `general_*` facades, and `spectral_fl` imports.

| Old name | Role |
|---|---|
| `spectral_filter_strength` | JSON config key alias via `config_io` (CLI: `--graph-filter-strength`) |
| `spectral_filtered_*` | aggregation target input alias via `targets.canonical_aggregation_target()` |
| `configs/general/...` | config path alias to `configs/vision/...` |
| `ours_spectral_filtered_*` | result-tag pairing in suite reporting only |

| Reference | Link |
|---|---|
| Removals and tombstones | [docs/removed-materials.md](docs/removed-materials.md) |
| Active policy | [docs/framework/naming-and-compatibility.md](docs/framework/naming-and-compatibility.md) |

## Git Policy

| Path | Git |
|---|---|
| source code, runners, scripts | include |
| `scripts/reports/` | include (report **code**) |
| docs, configs, tests, CI | include |
| `requirements.txt`, `pyproject.toml` | include |
| `data/`, `experiments_current/`, `outputs/`, `runs/` | ignore (local only) |
| `reports/` (repo root) | ignore (generated plots/tables; local only) |
| `.venv/`, editor cache | ignore |

Root `reports/` stays on disk but is not tracked; use `git rm -r --cached reports/` once if an old case study was committed before the ignore rule.
