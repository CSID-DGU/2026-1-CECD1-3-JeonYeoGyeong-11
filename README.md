# Graph-FL Design Lab

Graph-FL gain이 실제 graph structure 때문인지, 아니면 dominance, norm, smoothing, optimizer 같은 단순 요인 때문인지 분해하는 실험 프레임워크.

Current names:

```text
vision              current experiment track
graphfl             current strategy/runtime identity
graph_filtered_*    current graph aggregation spelling
general/spectral    compatibility names
```

## Project Summary

```text
Claim:
  새 Graph-FL 알고리즘 제안이 아니라 graph gain의 원인 분해/진단 framework.

Core question:
  Does graph structure explain the observed gain beyond simpler confounders?

Primary evidence:
  real-control gap, graph-free control gap, alignment, LOO, DI / N_eff.

Minimum experiments:
  Non-IID stress preflight
  real graph vs counterfactual + graph-free controls
  minimal source/mode/target attribution
  diagnostic mechanism chain
```

Design doc: [docs/framework/graph_fl_experimental_design.md](docs/framework/graph_fl_experimental_design.md)
Metric reference: [docs/framework/graph_fl_experimental_design_appendix.md](docs/framework/graph_fl_experimental_design_appendix.md)

## Quick Start

Run from repository root.

### Environment

```text
Python 3.11
pip
PowerShell or bash
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

macOS/Linux:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
```

Existing local environment:

```powershell
D:\jongseol\.venv311\Scripts\python.exe --version
```

### Verify

```powershell
python -m unittest discover -s tests
python run_vision_experiment.py --help
python run_vision_suite.py --help
```

Without activation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

### Smallest Graph Smoke

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
```

Output:

```text
experiments_current/default_similarity_knn_smoke/
```

Expected metadata:

```text
graph_method=default_similarity_knn
graph_design=default_similarity_knn
graph_source=update
graph_mode=rbf_knn
aggregation_target=graph_filtered_update
graph_empty=false
```

### Suite Preflight And Smoke

```powershell
python scripts/checks/diagnostic_suite_preflight.py
python run_vision_suite.py --config configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json
```

### Manual Assembly

```powershell
python run_vision_experiment.py `
  --method ours `
  --dataset fashionmnist `
  --model mlp `
  --num-clients 2 `
  --rounds 1 `
  --train-subset-size 20 `
  --test-subset-size 20 `
  --graph-method default_similarity_knn `
  --knn-k 1 `
  --out-dir experiments_current/manual_default_graph `
  --run-tag manual_k1
```

## Assembly Model

Graph-FL methods are represented as component assemblies, not strategy branches.

```text
method profile
  -> client_state
  -> relation_estimator
  -> topology_operator
  -> aggregation_operator
  -> diagnostics
```

CLI knobs:

| Component | CLI knob | Examples |
|---|---|---|
| client state | `--graph-source` | `update`, `ema_update`, `classifier_head_update`, `weight` |
| relation/topology | `--graph-mode` | `knn`, `rbf_knn`, `pfedgraph_qp`, custom builder |
| aggregation | `--aggregation-target` | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| method profile | `--graph-method` | `default_similarity_knn`, `pfedgraph`, custom profile |
| exact preset | `--graph-preset` | registered `GraphFLDesign` or compatibility alias |

Default method:

```text
default_similarity_knn
= update source
+ RBF similarity
+ kNN topology
+ graph-filtered update aggregation
```

Run:

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
```

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

1. Write method profile: `client_state`, `relation`, `topology`, `aggregation`, `delivery`, `local_objective`, `state_store`, `diagnostics`.
2. Assign support level: `core-supported`, `proxy-supported`, `interface-target`, `out-of-scope`.
3. Add `graph_source` only for a new client representation.
4. Add `graph_mode` or graph builder only for new relation/topology.
5. Expose runnable combinations through `GraphFLDesign` and `--graph-method`.
6. Add suite tokens/configs after lower-level source/mode/target path is verified.
7. Test shape, determinism, metadata, diagnostics, and control comparability.

Builder example:

```python
from graphfl_lab.graph import GraphBuildContext, register_graph_builder


@register_graph_builder("my_relation_graph")
def build_my_relation_graph(context: GraphBuildContext):
    z = context.z_mat
    adj = z @ z.T
    adj[adj < 0.0] = 0.0
    return adj, {"base_graph_kind": "my_relation_graph"}
```

Run:

```powershell
python run_vision_experiment.py `
  --method ours `
  --graph-plugin my_project.graph_plugins.my_method `
  --graph-method my_method `
  --graph-source classifier_head_update `
  --graph-mode my_relation_graph `
  --aggregation-target graph_filtered_update
```

Guide: [docs/framework/extension-guide.md](docs/framework/extension-guide.md)

## Repository Layout

```text
configs/                         experiment configs
  vision/                         current vision configs
  cora/                           Cora/FGL ablation configs

graphfl_lab/
  cli/                            parser-only modules
  data/                           dataset loading and partitioning
  designs/                        GraphFLDesign registry/presets
  diagnostics/                    metrics and artifact writers
  graph/                          client relation graph construction
  lifecycle/                      component contracts and traces
  strategies/
    graphfl/                      graph-FL runtime
    baselines/                    baseline strategies
    spectral/                     old import compatibility
  experiments/
    vision/                       current run orchestration
    suites/vision/                suite grammar and reporting
    general/                      old module compatibility

spectral_fl/
  __init__.py                     old package import shim

scripts/
  checks/                         non-training validation
  smoke/                          executable smoke runs
  reports/                        plotting/dashboard helpers
  analysis/                       analysis helpers and legacy wrappers

docs/
  framework/                      active framework docs
  research/                       literature/design notes
  archive/                        previous direction and migrations

tests/                            unit, structure, suite, graph, strategy tests
data/                             local dataset cache, ignored
experiments_current/              local experiment outputs, ignored
```

Detailed routing: [docs/structure.md](docs/structure.md)

## Documents

| Document | Purpose |
|---|---|
| [docs/README.md](docs/README.md) | docs index |
| [docs/structure.md](docs/structure.md) | edit routing and responsibility boundaries |
| [docs/framework/claim.md](docs/framework/claim.md) | claim boundary |
| [docs/framework/graph_fl_experimental_design.md](docs/framework/graph_fl_experimental_design.md) | current experimental design |
| [docs/framework/graph_fl_experimental_design_appendix.md](docs/framework/graph_fl_experimental_design_appendix.md) | metric definitions |
| [docs/framework/interfaces.md](docs/framework/interfaces.md) | implementation interfaces |
| [docs/framework/extension-guide.md](docs/framework/extension-guide.md) | source/builder extension workflow |
| [docs/framework/prior-work-mapping.md](docs/framework/prior-work-mapping.md) | exact/proxy/interface boundary |
| [docs/framework/diagnostics.md](docs/framework/diagnostics.md) | diagnostic interpretation |
| [docs/framework/naming-and-compatibility.md](docs/framework/naming-and-compatibility.md) | compatibility names |

## Execution Flow

```text
run_vision_suite.py
-> run_vision_experiment.py
-> graphfl_lab/flower_runner.py
-> graphfl_lab/flower_app.py
-> graphfl_lab/strategies/graphfl/strategy.py
-> graph source / graph builder / filtering / aggregation / diagnostics
```

Help commands:

```powershell
python run_vision_experiment.py --help
python run_vision_suite.py --help
python run_vision_client_count_sweep.py --help
python run_vision_stress_grid.py --help
```

## Smoke And Verification

```powershell
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
python scripts/checks/result_evidence_bundle.py experiments_current/default_similarity_knn_smoke/result_vision_ours_seed42_default_similarity_knn_smoke.json --kind single-run
python scripts/checks/diagnostic_suite_preflight.py
python run_vision_suite.py --config configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json
python scripts/smoke/prior_work_proxy.py
```

Full checks:

```powershell
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
python scripts/checks/prior_work_proxy_parity.py --summary experiments_current/prior_work_proxy_smoke/<stamp>/prior_work_proxy_summary.json
```

## Compatibility

New code uses `vision`, `graphfl`, `graph_filtered_*`.

| Old name | Role |
|---|---|
| `run_general_*.py` | `run_vision_*.py` compatibility wrapper |
| `spectral_fl` | old package import shim |
| `graphfl_lab/experiments/general/` | old import path wrapper |
| `graphfl_lab/strategies/spectral/` | `graphfl_lab/strategies/graphfl/` wrapper |
| `spectral_filtered_*`, `spectral_filter_strength` | config/result compatibility alias |

Do not add new logic to compatibility paths.

## Git Policy

| Path | Git |
|---|---|
| source code, runners, scripts | include |
| docs | include |
| configs | include |
| tests and CI | include |
| `requirements.txt`, `pyproject.toml` | include |
| `data/` | ignore |
| `experiments_current/`, `reports/`, `outputs/`, `runs/` | ignore |
| `.venv/`, `.venv311/`, editor cache | ignore |
