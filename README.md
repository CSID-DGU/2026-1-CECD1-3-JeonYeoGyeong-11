# Spectral Client Update Graph

Federated Learning에서 client update와 client model을 **client relation graph 위의 graph signal**로 보고, spectral filtering / message-passing style aggregation을 실험하는 연구 코드입니다.

현재 목표는 "바로 FedAvg보다 항상 좋다"를 주장하는 것이 아니라, Non-IID 환경에서 client relation graph가 실제로 의미 있는 aggregation signal을 만들 수 있는지 검증하는 것입니다.

## Overview

| Question | Implementation Direction |
|---|---|
| client들이 비슷한 update behavior를 공유하는가? | update delta 기반 graph construction |
| client의 semantic state를 더 잘 잡을 수 있는가? | classifier head update/weight graph |
| single-round update graph가 너무 noisy한가? | client update EMA graph / EMA signal filtering |
| weight 자체를 graph signal로 볼 수 있는가? | spectral filtered local weight aggregation |
| graph가 진짜 의미 있나, 그냥 regularization인가? | random-matched / magnitude / FedOpt baseline 비교 |

## Documents

| File | Content |
|---|---|
| [docs/structure.md](docs/structure.md) | repository layout, change routing, and boundary rules |
| [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) | idea decomposition, implemented knobs, experiment questions |
| [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md) | current results, corrected interpretation, rerun checklist |
| [docs/SEMANTIC_CLIENT_GRAPH_ROADMAP.md](docs/SEMANTIC_CLIENT_GRAPH_ROADMAP.md) | semantic graph roadmap and next paper direction |

Local dataset cache와 experiment output은 Git에 포함하지 않습니다. 보통 `data/`와 `experiments_current/` 아래에 생성됩니다.

## Formulation

Round `t`에서 server는 각 client가 local training 후 보낸 model weight를 받고 update delta를 계산합니다.

```text
g_i = w_i - w_global
G = [g_1; ...; g_N]
```

Client relation graph는 선택된 `graph_source`로 만듭니다.

```text
X_i = graph_source(w_i, g_i)
W = client_graph(X)
L = D - W
```

Spectral filtered update path는 client axis에 graph low-pass를 적용합니다.

```text
G_filtered = h(L)G
w_next = w_global + sum_i alpha_i G_filtered_i
```

Weight filtering path는 local model weight 자체를 graph signal로 봅니다.

```text
W_local = [w_1; ...; w_N]
W_filtered = h(L)W_local
w_next = sum_i alpha_i W_filtered_i
```

여기서 `g_i`는 raw mini-batch gradient가 아닙니다. 현재 구현은 local training 전체가 만든 **model update delta**를 pseudo-gradient처럼 사용합니다.

## Implemented Scope

### Graph Source

`--graph-source`는 client graph를 만들 때 어떤 representation을 볼지 정합니다.

| Option | Meaning |
|---|---|
| `update` | full update delta `g_i` |
| `normalized_update` | client별 L2-normalized update delta |
| `layerwise_update` | tensor block별 normalize 후 concat |
| `layer_slice_update` | `--graph-layer-start/end`로 고른 update slice |
| `classifier_head_update` | final weight/bias update pair |
| `classifier_head_ema_update` | client update EMA의 classifier head |
| `layerwise_classifier_head_update` | head update를 tensor별 normalize |
| `ema_update` | client별 update EMA |
| `normalized_ema_update` | client별 update EMA 방향 |
| `layerwise_ema_update` | client별 update EMA를 layerwise normalize |
| `weight` | local model weight `w_i` |
| `classifier_head_weight` | final classifier head weight |
| `layerwise_classifier_head_weight` | head weight를 tensor별 normalize |
| `layer_slice_weight` | selected local weight slice |
| `layerwise_weight` | tensor block별 normalized weight |

`classifier_head_*`는 semantic graph 후보입니다. 전체 update graph가 noisy할 수 있으므로, label skew가 직접 반영되는 head 쪽을 별도 main candidate로 봅니다.

### Graph Mode

`--graph-mode`는 edge construction rule입니다.

| Option | Meaning |
|---|---|
| `dense` | positive cosine dense graph |
| `knn` | top-k positive cosine graph |
| `mutual_knn` | mutual top-k graph |
| `threshold` | cosine threshold graph |
| `random` | edge-count matched random graph |
| `uniform` | all off-diagonal edges equal |
| `magnitude`, `magnitude_knn` | cosine plus update-norm mismatch penalty |
| `rbf`, `rbf_knn` | Gaussian/RBF graph |
| `signed_abs`, `negative` | relation-sign control graphs |
| `learned_smooth`, `learned_smooth_knn` | smoothness surrogate graph |

Random-matched graph는 필수 control입니다. semantic/head graph가 random보다 좋아야 graph construction claim이 생깁니다.

### Aggregation Target

`--aggregation-target`는 마지막에 무엇을 aggregate할지 정합니다.

| Option | Meaning |
|---|---|
| `update` | `w_global + sum_i alpha_i g_i` |
| `weight` | `sum_i alpha_i w_i` |
| `spectral_filtered_update` | current update matrix `G`를 graph low-pass 후 aggregate |
| `spectral_filtered_ema_update` | client update EMA를 graph low-pass 후 aggregate |
| `spectral_filtered_weight` | local weight matrix를 graph low-pass 후 aggregate |

`--spectral-filter-strength p` controls:

```text
h(lambda) = (1 - lambda / lambda_max) ** p
```

`p=0`은 no filtering, `p=1`은 linear low-pass, `p>1`은 stronger low-pass입니다.

### Client Update EMA

Client별 update signal을 temporal smoothing해서 graph 또는 filtering target으로 사용할 수 있습니다.

```text
u_i,t = beta * u_i,t-1 + (1 - beta) * g_i,t
```

| Usage | Config |
|---|---|
| graph만 EMA로 생성 | `--graph-source ema_update --aggregation-target spectral_filtered_update` |
| EMA signal 자체를 filtering | `--graph-source ema_update --aggregation-target spectral_filtered_ema_update` |
| EMA signal + server momentum | 위 설정 + `--ours-server-momentum` |

Option:

```text
--client-update-ema-alpha 0.8
```

### Baselines

`--method` 또는 suite variant token으로 실행합니다.

| Baseline | Meaning |
|---|---|
| `fedavg` | sample-weighted FedAvg |
| `fedavgm` | FedAvg with server momentum |
| `fedadagrad` | FedAdagrad server optimizer |
| `fedadam` | FedAdam server optimizer |
| `fedyogi` | FedYogi server optimizer |
| `fednova` | FedNova-style normalized averaging |
| `fedprox`, `fedprox_mu0p01` | local proximal regularization |
| `fedmedian` | coordinate-wise median |
| `fedtrimmedavg_beta0p1` | coordinate-wise trimmed mean |
| `fedsim_k{K}` | similarity-guided cluster aggregation proxy |

## Important Current Status

현재까지의 핵심 해석은 보수적입니다.

```text
Observed:
  update-cosine spectral-only is unstable and often weak.
  spectral + server momentum can improve N=20 stress runs.
  random-matched graph can be strong, so graph construction is not yet proven.

Implication:
  main contribution should move from raw update graph filtering
  to semantic/head/temporal client graph based signal denoising.
```

또한 이전 일부 결과는 Flower/Ray client result order가 round마다 달라지는 문제 때문에 graph EMA row가 client id와 섞였습니다. 현재 strategy는 numeric cid 기준으로 fit results를 sort한 뒤 graph를 만듭니다.

## Repository Layout

```text
spectral-client-update-graph/
├── README.md                         # project overview, execution path, current claim boundary
├── pyproject.toml                    # package metadata and Flower App component defaults
├── requirements.txt                  # runtime dependency bounds
├── .gitignore                        # excludes local data, outputs, caches, and virtualenvs
│
├── configs/                          # tracked experiment configs, grouped by research question
│   ├── general/
│   │   ├── smoke/                    # tiny end-to-end checks
│   │   ├── baselines/                # FedOpt/FedNova/FedProx/etc availability checks
│   │   ├── probes/
│   │   │   ├── frequency/            # graph-frequency diagnostic questions
│   │   │   ├── graph_source/         # update/head/EMA/weight representation probes
│   │   │   ├── structure/            # kNN/random/RBF/magnitude/learned graph controls
│   │   │   └── tau/                  # adaptive/fixed/normalized tau controls
│   │   ├── stress/
│   │   │   ├── client_count/         # N=20/N=50 and scaling stress settings
│   │   │   └── fedavg_collapse/      # strong label-skew collapse regimes
│   │   └── sweeps/
│   │       └── client_count/         # repeated suites over client-count grids
│   └── cora/
│       └── ablations/
│           └── graph/                # Cora graph-construction ablation configs
│
├── docs/                             # research notes and repo navigation
│   ├── structure.md                  # change-routing map for humans and AI agents
│   ├── EXPERIMENT_DESIGN.md          # hypothesis, axes, and experiment queue
│   ├── EXPERIMENT_RESULTS.md         # current observations and interpretation
│   └── SEMANTIC_CLIENT_GRAPH_ROADMAP.md
│
├── scripts/                          # offline analysis/report helpers
│   ├── analysis/                     # merge, deep-dive, aggregation analysis
│   ├── reports/                      # result JSON -> plots/tables/reports
│   └── util/                         # small console helpers
│
├── tests/                            # tests mirror the source responsibility tree
│   ├── core/                         # config/projection utilities
│   ├── clients/                      # Flower client behavior
│   ├── data/                         # dataset partitioning helpers
│   ├── graph/                        # graph construction, similarity, diagnostics
│   ├── experiments/
│   │   └── general/                  # suite variants, reporting, stress-grid logic
│   ├── strategies/
│   │   └── spectral/                 # aggregation, filtering, targets, momentum
│   ├── scripts/
│   │   └── reports/                  # report-script parsing behavior
│   └── structure/                    # import-boundary and facade-thinness guards
│
├── run_experiment.py                 # compatibility launcher: Cora single run
├── run_graph_ablation.py             # compatibility launcher: Cora graph ablation
├── run_general_experiment.py         # compatibility launcher: General FL single run
├── run_general_suite.py              # compatibility launcher: General FL suite
├── run_general_client_count_sweep.py # compatibility launcher: General client-count sweep
├── run_general_stress_grid.py        # compatibility launcher: General stress grid
│
├── spectral_fl/
│   ├── app/                          # Flower App runtime glue
│   │   ├── config.py                 # default run config and Context -> argparse Namespace
│   │   └── data_cache.py             # cached data loading for Flower ClientApp/ServerApp
│   │
│   ├── cli/                          # argparse only; no model/data/strategy construction
│   │   ├── cora_experiment.py
│   │   ├── graph_ablation.py
│   │   ├── general_experiment.py
│   │   ├── general_suite.py
│   │   ├── general_client_count_sweep.py
│   │   └── general_stress_grid.py
│   │
│   ├── experiments/                  # run orchestration and output writing
│   │   ├── cora/
│   │   │   ├── single_run.py         # Cora/FGL single experiment orchestration
│   │   │   └── graph_ablation.py     # Cora graph-ablation suite orchestration
│   │   ├── general/
│   │   │   ├── single_run.py         # torchvision/FashionMNIST/MNIST/CIFAR single run
│   │   │   ├── suite.py              # variant-by-seed suite runner
│   │   │   ├── client_count_sweep.py # repeated suites over num_clients
│   │   │   └── stress_grid.py        # multi-axis stress grid runner
│   │   └── suites/
│   │       ├── stats.py              # shared suite summary math
│   │       └── general/
│   │           ├── variants.py       # suite variant token grammar
│   │           └── reporting.py      # summary/interpretation writers
│   │
│   ├── graph/                        # client relation graph construction
│   │   ├── builders.py               # assembles signal, similarity, sparsification
│   │   ├── diagnostics.py            # graph density/edge diagnostics
│   │   ├── sparsification.py         # dense, kNN, random, threshold, uniform rules
│   │   ├── signals/                  # update/head/weight vector extraction
│   │   ├── similarity/               # cosine, RBF, magnitude-aware relations
│   │   └── sources/                  # graph_source option selection and config
│   │
│   ├── strategies/
│   │   ├── spectral/                 # Ours strategy internals
│   │   │   ├── strategy.py           # Flower strategy lifecycle only
│   │   │   ├── aggregation.py        # alpha/conflict/client-weight math
│   │   │   ├── filtering.py          # Laplacian and graph low-pass filters
│   │   │   ├── targets.py            # update/EMA/weight aggregation targets
│   │   │   ├── momentum.py           # server optimizer / momentum path
│   │   │   ├── diagnostics.py        # spectral energy and heterogeneity metrics
│   │   │   ├── config.py             # spectral strategy state containers
│   │   │   └── tracing.py            # optional round trace helpers
│   │   └── baselines/                # FedAvgM/FedOpt/FedNova/FedSim/tracing helpers
│   │
│   ├── clients/                      # Cora and vision Flower clients
│   ├── data/                         # Cora and vision dataset loaders/partitioners
│   ├── models/                       # Cora GCN and vision MLP/CNN models
│   ├── flower_app.py                 # Flower ClientApp/ServerApp entrypoints
│   ├── flower_runner.py              # local Flower App runner and flwr-run bridge
│   ├── projection.py                 # flatten/unflatten/random projection helpers
│   ├── strategy.py                   # backward-compatible strategy facade
│   ├── update_graph.py               # backward-compatible graph facade
│   ├── aggregation.py                # backward-compatible aggregation facade
│   └── spectral_diagnostics.py       # backward-compatible diagnostics facade
│
├── data/                             # local-only dataset cache, ignored by Git
└── experiments_current/              # local-only experiment outputs, ignored by Git
```

The guiding rule is simple: implementation code is grouped by **reason to
change**, while configs and generated outputs are grouped by **experiment
question**.  For example, a new graph source should land in
`spectral_fl/graph/signals/` and `spectral_fl/graph/sources/`, not inside a
specific experiment folder.  A new stress config should land under the narrowest
`configs/<track>/<question>/` folder.

Compatibility files such as `run_*.py`, `spectral_fl/strategy.py`, and
`spectral_fl/update_graph.py` intentionally stay thin.  They preserve older
imports and command paths, but new logic should go into the scoped modules shown
above.  The guard tests in `tests/structure/` enforce those boundaries.

## Environment

Recommended:

```text
Python 3.11
Flower simulation
PyTorch / torchvision
```

Install:

```powershell
python -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## Execution Flow

General FL suite:

```powershell
python run_general_suite.py --config configs/general/baselines/fedopt_smoke.json
```

Client-count sweep smoke:

```powershell
python run_general_client_count_sweep.py --config configs/general/smoke/semantic_ema_weight.json
```

Client-count stress check:

```powershell
python run_general_client_count_sweep.py --config configs/general/stress/client_count/lowpass_spectral_only_n20_n50_postfix.json
```

Single run:

```powershell
python run_general_experiment.py `
  --method ours `
  --dataset fashionmnist `
  --model mlp `
  --num-clients 20 `
  --rounds 10 `
  --partition dirichlet `
  --dirichlet-alpha 0.03 `
  --graph-source classifier_head_update `
  --graph-mode knn `
  --knn-k 1 `
  --aggregation-target spectral_filtered_update
```

## Smoke Experiments

| Config | Purpose |
|---|---|
| `configs/general/baselines/fedopt_smoke.json` | FedOpt/FedNova baseline availability |
| `configs/general/smoke/semantic_ema_weight.json` | head graph, EMA graph, EMA signal, weight filtering smoke |
| `configs/general/stress/client_count/lowpass_spectral_only_n20_n50_postfix.json` | post cid-order fix spectral-only stress rerun |

## Verification

Preferred checks before pushing:

```powershell
python -m compileall spectral_fl tests
python -m unittest discover -s tests
```

If local Python execution is blocked, at least run:

```powershell
git diff --check
```

and then run the smoke configs as soon as the environment allows.
