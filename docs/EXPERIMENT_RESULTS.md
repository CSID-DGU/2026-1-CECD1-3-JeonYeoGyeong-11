# Experiment Results

이 문서는 현재까지의 실험 결과와 해석 상태를 기록합니다. 숫자는 claim을 확정하기 위한 것이 아니라, 다음 실험 방향을 정하기 위한 working log입니다.

## Current Status

요약:

```text
raw update-cosine spectral-only:
  weak / unstable

spectral + server momentum:
  N=20 stress에서 개선 신호 있음

random-matched graph:
  생각보다 강함

main bottleneck:
  graph construction
```

따라서 현재 연구 상태는 다음과 같습니다.

```text
Not final claim:
  spectral filtering alone works.

Current hypothesis:
  semantic/head/temporal client graph may make graph signal denoising useful.
```

## Important Fix: Client Ordering

Flower/Ray fit result order가 round마다 바뀌면서, graph row와 client id가 섞이는 문제가 있었습니다.

Problem:

```text
round 1 graph row 0 = client 5
round 2 graph row 0 = client 8
EMA graph mixes rows across different clients
```

Fix:

```text
sort fit results by numeric cid before graph construction and aggregation
```

After fix:

```text
round cids = 0,1,2,... consistently
```

이전 smoke 결과 중 spectral variant 숫자는 historical note로만 봐야 합니다.

## N=20 Pair Check After Fix

Setting:

```text
dataset=fashionmnist
model=mlp
clients=20
partition=dirichlet
dirichlet_alpha=0.03
rounds=10
local_epochs=2
warmup_rounds=3
seed=42
aggregation_target=spectral_filtered_update
conflict_mix=0.0
min_client_weight=0.0
fixed_tau=1.0
```

Result paths:

```text
experiments_current/lowpass_pair_check_n20_seed42_sorted
experiments_current/lowpass_pair_check_n20_seed42_sorted_repeat
```

| Clients | Variant | Final Acc | Delta vs FedAvg |
|---:|---|---:|---:|
| 20 | FedAvg | 0.369 | +0.000 |
| 20 | kNN k1 low-pass 0.5 + server momentum | 0.458 | +0.089 |
| 20 | kNN k1 low-pass 1.0 + server momentum | 0.475 | +0.106 |

Interpretation:

```text
N=20 seed42에서 spectral + server momentum은 FedAvg collapse를 완화했다.
하지만 이 결과만으로 graph semantics claim은 부족하다.
```

## Post-Fix Spectral-Only Sweep

Config:

```text
configs/general/stress/client_count/lowpass_spectral_only_n20_n50_postfix.json
```

Output:

```text
experiments_current/lowpass_stress_spectral_only_n20_n50_postfix/client_count_sweep_summary.md
```

N=20:

| Variant | Final Acc | Delta vs FedAvg |
|---|---:|---:|
| FedAvg | 0.369 | +0.000 |
| random matched k1 lp0.5 | 0.375 | +0.006 |
| magnitude k1 lp0.5 | 0.367 | -0.002 |
| dense lp0.5 | 0.358 | -0.011 |
| kNN k1 lp0.25 | 0.355 | -0.014 |
| kNN k1 lp0.5 | 0.350 | -0.019 |
| kNN k1 lp1.0 | 0.343 | -0.026 |

N=50:

| Variant | Final Acc | Delta vs FedAvg |
|---|---:|---:|
| FedAvg | 0.297 | +0.000 |
| random matched k1 lp0.5 | 0.309 | +0.012 |
| dense lp0.5 | 0.295 | -0.002 |
| magnitude k1 lp0.5 | 0.292 | -0.005 |
| kNN k1 lp0.5 | 0.291 | -0.006 |
| kNN k1 lp0.25 | 0.289 | -0.008 |
| kNN k1 lp1.0 | 0.278 | -0.019 |

Interpretation:

```text
raw update graph spectral-only does not currently beat FedAvg reliably.
random control being competitive suggests graph quality is not yet established.
```

## Baseline Implementation Status

Implemented and configurable:

| Method | Status |
|---|---|
| FedAvg | available |
| FedAvgM | available |
| FedAdagrad | available |
| FedAdam | available |
| FedYogi | available |
| FedNova-style | available |
| FedProx | available |
| FedMedian | available |
| FedTrimmedAvg | available |
| FedSim-style | available |

Smoke config:

```text
configs/general/baselines/fedopt_smoke.json
```

Small smoke previously confirmed `fedadam` and `fednova` result JSON generation, but full baseline comparison still needs systematic rerun.

## New Variants To Validate

New graph/signal options were added and need smoke verification.

Config:

```text
configs/general/smoke/semantic_ema_weight.json
```

Included variants:

```text
fedavg
fedavgm
ours_head_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_head_weight_graph_spectral_weight_agg_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_signal_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_signal_knn_k1_lp0p5_serverm_fixed_tau
```

Expected questions:

| Variant Group | Question |
|---|---|
| head graph | classifier head가 semantic graph proxy가 되는가? |
| head weight graph | local decision boundary state가 더 안정적인가? |
| EMA graph | graph edge noise를 줄이는가? |
| EMA signal | temporal denoising 자체가 도움이 되는가? |
| weight filtering | model-space graph denoising이 의미 있는가? |

## Next Experiment Plan

### Step 1. Smoke

Run:

```powershell
python run_general_client_count_sweep.py --config configs/general/smoke/semantic_ema_weight.json
```

Pass criteria:

```text
all variants finish
result JSON includes graph_source_used
result JSON includes aggregation_target_used
no crash from EMA or spectral_filtered_weight path
```

### Step 2. N=20 Stress

Run promising variants against:

```text
fedavg
fedavgm
fedadam
fedyogi
fednova
random-matched graph control
```

Primary metrics:

```text
final accuracy
delta vs FedAvg
delta vs FedAvgM
delta vs random graph
win rate over seeds
```

### Step 3. N=50 Stress

Repeat N=20 winners at N=50.

Key question:

```text
Does semantic/head/EMA graph reduce the N=50 degradation?
```

### Step 4. Filter Strength Sweep

For promising variants:

```text
lp0p25
lp0p5
lp1p0
```

Question:

```text
Does the method respond meaningfully to filter strength,
or are gains mostly from optimizer/regularization side effects?
```

## Current Claim Boundary

Safe to say:

```text
The raw update graph version is not sufficient.
Server momentum interaction shows possible value.
Graph construction is the main bottleneck.
We now have code paths to test head graph, EMA graph, and weight-space filtering.
```

Do not claim yet:

```text
SOTA performance.
Semantic graph is proven.
Spectral-only is robust.
Weight GNN aggregation is validated.
```
