# Experiment Design

이 문서는 client graph 기반 FL aggregation 아이디어를 어떤 질문으로 나누어 실험할지 정리합니다. 스타일은 README와 맞춰서, 한국어 설명과 English technical term을 함께 사용합니다.

## 1. Core Hypothesis

Federated clients는 독립적인 update vector 집합이 아니라, relation graph 위에 놓인 graph signal population으로 볼 수 있습니다.

```text
node: client i
signal: update delta g_i = w_i - w_global
optional signal: local weight w_i or client update EMA u_i
edge: similarity between selected client representations
graph: W
laplacian: L = D - W
```

Main question:

```text
Can a meaningful client graph denoise client model/update signals
better than simple sample-weighted averaging?
```

현재 가장 중요한 분리는 다음 세 가지입니다.

| Axis | Question |
|---|---|
| graph construction | 어떤 representation으로 client relation을 만들 것인가? |
| signal filtering | update, EMA update, weight 중 무엇을 graph signal로 정제할 것인가? |
| server optimization | momentum/FedOpt 효과와 graph filtering 효과를 어떻게 분리할 것인가? |

## 2. Current Implementation Axes

### A. Graph Source

`graph_source`는 graph edge를 만들 node feature입니다.

| Source Group | Options | Purpose |
|---|---|---|
| update behavior | `update`, `normalized_update`, `layerwise_update` | current round에서 global model을 어떻게 바꾸려는지 |
| layer/head update | `layer_slice_update`, `classifier_head_update`, `layerwise_classifier_head_update` | label skew가 강하게 반영되는 head 중심 graph |
| temporal update | `ema_update`, `normalized_ema_update`, `classifier_head_ema_update` | single-round update noise 완화 |
| model state | `weight`, `classifier_head_weight`, `layerwise_weight` | local model state 자체의 similarity |

Interpretation:

```text
update graph:
  behavior similarity

classifier head graph:
  label/decision-boundary similarity proxy

EMA graph:
  temporal denoised behavior similarity

weight graph:
  model-space state similarity
```

### B. Graph Mode

`graph_mode`는 edge construction rule입니다.

| Mode | Role |
|---|---|
| `knn` | sparse similarity graph |
| `dense` | full positive cosine graph |
| `random` | matched random control |
| `uniform` | graph structure 없는 smoothing control |
| `magnitude_knn` | update magnitude mismatch를 penalize |
| `rbf_knn` | Euclidean/RBF similarity |
| `learned_smooth_knn` | smoothness surrogate graph |

Key control:

```text
semantic/head graph must beat random-matched graph
otherwise graph construction claim is weak.
```

### C. Aggregation Target

`aggregation_target`는 graph가 만들어진 뒤 무엇을 aggregate할지 정합니다.

| Target | Formula Sketch | Meaning |
|---|---|---|
| `update` | `w + sum alpha_i g_i` | FedAvg-style update aggregation |
| `weight` | `sum alpha_i w_i` | local model averaging |
| `spectral_filtered_update` | `w + sum alpha_i h(L)G_i` | current update signal denoising |
| `spectral_filtered_ema_update` | `w + sum alpha_i h(L)U_i` | temporal-spatial denoising |
| `spectral_filtered_weight` | `sum alpha_i h(L)W_i` | model-space graph denoising |

Important note:

```text
h(L)G then alpha-average is linear.
If h(L) is symmetric, it can look like graph-smoothed alpha.
Therefore the real contribution must come from meaningful graph source,
normalization, layer/head choice, or temporal denoising.
```

## 3. Main Experimental Questions

### Q1. Is raw update graph enough?

Baseline variants:

```text
fedavg
fedavgm
ours_spectral_filtered_knn_k1_lp0p5_fixed_tau_spectral_only
ours_spectral_filtered_random_matched_k1_lp0p5_fixed_tau_spectral_only
```

Expected interpretation:

```text
If kNN update graph <= random graph,
raw update cosine is not a strong semantic graph.
```

### Q2. Does classifier head graph help?

Variants:

```text
ours_head_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_head_weight_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_layerwise_head_graph_knn_k1_lp0p5_fixed_tau_spectral_only
```

Question:

```text
Does classifier head similarity produce cleaner client neighborhoods
under label skew?
```

### Q3. Does temporal smoothing explain momentum gains?

Variants:

```text
ours_ema_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_signal_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_signal_knn_k1_lp0p5_serverm_fixed_tau
```

Interpretation:

| Result | Meaning |
|---|---|
| EMA graph improves spectral-only | graph edge noise was important |
| EMA signal improves spectral-only | temporal denoising itself helps |
| only serverM helps | effect may be mostly server optimizer |

### Q4. Is weight-space graph denoising meaningful?

Variants:

```text
ours_weight_graph_spectral_weight_agg_knn_k1_lp0p5_fixed_tau_spectral_only
ours_head_weight_graph_spectral_weight_agg_knn_k1_lp0p5_fixed_tau_spectral_only
```

Interpretation:

```text
This is a fixed graph message-passing / model-space denoising view.
It should be framed conservatively, not as a fully learnable GNN.
```

## 4. Stress Settings

Primary stress condition:

```text
dataset=fashionmnist
model=mlp
partition=dirichlet
dirichlet_alpha=0.03
clients=20 and 50
rounds=10
local_epochs=2
batch_size=64
train_subset_size=1000
test_subset_size=300 or 1000
warmup_rounds=3
fixed_tau=1.0
conflict_mix=0.0
min_client_weight=0.0
```

Why this setting:

```text
FedAvg collapse appears under strong label skew.
It is small enough for quick iteration.
N=20 and N=50 expose client-count sensitivity.
```

## 5. Baseline Plan

Minimum comparison set:

```text
fedavg
fedavgm
fedadagrad
fedadam
fedyogi
fednova
fedprox_mu0p01
```

Ours comparison should always include:

```text
raw update graph
random-matched graph
head graph
EMA graph
weight-space filtering
serverM and no-serverM variants
```

## 6. Decision Criteria

| Criterion | Why it matters |
|---|---|
| spectral-only beats FedAvg | filtering itself has signal |
| head/EMA graph beats raw update graph | graph source matters |
| semantic graph beats random graph | graph construction claim survives |
| method competes with FedAvgM/FedOpt | baseline strength is adequate |
| N=50 does not collapse | scalability beyond small client count |

Current claim boundary:

```text
Allowed:
  The update graph version is preliminary and unstable.
  Momentum interaction shows possible value.
  Better graph construction is the main research bottleneck.

Not allowed yet:
  Our method is SOTA.
  Spectral filtering alone is proven.
  Full weight GNN aggregation is validated.
```

## 7. Immediate Experiment Queue

Smoke:

```powershell
python run_general_client_count_sweep.py --config configs/general/smoke/semantic_ema_weight.json
```

Baseline smoke:

```powershell
python run_general_suite.py --config configs/general/baselines/fedopt_smoke.json
```

Post-fix spectral-only check:

```powershell
python run_general_client_count_sweep.py --config configs/general/stress/client_count/lowpass_spectral_only_n20_n50_postfix.json
```

After smoke passes, run multi-seed N=20/N=50 with the best 3 to 5 graph variants.
