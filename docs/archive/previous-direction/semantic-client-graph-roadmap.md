# Semantic Client Graph Roadmap

이 문서는 현재 아이디어를 논문 방향으로 정리한 roadmap입니다. 핵심은 "spectral trick"이 아니라, **semantic client graph를 어떻게 정의하고 그 위에서 client signal을 어떻게 denoise할 것인가**입니다.

## 1. Core Thesis

Working thesis:

```text
Federated clients form a model/update signal population.
The main contribution is constructing a client graph whose spectrum
captures meaningful agreement and disagreement under Non-IID data.
```

Current formulation:

```text
client graph W
graph Laplacian L = D - W
client signal G or W_local
denoised signal = h(L) signal
global model = conservative aggregation of denoised signals
```

Main shift:

```text
from:
  raw update-cosine graph + low-pass

to:
  semantic/head/temporal graph + model-space signal denoising
```

## 2. Why Raw Update Graph Was Not Enough

Initial implementation used:

```text
g_i = w_i - w_global
graph_source = full update delta g_i
```

This is a reasonable first proxy, but it mixes:

```text
label skew signal
optimizer noise
sample-count imbalance
local trajectory
round-specific instability
```

Observed issue:

```text
spectral-only with raw update graph did not reliably beat FedAvg.
random-matched graph sometimes looked competitive.
```

Interpretation:

```text
The idea is not disproven.
The weak version of the graph was tested first.
The main missing piece is graph construction.
```

## 3. Semantic Graph Candidates

### Level 1. Classifier Head Graph

Use final classifier layer update or weight.

```text
graph_source = classifier_head_update
graph_source = classifier_head_weight
```

Why:

```text
label skew and class preference are more directly expressed in the classifier head
than in the whole model update.
```

First variants:

```text
ours_head_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_head_weight_graph_knn_k1_lp0p5_fixed_tau_spectral_only
```

### Level 2. Layerwise Head Graph

Normalize head weight and bias separately.

```text
graph_source = layerwise_classifier_head_update
graph_source = layerwise_classifier_head_weight
```

Why:

```text
one tensor block should not dominate cosine similarity only because of scale.
```

### Level 3. Temporal Head/Update Graph

Smooth client updates over rounds before graph construction.

```text
u_i,t = beta * u_i,t-1 + (1-beta) * g_i,t
graph_source = ema_update
graph_source = classifier_head_ema_update
```

Why:

```text
single-round local updates are noisy.
server momentum helped before, so temporal denoising should be isolated.
```

### Level 4. Prototype / Proxy Behavior Graph

Not implemented yet.

Possible sources:

```text
class feature prototypes
proxy dataset softmax outputs
CKA over representations
KL divergence over prediction behavior
```

Why:

```text
these are closer to semantic knowledge state than raw parameter deltas.
```

## 4. Signal Denoising Directions

### A. Current Update Filtering

```text
G_filtered = h(L)G
w_next = w_global + sum_i alpha_i G_filtered_i
```

Status:

```text
implemented as spectral_filtered_update
```

### B. EMA Signal Filtering

```text
U_filtered = h(L)U
w_next = w_global + sum_i alpha_i U_filtered_i
```

Status:

```text
implemented as spectral_filtered_ema_update
```

Why:

```text
tests whether temporal denoising plus graph denoising can stabilize spectral-only.
```

### C. Weight-Space Filtering

```text
W_filtered = h(L)W_local
w_next = sum_i alpha_i W_filtered_i
```

Status:

```text
implemented as spectral_filtered_weight
```

Interpretation:

```text
fixed graph message passing over client model weights.
This should be framed as conservative model-space denoising,
not as a fully learnable GNN yet.
```

## 5. GNN Framing

The conservative framing:

```text
client = node
client update/weight = node feature
client graph = semantic or temporal relation
message passing = graph denoising
global model = aggregate denoised node features
```

Parameter-free first step:

```text
X' = h(L)X
```

Possible future step:

```text
X' = X + gamma * GNN_theta(X, W)
```

Do not overclaim:

```text
Full learnable GNN over weights needs a training signal.
Without proxy validation, it risks becoming an unconstrained hypernetwork.
```

## 6. Momentum Interaction

Three variants are worth separating.

| Variant | Meaning |
|---|---|
| EMA graph only | graph edges are temporally smoothed, current update is filtered |
| EMA signal | graph and filtering target use temporally smoothed update |
| EMA signal + serverM | temporal smoothing at client-signal level plus server optimizer |

Variant examples:

```text
ours_ema_graph_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_signal_knn_k1_lp0p5_fixed_tau_spectral_only
ours_ema_signal_knn_k1_lp0p5_serverm_fixed_tau
```

Interpretation:

```text
If EMA graph helps:
  graph construction was noisy.

If EMA signal helps:
  update signal itself needed temporal denoising.

If only serverM helps:
  previous gains were mostly optimizer dynamics.
```

## 7. Experiment Roadmap

### Stage 1. Smoke

Config:

```text
configs/general/smoke/semantic_ema_weight.json
```

Goal:

```text
all new graph_source and aggregation_target paths run without crashing.
```

### Stage 2. N=20 Stress

Compare:

```text
fedavg
fedavgm
fedadam
fedyogi
fednova
raw update graph
random-matched graph
head graph
EMA graph
EMA signal
weight filtering
```

### Stage 3. N=50 Stress

Question:

```text
Does semantic/temporal graph reduce the degradation seen with more clients?
```

### Stage 4. Multi-Seed Confirmation

Use at least:

```text
seeds = 42, 43, 44
```

Report:

```text
mean accuracy
min delta
win rate
delta vs FedAvg
delta vs FedAvgM
delta vs random graph
```

## 8. Paper Claim Boundary

Weak but honest current claim:

```text
Raw update graph spectral filtering is unstable.
However, server momentum interaction and collapse-regime behavior suggest
graph/temporal denoising is worth further testing.
```

Target claim:

```text
Semantic client graph construction enables model/update signal denoising
that is more robust than raw update averaging under label-skew FL.
```

ICML-level requirement:

```text
semantic/head/proxy graph > raw update graph
semantic/head/proxy graph > random-matched graph
method competitive with FedAvgM/FedOpt
effect holds at N=20 and N=50
multi-seed stability
```

## 9. Open Work

Not implemented yet:

```text
prototype graph
proxy prediction graph
CKA graph
learnable GNN denoiser
layer-specific filter strength
edge denoising / entropy threshold
```

Next implementation candidate:

```text
server-side proxy prediction graph
```

Why:

```text
It measures model behavior directly and can separate semantic similarity
from raw update noise.
```
