# Diagnostic Metric Interpretation Guide

Use with `round_metrics.csv`, `client_metrics.csv`, and `graph_stats.csv`.

## 1. Notation

Pre/post metrics compare the raw aggregation path against the corrected or graph-filtered path.

Raw update:

$$
g_i
$$

Corrected update:

$$
\tilde g_i
$$

Pre-update:

$$
\Delta_{pre} = \sum_i p^{pre}_i g_i
$$

Post-update:

$$
\Delta_{post} = \sum_i p^{post}_i \tilde g_i
$$

Interpret pre/post metrics as both weight-change and update-vector-change diagnostics.

## 2. Metrics

Read each metric with its paired caveat; no single metric is causal evidence.

| Metric | Measures | Use | Caveat |
|---|---|---|---|
| `q_i` | $p_i\|g_i\|$ share | client physical influence | no direction information |
| `DI = max(q_i)` | max contribution share | dominance pathology | DI drop alone is not evidence |
| `N_eff = 1 / sum(q_i^2)` | effective contributing clients | contribution spread | higher can mean over-mixing |
| `alignment_i = cos(g_i, Delta)` | client/global direction match | aggregate representativeness | dominant client can inflate it |
| `LOO_i = 1 - cos(Delta, Delta_-i)` | single-client sensitivity | aggregation fragility | low LOO can erase minority signal |
| `update_norm_raw/corrected` | update size | vector scaling or clipping effect | norm change is not outcome evidence |
| `graph_density` | edge ratio | mixing opportunity | match density in controls |
| `graph_entropy` | edge weight spread | diffuse smoothing vs edge concentration | not relation quality |
| `alpha_entropy` | aggregation weight spread | weight balancing | uniform weights can help harmful clients |
| `accuracy/loss` | task outcome | final outcome | not causal evidence alone |

## 3. Mechanism Rules

Use these rules after variant comparisons are available.

### Fine-Grained Graph Relation

Evidence:

```text
real > shuffled/random/uniform/identity
real > density/entropy-matched controls
alignment_post improves
LOO_post stabilizes
graph-free dominance correction does not reproduce gain
real > cluster-only
```

Avoid:

```text
real > FedAvg => graph relation is valid
```

### Coarse Clustering

Evidence:

```text
cluster-only ≈ real > random/uniform
cluster_id-wise alignment/q pattern
cluster-only stabilizes DI and LOO
```

Conclusion:

```text
coarse grouping may explain gain without fine-grained edges
```

### Dominance Suppression

Evidence:

```text
DI_post < DI_pre
N_eff_post > N_eff_pre
top q_i clients have lower q_corrected and LOO_corrected
graph-free_normclip/graphfree_cap/graphfree_reweight ≈ real
performance tracks DI drop or N_eff gain more than graph metrics
```

Conclusion:

```text
gain may come from contribution correction, not relation quality
```

### Generic Smoothing / Mixing

Evidence:

```text
shuffled/random/uniform ≈ real
graph identity removal preserves alignment/LOO/DI changes
high graph_entropy with performance gain
```

Conclusion:

```text
mixing regularization may explain gain
```

### Over-Smoothing / Under-Correction

Over-smoothing:

```text
DI down, N_eff up, accuracy/loss not improved
update_norm_corrected sharply down
alignment_post down
real and uniform both hurt performance
```

Under-correction:

```text
pre/post DI, N_eff, alignment, LOO nearly unchanged
baseline performance unchanged
```

## 4. Report Language

Report language should name both the outcome and the control that failed or succeeded.

Allowed:

```text
ours_real_graph_k2는 FedAvg보다 높았지만, ours_graphfree_reweight와 유사한 DI 감소와 N_eff 증가를 보였다.
```

```text
real graph가 density/entropy-matched shuffled control보다 높은 accuracy와 alignment_post를 보였고, graph-free correction은 재현하지 못했다.
```

Avoid:

```text
real graph가 FedAvg보다 높으므로 graph relation이 유효하다.
DI가 낮아졌으므로 성능이 좋아진 원인은 dominance suppression이다.
```

## 5. Minimum Report Unit

```text
mean_delta_vs_fedavg
mean_di_drop = mean(DI_pre - DI_post)
mean_neff_gain = mean(N_eff_post - N_eff_pre)
mean_alignment_gain = mean(alignment_post - alignment_pre)
mean_loo_drop = mean(loo_pre - loo_post)
mean_graph_density
mean_graph_entropy
graph-free gap
control graph gap
```
