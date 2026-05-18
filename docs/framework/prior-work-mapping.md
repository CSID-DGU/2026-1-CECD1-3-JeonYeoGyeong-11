# Prior Graph-FL Implementation Mapping

## Support Vocabulary

Support level separates runnable proxies from exact paper reproduction.

| Level | Meaning |
|---|---|
| `core-supported` | current framework components cover execution and diagnostics |
| `proxy-supported` | key mechanism projected into global-model diagnostic path; not exact reproduction |
| `interface-target` | hook location known; runnable component missing |
| `out-of-scope` | requires separate system beyond aggregation-level framework |

Compatibility aliases:

```text
fedamp_like -> fedamp_proxy
sfl_like -> sfl_proxy
pfedgraph_like -> pfedgraph_proxy
```

## Method Profile Slots

| Slot | Meaning |
|---|---|
| `client_state` | update, weight, gradient history, functional embedding |
| `relation_estimator` | kNN, RBF, QP, attention, functional embedding cosine |
| `topology_operator` | dense, sparse, row-stochastic, directed, hypergraph |
| `aggregation_operator` | global smoothing, client-specific mixture, server GCN |
| `personalization_site` | server, client, local objective |
| `local_objective_hook` | proximal, regularization, mask term |
| `diagnostic_projection` | exact, proxy, future interface |

## Current Mapping

Current mappings expose comparable diagnostic paths; exact personalized algorithms need additional targets.

| Method family | Paper-level behavior | Current mapping | Support |
|---|---|---|---|
| FedAMP | personalized model distance, attentive message passing, proximal local update | `fedamp_proxy`: `weight + rbf + graph_filtered_weight` | proxy |
| SFL | client relation graph into server GCN for client-specific models | `sfl_proxy`: `weight + learned_smooth + graph_filtered_weight` | proxy |
| pFedGraph | cosine difference + dataset-size prior + row-wise simplex QP + neighbor model mixture | `pfedgraph_proxy`: `update + pfedgraph_qp + graph_filtered_update` | proxy |
| FedAGA | accumulated gradient similarity, dynamic graph, adaptive aggregation timing | `fedaga_like`: `ema_update + magnitude_knn + graph_filtered_ema_update` | proxy |
| FED-PUB / GPFL | functional embedding from proxy graph, personalized aggregation | needs functional embedding source + `personalized_weight` target | interface target |
| pFedGAT / FedAGHN / FedHyperGraph | attention/hypernetwork/hypergraph personalized operators | needs personalized aggregation operator | interface target |

## pFedGraph Boundary

Implemented proxy:

```text
local/global update difference
cosine-difference relation
sample-size prior
row-wise simplex QP relation
diagnostic graph projection
```

Missing exact path:

```text
row-wise personalized model delivery
client-specific neighbor model mixture
cluster model regularization
```

## FED-PUB Boundary

Missing components:

```text
functional_embedding graph source
personalized_weight aggregation target
client-specific delivery
mask/previous-model local hook
```

## Interpretation Rule

Every prior-work-inspired preset needs the same control comparisons as native methods.

| Comparison | Question |
|---|---|
| real vs matched random/shuffled/uniform/identity | relation information |
| real vs clustering-only | fine-grained edge vs coarse group |
| real vs graph-free norm/cap/reweight | dominance/magnitude correction |
| pFedGraph-like vs learned_smooth/RBF | QP and sample-size prior effect |
| functional embedding source vs update/weight source | representation source effect |

Preferred spelling:

```text
graph_filtered_*
```

Compatibility spelling:

```text
spectral_filtered_*
```
