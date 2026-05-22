# Prior Work Review

Lens: what mechanism can the framework diagnose, not what paper is exactly reproduced.

## Topic Judgment

| Question | Judgment |
|---|---|
| Topic | diagnostic framework for graph-FL gain attribution |
| Novelty | graph gain attribution and control design over leaderboard comparison |
| Research value | converts graph-FL claims into testable mechanisms |
| Implementation risk | requires explicit proxy/support level |
| Expansion | privacy, robustness, fairness, scalability diagnostics |

Novelty wording:

```text
Semantic relation vs generic smoothing is not claimed as a new phenomenon by itself.
The project claim is the executable attribution protocol: matched controls,
graph-free controls, and diagnostics that test whether a graph-specific
explanation survives.
```

## Relevant Families

| Family | Contribution | Framework role |
|---|---|---|
| FedAMP-like | model-distance relation, attentive personalized aggregation | `weight + rbf + graph_filtered_weight` proxy |
| SFL-like | relation graph + server-side graph model | `weight + learned_smooth + graph_filtered_weight` proxy |
| pFedGraph-like | QP relation estimator with sample-size prior | `update + pfedgraph_qp + graph_filtered_update` proxy |
| FedAGA-like | accumulated-gradient relation, adaptive graph aggregation | `ema_update + magnitude_knn + graph_filtered_ema_update` proxy |
| FED-PUB/GPFL-like | functional embedding, client-specific delivery | `interface-target`; needs source and personalized target |
| Hypergraph/attention/hypernetwork | rich topology and personalized operators | mostly `interface-target` |

## Gap In Prior Work

Questions this repo makes runnable:

```text
meaningful relation edges vs generic smoothing
matched random/shuffled/identity graph controls
coarse clustering vs fine-grained edges
dominance/norm/contribution correction
source/topology/target attribution
```

## Claim Boundary

Exact reproduction requires matching:

```text
client state
relation estimator
topology operator
aggregation/delivery target
local objective hook
personalization semantics
state carried across rounds
diagnostics and evaluation protocol
```

Partial match:

```text
proxy-supported
```

## Required Controls

| Control | Purpose |
|---|---|
| matched random graph | edge count/weight distribution |
| shuffled graph | client identity assignment |
| identity graph | no cross-client mixing |
| uniform graph | relation-free averaging |
| clustering-only graph | coarse groups |
| graph-free norm/cap/reweight | dominance or magnitude correction |

## Implementation Implication

Start from:

```text
docs/framework/prior-work-mapping.md
docs/framework/extension-guide.md
```

Default workflow:

```text
1. Select or add GraphFLDesign.
2. Add graph source only for new client state.
3. Add graph builder only for new relation/topology.
4. Prefer graph_filtered_* targets.
5. Keep spectral_filtered_* only for compatibility.
6. Add controls and diagnostics before interpreting performance.
```
