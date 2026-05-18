# Graph-FL Lifecycle Intervention Framework

## 1. Lifecycle

Lifecycle modules define where a graph-FL method intervenes in a round.

```text
Round t start
-> decide model/state delivered to each client
-> optional local objective hook
-> client local training
-> extract client state
-> estimate client-client relation
-> construct graph/topology
-> apply graph-conditioned aggregation or personalization
-> record diagnostics and counterfactual traces
-> store state for round t+1
```

Design unit:

```text
GraphFLDesign
-> lifecycle modules
-> standard trace
-> counterfactual diagnostics
```

Execution knobs:

```text
graph_source
graph_mode
aggregation_target
```

## 2. Lifecycle Modules

Use this table to map a paper method into implementation slots.

| Module | Point | Role | Examples |
|---|---|---|---|
| `DeliveryPolicy` | before client training | delivered model/state | global model, previous personalized model |
| `LocalObjectiveHook` | local training | graph/personalization loss term | FedAMP proximal term, pFedGraph cluster regularization |
| `ClientStateExtractor` | during/after local training | graph representation | update, weight, EMA update, classifier head, functional embedding |
| `RelationEstimator` | before aggregation | client-client score | cosine, RBF, attention, QP, learned metric |
| `TopologyOperator` | after relation score | graph/topology | dense, kNN, threshold, row-stochastic, clustering-only, control |
| `AggregationOperator` | server aggregation | graph-conditioned update/model | graph-filtered update, weight smoothing, client-wise mixture |
| `StateStore` | round boundary | persistent state | EMA graph, accumulated gradient, previous relation |
| `DiagnosticProtocol` | all stages | traces and counterfactuals | DI, N_eff, alignment, LOO, graph entropy |

## 3. Prior-Work Mapping

| Method | Lifecycle combination | Status |
|---|---|---|
| FedAMP | personalized delivery, weight state, attentive distance kernel, client-specific aggregate, proximal hook | relation/weight proxy; exact delivery target |
| pFedGraph | update + sample size, cosine-difference QP, row-stochastic graph, neighbor model mixture | QP relation implemented; personalized mixture target |
| FedAGA | accumulated gradient, gradient similarity, dynamic graph, adaptive timing | EMA update proxy |
| SFL | model state, client relation graph, server GCN personalized model | graph-filter proxy; server GCN target |
| FED-PUB / GPFL | functional embedding, embedding cosine, personalized aggregation, mask hook | source plugin target; personalized target missing |

Support levels:

```text
core-supported
proxy-supported
interface-target
out-of-scope
```

## 4. Standard Trace Schema

Trace fields are the bridge between implementation components and diagnostic claims.

| Module | Trace | Question |
|---|---|---|
| `DeliveryPolicy` | delivered model id/type, global-personalized distance, per-client delivery norm | delivery state difference |
| `LocalObjectiveHook` | hook loss, hook/base ratio, hook gradient norm | local hook effect |
| `ClientStateExtractor` | state norm, layer/head norm, cosine distribution, sample-size prior | graph material |
| `RelationEstimator` | relation matrix stats, pairwise score distribution, relation entropy | informative relation |
| `TopologyOperator` | density, degree stats, graph entropy, components, row entropy | topology shape |
| `AggregationOperator` | alpha, q_i, alpha entropy, pre/post aggregate, client contribution | dominance and participation |
| `StateStore` | EMA/history norm, previous graph distance, state drift | memory effect |
| `DiagnosticProtocol` | DI, N_eff, alignment, LOO, matched control deltas | graph vs control separation |

Derived diagnostics:

| Value | Source |
|---|---|
| update size and dominance | `ClientStateExtractor`, `AggregationOperator` |
| dominant client | `q_i`, `DI`, `N_eff`, `alpha_entropy` |
| relation information | `RelationEstimator`, `TopologyOperator`, matched controls |
| smoothing effect | uniform/identity graph, graph-free correction |
| clustering effect | clustering-only counterfactual |
| fine-grained edge effect | real graph vs clustering-only |
| sample-size prior effect | pFedGraph prior on/off |
| direction improvement | alignment pre/post |
| single-client sensitivity | LOO pre/post |
| dense/sparse graph | density, degree, entropy |
| personalized aggregation | row entropy, delivery distance, mixture drift |
| local hook effect | hook loss, hook gradient norm |

## 5. Counterfactual Diagnostic Runner

The actual training path can be single, but diagnostic paths must reuse the same client artifacts.

Actual path:

```text
real graph -> aggregation -> model update
```

Shadow paths:

```text
same client states -> matched random graph -> diagnostics
same client states -> shuffled graph -> diagnostics
same client states -> uniform graph -> diagnostics
same client states -> identity graph -> diagnostics
same client states -> clustering-only graph -> diagnostics
same client states -> graph-free correction -> diagnostics
```

Questions:

```text
real graph vs matched control
norm reduction vs relation effect
coarse clustering sufficiency
fine-grained edge contribution
dominance correction sufficiency
sample-size prior effect
```

## 6. Framework Claim

```text
Graph-based FL methods can be decomposed into lifecycle-level intervention modules.
Standard traces and counterfactual diagnostics identify whether gains come from
relation information, topology, smoothing, clustering, sample-size prior,
dominance suppression, or local objective effects.
```

## 7. Implementation Principles

1. Return output and trace from each module.
2. Use common trace keys.
3. Separate actual path and shadow diagnostic path.
4. Mark hard methods as `proxy-supported` or `interface-target`.
5. Keep `graph_source`, `graph_mode`, `aggregation_target` as execution knobs.
6. Define hook location and trace output before adding a method.

Current priorities:

```text
graph_fl_experimental_design.md
cleanup-plan.md
```
