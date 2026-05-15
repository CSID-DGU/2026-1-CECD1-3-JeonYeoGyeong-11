# Graph-FL Design Pattern Survey

## Agent Usage

Use this document as normative design reference, not as an implementation backlog.

- Use it to classify a method's lifecycle role and support level.
- Use it to choose trace keys and design-space vocabulary.
- Use it to ensure the phase implementation can represent the method family at the declared support level.
- Do not treat the method table as an implementation checklist.
- Do not implement exact prior-work algorithms from this document unless the active phase explicitly asks for that scope.
- When this survey conflicts with a phase document, follow the phase document.

## Purpose

This document summarizes how existing graph-based FL and personalized graph-FL methods construct and use graphs.

The goal is not exact reproduction. The goal is to identify common design axes that the lifecycle framework should support.

The mapping table is a design-pattern mapping, not a claim of exact reproduction.

## Design Axes

1. Client state
2. Relation estimation
3. Topology construction
4. Aggregation target
5. Delivery / state / local objective
6. Diagnostics and counterfactual controls

## Method Mapping Table

| Method | Client state | Relation | Topology | Aggregation | Personalized component | Framework support |
|---|---|---|---|---|---|---|
| FedAMP | model weights | distance / RBF / attentive relation | dense weighted graph | personalized weight mixture | cloud model + proximal | proxy-supported |
| FedFomo | local models + validation utility | validation utility | directed top-M | weight mixture | validation-based collaborator selection | proxy-supported |
| APPLE | core models | learned directed relationship vector | directed sparse/full graph | client-side weight mixture | download budget | proxy-supported |
| SFL | model / client relation graph | given or learned relation | client-wise graph | personalized graph sharing | graph regularization | proxy-supported |
| pFedGraph | personalized models + sample prior | cosine + dataset-size prior / QP | row-stochastic collaboration graph | weight mixture | local regularization | proxy-supported |
| GCFL | gradients / gradient sequence | gradient norm / DTW | cluster block graph | cluster-wise FedAvg | cluster model | proxy-supported |
| FedCCH | hash signature + local personalization factor | Hamming similarity | cluster graph | cluster-wise aggregation | intra-client personalization | proxy-supported |
| FED-PUB | functional embedding | functional similarity | community graph | weight averaging + mask | sparse mask | interface-target |
| FedGTA | mixed moments + confidence | cosine of moments | threshold aggregation set | weighted personalized aggregation | confidence weighting | proxy-supported |
| GPFedRec | item embeddings | cosine | adaptive threshold graph | GCN-guided embedding aggregation | user-specific embedding | proxy-supported |
| GPFL | marginal parameters + graph descriptors | graph autoencoder score | reconstructed sparse graph | GNN-guided update aggregation | dynamic client network | interface-target |
| pFedGAT | model parameters | GAT attention | learned dynamic graph | personalized mixture | loss feedback | interface-target |
| FedAGHN | params / updates | attentive graph hypernetwork | layer-wise graph | generated personalized weights | hypernetwork | interface-target or out-of-scope |
| FedSheafHN | graph embeddings | collaboration graph | sheaf diffusion graph | hypernetwork parameters | generated model | interface-target or out-of-scope |
| ADPFedGNN | masked global/local params | local graph neighbor relation | masks / neighbor graph | masked aggregation | MI loss, masks | interface-target |

## Implication For The Framework

Most methods can be expressed by replacing one or more lifecycle components.

The framework should prioritize:

- flexible client-state envelope
- relation/topology separation
- global `alpha` vs personalized `alpha_matrix` distinction
- explicit support levels
- same-round counterfactual diagnostics
- component-aware ablations

## Support-Level Policy

Prior-work-inspired presets should declare their support level explicitly.

- `core-supported`: the implementation executes the behavior as part of the framework core.
- `proxy-supported`: the implementation preserves the design pattern through a simplified executable proxy.
- `interface-target`: the lifecycle boundary and trace contract exist, but the full method is not implemented.
- `out-of-scope`: the method requires machinery outside the current framework scope.

Proxy-supported presets must not be presented as exact reproductions of the original algorithms.
