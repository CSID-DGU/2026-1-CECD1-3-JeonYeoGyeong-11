# Graph-FL Design Pattern Survey

Normative design reference, not implementation backlog.

## Use

```text
classify lifecycle role
assign support level
choose trace keys
check design-space vocabulary
```

Do not use this table to claim exact reproduction.

## Design Axes

```text
client state
relation estimation
topology construction
aggregation target
delivery / state / local objective
diagnostics and counterfactual controls
```

## Method Mapping

| Method | Client state | Relation | Topology | Aggregation | Personalized component | Framework support |
|---|---|---|---|---|---|---|
| FedAMP | model weights | distance/RBF/attentive relation | dense weighted graph | personalized weight mixture | cloud model + proximal | proxy-supported |
| FedFomo | local models + validation utility | validation utility | directed top-M | weight mixture | collaborator selection | proxy-supported |
| APPLE | core models | learned relationship vector | directed sparse/full graph | client-side weight mixture | download budget | proxy-supported |
| SFL | model/client relation graph | given or learned relation | client-wise graph | personalized graph sharing | graph regularization | proxy-supported |
| pFedGraph | personalized models + sample prior | cosine + dataset-size prior/QP | row-stochastic collaboration graph | weight mixture | local regularization | proxy-supported |
| GCFL | gradients/gradient sequence | gradient norm/DTW | cluster block graph | cluster-wise FedAvg | cluster model | proxy-supported |
| FedCCH | hash signature + local factor | Hamming similarity | cluster graph | cluster-wise aggregation | intra-client personalization | proxy-supported |
| FED-PUB | functional embedding | functional similarity | community graph | weight averaging + mask | sparse mask | interface-target |
| FedGTA | mixed moments + confidence | moment cosine | threshold aggregation set | weighted personalized aggregation | confidence weighting | proxy-supported |
| GPFedRec | item embeddings | cosine | adaptive threshold graph | GCN-guided embedding aggregation | user-specific embedding | proxy-supported |
| GPFL | marginal parameters + graph descriptors | graph autoencoder score | reconstructed sparse graph | GNN-guided update aggregation | dynamic client network | interface-target |
| pFedGAT | model parameters | GAT attention | learned dynamic graph | personalized mixture | loss feedback | interface-target |
| FedAGHN | params/updates | attentive graph hypernetwork | layer-wise graph | generated personalized weights | hypernetwork | interface-target or out-of-scope |
| FedSheafHN | graph embeddings | collaboration graph | sheaf diffusion graph | hypernetwork parameters | generated model | interface-target or out-of-scope |
| ADPFedGNN | masked global/local params | local graph neighbor relation | masks/neighbor graph | masked aggregation | MI loss, masks | interface-target |

## Framework Implication

Prioritize:

```text
flexible client-state envelope
relation/topology separation
global alpha vs personalized alpha_matrix
explicit support levels
same-round counterfactual diagnostics
component-aware ablations
```

## Support-Level Policy

| Level | Meaning |
|---|---|
| `core-supported` | behavior executes as framework core |
| `proxy-supported` | design pattern preserved through simplified executable proxy |
| `interface-target` | lifecycle boundary and trace contract exist; full method missing |
| `out-of-scope` | requires machinery outside current framework scope |

Rule:

```text
proxy-supported != exact reproduction
```
