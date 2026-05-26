# Graph-FL Design Pattern Survey

## 목적

Graph-FL/PFL 논문의 mechanism을 framework lifecycle slot으로 분류한다.

## Design Axis

```text
client_state
relation_estimator
topology_operator
aggregation_target
delivery / state / local_objective
diagnostics / counterfactual controls
```

## Method Mapping

| Method | Client State | Relation | Topology | Aggregation | Personalized Component | Support |
|---|---|---|---|---|---|---|
| FedAMP | model weights | distance/RBF/attention | dense weighted graph | personalized weight mixture | cloud model + proximal | proxy-supported |
| FedFomo | local models + validation utility | validation utility | directed top-M | weight mixture | collaborator selection | proxy-supported |
| APPLE | core models | learned relationship vector | directed sparse/full graph | client-side weight mixture | download budget | proxy-supported |
| SFL | model/client relation graph | learned relation | client-wise graph | personalized graph sharing | graph regularization | proxy-supported |
| pFedGraph | personalized models + sample prior | cosine + dataset-size prior/QP | row-stochastic graph | weight mixture | local regularization | proxy-supported |
| GCFL | gradient sequence | gradient norm/DTW | cluster block graph | cluster FedAvg | cluster model | proxy-supported |
| FED-PUB | functional embedding | functional similarity | community graph | weight averaging + mask | sparse mask | interface-target |
| GPFL | marginal parameters + graph descriptor | graph autoencoder score | reconstructed sparse graph | GNN-guided update aggregation | dynamic client network | interface-target |
| pFedGAT | model parameters | GAT attention | learned dynamic graph | personalized mixture | loss feedback | interface-target |
| FedAGHN | params/updates | graph hypernetwork | layer-wise graph | generated personalized weights | hypernetwork | interface-target |

## 사용 기준

| 목적 | 사용 방식 |
|---|---|
| lifecycle 분류 | method를 slot 조합으로 표현 |
| support level | `core-supported`, `proxy-supported`, `interface-target` 지정 |
| trace 설계 | 어떤 module trace가 필요한지 결정 |
| evidence 설계 | exact-reference, paper-kernel, proxy-reference 구분 |
