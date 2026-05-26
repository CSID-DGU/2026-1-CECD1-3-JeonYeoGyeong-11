# Graph-FL Components

## Lifecycle

```text
round start
-> delivery policy
-> local objective hook
-> client local training
-> client state extraction
-> relation estimation
-> topology construction
-> graph-conditioned aggregation
-> diagnostics and counterfactual traces
-> state store
```

## Interface Map

| Layer | Interface | 위치 |
|---|---|---|
| method design | `GraphFLDesign`, `ComponentSpec` | `graphfl_lab/designs/` |
| client state | `register_graph_source`, `GraphSourceResult` | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| relation estimator | `register_graph_builder`, `GraphBuildContext` | `graphfl_lab/graph/registry.py` |
| topology operator | graph builder, sparsification helper | `graphfl_lab/graph/sparsification.py`, `graphfl_lab/graph/controls.py` |
| aggregation operator | `aggregation_target` | `graphfl_lab/strategies/graphfl/targets.py` |
| runtime strategy | `GraphFLDiagnosticStrategy` | `graphfl_lab/strategies/graphfl/strategy.py` |
| diagnostics | metric and artifact writer | `graphfl_lab/diagnostics/` |
| suite grammar | variant parser | `graphfl_lab/experiments/suites/vision/variants.py` |

## GraphFLDesign Slot

Required slots:

```text
client_state
relation
topology
aggregation
```

Extension slots:

```text
delivery
local_objective
state_store
diagnostics
```

## Component Contract

| Component | Contract |
|---|---|
| `graph_source` | client order 유지, fixed-length vector, finite numeric value, metadata |
| `graph_builder` | `(num_clients, num_clients)` adjacency, finite non-negative weight, zero diagonal |
| `aggregation_target` | output shape 유지, finite value, target metadata |
| `GraphFLDesign` | source/mode/target/preset metadata와 support level 기록 |
| artifact | trace, diagnostics, graph stats, evidence row 기록 |

## Built-In Preset

| Preset | Method Profile | Runnable Knobs | Support |
|---|---|---|---|
| `fedamp_proxy` | FedAMP attentive message passing | `weight + rbf + graph_filtered_weight` | proxy-supported |
| `sfl_proxy` | SFL graph-structured aggregation | `weight + learned_smooth + graph_filtered_weight` | proxy-supported |
| `pfedgraph_proxy` | pFedGraph inferred collaboration graph | `update + pfedgraph_qp + graph_filtered_update` | proxy-supported |
| `ema_magnitude_knn_filtered` | FedAGA accumulated-gradient graph | `ema_update + magnitude_knn + graph_filtered_ema_update` | proxy-supported |
| `fedpub_like` | functional embedding + personalized aggregation | interface target | interface-target |

## Prior Work Mapping

| Method | Paper Mechanism | Current Mapping | Support |
|---|---|---|---|
| FedAMP | model-distance relation, attentive message passing, proximal update | `fedamp_proxy` | proxy-supported |
| SFL | client relation graph, server GCN, personalized model | `sfl_proxy` | proxy-supported |
| pFedGraph | cosine difference, sample-size prior, simplex QP, neighbor mixture | `pfedgraph_proxy` | proxy-supported |
| FedAGA | accumulated-gradient similarity, dynamic graph, adaptive aggregation | `fedaga_like` | proxy-supported |
| FED-PUB / GPFL | functional embedding, client-specific aggregation | functional embedding source + personalized target | interface-target |
| pFedGAT / FedAGHN / FedHyperGraph | attention, hypernetwork, hypergraph personalized operator | personalized aggregation operator | interface-target |

## Support Vocabulary

| Level | 의미 |
|---|---|
| `core-supported` | framework component로 실행과 diagnostics 가능 |
| `proxy-supported` | 핵심 mechanism을 diagnostic path로 투영 |
| `paper-kernel` | paper 수식/설명 기반 independent kernel |
| `exact-reference` | source URL과 commit/version이 있는 official reference 비교 |
| `interface-target` | hook 위치와 설계 slot 정의 |

## Extension Workflow

1. Method profile을 `client_state`, `relation`, `topology`, `aggregation` slot으로 작성한다.
2. 기존 `graph_source`, `graph_mode`, `aggregation_target` 재사용 범위를 확인한다.
3. 새 representation이면 `graph_source`를 추가한다.
4. 새 relation/topology면 `graph_builder`를 추가한다.
5. invalid 조합은 `require_graph_context(...)`로 분류한다.
6. `GraphFLDesign` preset을 등록한다.
7. diagnostics와 artifact field를 추가한다.
8. deterministic adjacency, metadata, graph stats, control path test를 추가한다.
