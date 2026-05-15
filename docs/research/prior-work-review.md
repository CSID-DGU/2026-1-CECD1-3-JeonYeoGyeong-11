# Prior Work Review

이 문서는 graph-FL 선행연구를 “우리가 무엇을 exact하게 구현했는가”가 아니라 “현재 framework가 어떤 mechanism을 진단할 수 있는가”라는 관점으로 정리한다.

## Topic Judgment

현재 주제는 새 graph aggregation 알고리즘 하나를 제안하는 것이 아니다. 주제는 **graph-based aggregation이 좋아 보일 때 그 이득이 relation graph 때문인지, smoothing 때문인지, clustering 때문인지, dominance correction 때문인지 분해하는 diagnostic framework**다.

이 방향의 장점은 명확하다.

| Question | Judgment |
|---|---|
| Novelty | 높음. 성능 경쟁보다 graph gain attribution과 control design을 전면에 둔다. |
| Research value | 높음. graph-FL 계열의 과장된 claim을 해석 가능한 실험 구조로 낮춘다. |
| Implementation risk | 중간. prior-work exact reproduction을 주장하지 않고 proxy/support level을 명시해야 한다. |
| Expansion | 좋음. privacy, robustness, fairness, scalability 진단으로 확장 가능하다. |

## Directly Relevant Families

| Family | What it contributes | Current framework role |
|---|---|---|
| FedAMP-like methods | Model-distance relation and attentive personalized aggregation | `weight + rbf + graph_filtered_weight` proxy |
| SFL-like methods | Relation graph plus server-side graph model/personalized sharing | `weight + learned_smooth + graph_filtered_weight` proxy |
| pFedGraph-like methods | QP relation estimator with sample-size prior | `update + pfedgraph_qp + graph_filtered_update` proxy |
| FedAGA-like methods | Accumulated-gradient relation and adaptive graph aggregation | `ema_update + magnitude_knn + graph_filtered_ema_update` proxy |
| FED-PUB/GPFL-like methods | Functional embedding and client-specific personalized model delivery | `interface-target`; needs new source and personalized target |
| Hypergraph/attention/hypernetwork methods | Rich topology and personalized operators | mostly `interface-target` until delivery/target interfaces expand |

## Gap In Prior Work

Most graph-FL papers answer “does this method outperform a baseline?” They less often answer:

- Is the gain from meaningful relation edges or just generic smoothing?
- Would a matched random/shuffled/identity graph produce the same effect?
- Is coarse clustering enough, or are fine-grained edges needed?
- Is the real effect dominance suppression or norm/contribution correction?
- Does the method still help when source, topology, and aggregation target are swapped independently?

This repository is designed to make those questions runnable.

## Claim Boundary

When using a prior-work-inspired preset, do not claim exact reproduction unless all of these match the paper:

1. Client state
2. Relation estimator
3. Topology operator
4. Aggregation/delivery target
5. Local objective hook
6. Personalization semantics
7. State carried across rounds
8. Diagnostics and evaluation protocol

If only some of those match, call it a proxy and record the support level in `GraphFLDesign` metadata.

## Required Controls

A prior-work proxy is incomplete without controls.

| Control | Purpose |
|---|---|
| matched random graph | separates edge count/weight distribution from relation information |
| shuffled graph | preserves graph statistics while breaking client identity |
| identity graph | tests whether no cross-client mixing is enough |
| uniform graph | tests whether relation-free averaging explains the effect |
| clustering-only graph | tests whether coarse groups explain the effect |
| graph-free norm/cap/reweight | tests whether dominance or magnitude correction explains the effect |

## Implementation Implication

New prior-work work should start from `docs/framework/prior-work-mapping.md` and `docs/framework/extension-guide.md`. The default path is:

1. Add or select a `GraphFLDesign` profile.
2. Add a graph source only when the client state is new.
3. Add a graph builder only when relation/topology is new.
4. Prefer `graph_filtered_*` aggregation targets in new commands.
5. Keep `spectral_filtered_*` only as compatibility spelling for old configs/results.
6. Add controls and diagnostics before interpreting performance.
