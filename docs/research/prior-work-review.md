# Prior Work Review

## 연구 위치

이 repository의 연구 위치는 Graph-FL gain attribution framework다.
핵심 novelty는 graph-specific explanation을 control과 diagnostic metric으로 검증하는 실행 protocol이다.

## Topic Judgment

| Question | Judgment |
|---|---|
| topic | Graph-FL gain attribution |
| novelty | matched control과 graph-free control을 통한 mechanism 분해 |
| research value | Graph-FL claim을 testable mechanism으로 변환 |
| implementation risk | exact/proxy/interface level 구분 필요 |
| expansion | privacy, robustness, fairness, scalability diagnostics |

## Relevant Family

| Family | Contribution | Framework Role |
|---|---|---|
| FedAMP-like | model-distance relation, attentive personalized aggregation | `weight + rbf + graph_filtered_weight` proxy |
| SFL-like | relation graph + server-side graph model | `weight + learned_smooth + graph_filtered_weight` proxy |
| pFedGraph-like | QP relation estimator, sample-size prior | `update + pfedgraph_qp + graph_filtered_update` proxy |
| FedAGA-like | accumulated-gradient relation, adaptive graph aggregation | `ema_update + magnitude_knn + graph_filtered_ema_update` proxy |
| FED-PUB/GPFL-like | functional embedding, client-specific delivery | interface-target |
| Hypergraph/attention/hypernetwork | rich topology and personalized operators | interface-target |

## Framework Gap

이 repository가 실행 가능하게 만드는 질문:

```text
meaningful relation edges vs generic smoothing
matched random/shuffled/identity graph controls
coarse clustering vs fine-grained edges
dominance/norm/contribution correction
source/topology/target attribution
```

## Evidence Vocabulary

| Term | 의미 |
|---|---|
| `exact-reference` | official implementation/version과 직접 비교 |
| `paper-kernel` | paper 수식/설명 기반 independent kernel |
| `proxy-reference` | paper mechanism에서 유도한 proxy |
| `interface-target` | framework slot은 정의, runnable path는 확장 대상 |
