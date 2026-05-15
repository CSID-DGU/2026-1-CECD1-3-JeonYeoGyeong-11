# Graph Algorithm Extension Guide

새 graph algorithm은 `strategy.py`를 직접 크게 고치지 않고 붙일 수 있어야 한다. 기준은 단순한 `source/mode/target` 세 단어가 아니라, graph-FL 방법을 lifecycle component로 읽고 필요한 component만 교체하는 것이다.

## Method Profile

새 방법을 추가하기 전에 아래 항목을 먼저 적는다.

| Slot | Question |
|---|---|
| `client_state` | server가 client를 어떤 state/vector로 표현하는가? |
| `relation_estimator` | 그 state에서 edge/relation을 어떻게 추정하는가? |
| `topology_operator` | relation을 dense, sparse, row-stochastic, directed, hypergraph 등 어떤 topology로 만드는가? |
| `aggregation_operator` | relation을 update, weight, personalized model generation 중 어디에 적용하는가? |
| `personalization_site` | personalization이 server, client, local objective 중 어디에서 생기는가? |
| `local_objective_hook` | local loss에 proximal, mask, cluster-model regularization 같은 항이 붙는가? |
| `diagnostic_projection` | 현재 코드는 exact implementation, diagnostic proxy, future interface 중 무엇인가? |

`graph_method`는 runnable method/profile을 고르는 상위 CLI이고,
`graph_source`, `graph_mode`, `aggregation_target`은 이 profile을 실행 가능한
하위 knob로 옮긴 것이다. 예를 들어 pFedGraph는 `--graph-method pfedgraph`
또는 `update + pfedgraph_qp + graph_filtered_update`로 붙일 수 있지만, 이것은
pFedGraph의 QP relation estimator를 diagnostic graph로 투영한 것이다.
pFedGraph 전체 personalized algorithm을 exact reproduction한 것은 아니다.

## Built-In Presets

| Preset | Method profile | Runnable knobs | Support |
|---|---|---|---|
| `fedamp_like` -> `fedamp_proxy` | FedAMP attentive message passing | `weight + rbf + graph_filtered_weight` | `proxy-supported`; personalized cloud model delivery is not exact |
| `sfl_like` -> `sfl_proxy` | SFL graph-structured aggregation | `weight + learned_smooth + graph_filtered_weight` | `proxy-supported`; server GCN aggregation is not exact |
| `pfedgraph_like` -> `pfedgraph_proxy` | pFedGraph inferred collaboration graph | `update + pfedgraph_qp + graph_filtered_update` | `proxy-supported`; sample-size prior + simplex QP relation proxy |
| `fedaga_like` -> `ema_magnitude_knn_filtered` | FedAGA accumulated-gradient graph | `ema_update + magnitude_knn + graph_filtered_ema_update` | `proxy-supported`; accumulated gradient relation uses EMA proxy |

`fedpub_like`은 아직 runnable preset이 아니다. FED-PUB은 functional embedding source와 client별 personalized model target이 필요하므로 현재는 `interface-target`이다.

## Graph Source Plugin

새 client-state가 필요하면 `spectral_fl.graph.register_graph_source`로 `--graph-source`를 등록한다. FED-PUB/GPFL류의 functional embedding, dataset-stat embedding, gradient-history embedding은 이 지점에 붙인다.

```python
from spectral_fl.graph import GraphSourceResult, register_graph_source
from spectral_fl.projection import flatten_weights


@register_graph_source("my_functional_embedding")
def build_my_functional_embedding(context):
    vectors = []
    for local_weights in context.local_weights:
        vectors.append(flatten_weights(local_weights))
    return GraphSourceResult(
        vectors=vectors,
        source_used="my_functional_embedding",
        metadata={"state": "functional_embedding"},
    )
```

## Graph Builder Plugin

새 relation estimator나 topology가 필요하면 `spectral_fl.graph.register_graph_builder`로 `--graph-mode`를 등록한다.

```python
import numpy as np

from spectral_fl.graph import (
    GraphBuildContext,
    register_graph_builder,
    require_graph_context,
)


@register_graph_builder("my_relation_graph")
def build_my_relation_graph(context: GraphBuildContext):
    require_graph_context(
        context,
        graph_sources=("classifier_head_update",),
        aggregation_targets=("graph_filtered_update",),
    )

    z = context.z_mat
    adj = np.maximum(z @ z.T, 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj, {
        "base_graph_kind": "my_relation_graph",
        "method_profile": "my_method",
    }
```

Run it with:

```bash
python run_vision_experiment.py \
  --method ours \
  --graph-method my_method \
  --graph-plugin my_project.graph_plugins.my_method \
  --graph-source classifier_head_update \
  --graph-mode my_relation_graph \
  --aggregation-target graph_filtered_update
```

## GraphBuildContext

Builder가 받는 context는 다음 필드를 가진다.

| Field | Meaning |
|---|---|
| `z_mat` | `graph_source`가 만든 client representation matrix |
| `mode` | 현재 `--graph-mode` |
| `graph_source` | 실제 graph source 이름 |
| `aggregation_target` | graph가 적용될 aggregation target |
| `correction_family` | real/control/clustering/graph-free family |
| `knn_k` | kNN 계열 공통 knob |
| `edge_threshold` | threshold 계열 공통 knob |
| `rng` | round별 deterministic RNG |
| `graph_scale_sigma` | RBF/magnitude/custom scale knob |
| `learned_graph_lambda` | smooth/QP/custom regularization knob |
| `extras["client_sample_weights"]` | round 참여 client의 sample-size 비율. pFedGraph prior에 사용 |

특정 source/target에서만 의미 있는 algorithm이라면 `require_graph_context(...)`로 제한한다. 그렇지 않으면 같은 algorithm 이름이 의미 없는 source/target 조합에서도 실행되어 결과 해석이 흐려진다.

## Source/Mode/Target 해석

| Combination | Interpretation |
|---|---|
| `update + graph_filtered_update` | full client update relation으로 update 자체를 smoothing/filtering |
| `classifier_head_update + graph_filtered_update` | task-relevant head signal로 relation을 만들고 full update correction |
| `ema_update + graph_filtered_ema_update` | round noise를 줄인 temporal relation과 EMA update aggregation |
| `weight + graph_filtered_weight` | local model state similarity 기반 model smoothing |
| `update + pfedgraph_qp + graph_filtered_update` | pFedGraph식 sample-size prior와 cosine-difference QP를 diagnostic graph로 사용 |
| `graph_free correction + update` | graph relation 없이 dominance/magnitude control만 적용 |

Older `spectral_filtered_*` targets are accepted as compatibility aliases. Prefer `graph_filtered_*` in new docs and commands.

## Minimum Verification

새 graph algorithm이 framework에 들어오려면 최소한 다음을 확인한다.

- 같은 seed에서 adjacency가 재현된다.
- adjacency shape가 `(num_clients, num_clients)`다.
- negative, NaN, Inf edge weight가 없다.
- `graph_stats.csv`에 density, entropy, degree가 기록된다.
- real graph와 matched control graph를 비교할 수 있다.
- preflight가 해당 suite command를 만들 수 있다.
