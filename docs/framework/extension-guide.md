# Graph Algorithm Extension Guide

## Method Profile

Start with the method profile; code changes follow from the occupied slots.

| Slot | Question |
|---|---|
| `client_state` | server가 client를 어떤 state/vector로 표현하는가? |
| `relation_estimator` | state에서 edge/relation을 어떻게 추정하는가? |
| `topology_operator` | relation을 어떤 topology로 만드는가? |
| `aggregation_operator` | relation을 update, weight, personalized model generation 중 어디에 적용하는가? |
| `personalization_site` | personalization 위치 |
| `local_objective_hook` | local loss hook |
| `diagnostic_projection` | exact, proxy, future interface 중 무엇인가? |

CLI split:

```text
graph_method      = runnable method/profile
graph_source      = client representation
graph_mode        = relation/topology builder
aggregation_target = graph application target
```

## Built-In Presets

Presets are runnable profiles or proxies, not paper-exact claims.

| Preset | Method profile | Runnable knobs | Support |
|---|---|---|---|
| `fedamp_proxy` | FedAMP attentive message passing | `weight + rbf + graph_filtered_weight` | proxy-supported |
| `sfl_proxy` | SFL graph-structured aggregation | `weight + learned_smooth + graph_filtered_weight` | proxy-supported |
| `pfedgraph_proxy` | pFedGraph inferred collaboration graph | `update + pfedgraph_qp + graph_filtered_update` | proxy-supported |
| `ema_magnitude_knn_filtered` | FedAGA accumulated-gradient graph | `ema_update + magnitude_knn + graph_filtered_ema_update` | proxy-supported |
| `fedpub_like` | functional embedding + personalized aggregation | not runnable | interface-target |

Compatibility aliases:

```text
fedamp_like -> fedamp_proxy
sfl_like -> sfl_proxy
pfedgraph_like -> pfedgraph_proxy
```

## Graph Source Plugin

```python
from spectral_fl.graph import GraphSourceResult, register_graph_source
from spectral_fl.projection import flatten_weights


@register_graph_source("my_functional_embedding")
def build_my_functional_embedding(context):
    vectors = [flatten_weights(local_weights) for local_weights in context.local_weights]
    return GraphSourceResult(
        vectors=vectors,
        source_used="my_functional_embedding",
        metadata={"state": "functional_embedding"},
    )
```

Attach here:

```text
functional embedding
dataset-stat embedding
gradient-history embedding
custom client state
```

## Graph Builder Plugin

```python
import numpy as np

from spectral_fl.graph import GraphBuildContext, register_graph_builder, require_graph_context


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

Run:

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

Use context fields rather than global state so builders remain reproducible.

| Field | Meaning |
|---|---|
| `z_mat` | client representation matrix |
| `mode` | current `--graph-mode` |
| `graph_source` | source name |
| `aggregation_target` | target name |
| `correction_family` | real/control/clustering/graph-free family |
| `knn_k` | kNN knob |
| `edge_threshold` | threshold knob |
| `rng` | round deterministic RNG |
| `graph_scale_sigma` | RBF/magnitude/custom scale |
| `learned_graph_lambda` | smooth/QP/custom regularization |
| `extras["client_sample_weights"]` | sample-size prior |

Use `require_graph_context(...)` for source/target-specific algorithms.

## Source/Mode/Target Interpretation

| Combination | Interpretation |
|---|---|
| `update + graph_filtered_update` | full update relation, update smoothing |
| `classifier_head_update + graph_filtered_update` | head signal relation, full update correction |
| `ema_update + graph_filtered_ema_update` | temporal relation, EMA aggregation |
| `weight + graph_filtered_weight` | model state similarity, weight smoothing |
| `update + pfedgraph_qp + graph_filtered_update` | pFedGraph-style prior/QP as diagnostic graph |
| `graph_free correction + update` | dominance/magnitude control without graph relation |

## Minimum Verification

```text
same-seed adjacency reproducible
adjacency shape = (num_clients, num_clients)
no negative/NaN/Inf edge weight
graph_stats.csv includes density, entropy, degree
matched control graph available
preflight can produce suite command
```
