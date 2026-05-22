# Composable Graph Algorithm Interfaces

## Interface Map

Use these interfaces instead of adding method-specific branches to the strategy runtime.

| Layer | Question | Interface | Location |
|---|---|---|---|
| Method design | component 조합 | `GraphFLDesign`, `ComponentSpec` | `graphfl_lab/designs/` |
| Client state | client representation | `register_graph_source`, `GraphSourceContext`, `GraphSourceResult` | `graphfl_lab/graph/sources/`, `graphfl_lab/graph/signals/` |
| Relation estimator | relation score | `register_graph_builder`, `GraphBuildContext`, `GraphBuildResult` | `graphfl_lab/graph/registry.py` |
| Topology operator | dense/sparse/control graph | graph builder, sparsification helpers | `graphfl_lab/graph/sparsification.py`, `graphfl_lab/graph/controls.py` |
| Aggregation operator | graph application target | `aggregation_target` | `graphfl_lab/strategies/graphfl/targets.py` |
| Runtime strategy | round execution order | `GraphFLDiagnosticStrategy` | `graphfl_lab/strategies/graphfl/strategy.py` |
| Diagnostics | correction effects | metric/artifact writers | `graphfl_lab/diagnostics/`, `graphfl_lab/strategies/graphfl/diagnostics.py` |
| Suite grammar | repeatable run token | variant parser | `graphfl_lab/experiments/suites/vision/variants.py` |

## GraphFLDesign

`GraphFLDesign` records which lifecycle slots a method occupies.

Required slots:

```text
client_state
relation
topology
aggregation
```

Optional slots:

```text
delivery
local_objective
state_store
diagnostics
```

Support levels:

```text
core-supported
proxy-supported
interface-target
out-of-scope
```

## Graph Source Interface

Add a source only when the client representation is new.

```python
from graphfl_lab.graph import GraphSourceResult, register_graph_source


@register_graph_source("my_client_state")
def build_my_client_state(context):
    vectors = [my_vectorize(update) for update in context.local_updates]
    return GraphSourceResult(
        vectors=vectors,
        source_used="my_client_state",
        metadata={"state_kind": "my_client_state"},
    )
```

Contract:

```text
preserve client order
fixed vector length
finite values only
method-specific metadata
no duplicate name for existing semantics
```

## Graph Builder Interface

Add a builder only when relation or topology construction is new.

```python
import numpy as np

from graphfl_lab.graph import GraphBuildContext, register_graph_builder, require_graph_context


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
    return adj, {"base_graph_kind": "my_relation_graph"}
```

Adjacency contract:

```text
shape = (num_clients, num_clients)
finite
non-negative
symmetric
zero diagonal
```

Directed or signed graph requires topology contract extension.

## Aggregation Target

| Target | Meaning |
|---|---|
| `update` | FedAvg-style local update aggregation |
| `graph_filtered_update` | graph low-pass on current round update matrix |
| `graph_filtered_ema_update` | graph low-pass on client update EMA |
| `weight` | local model weight aggregation |
| `graph_filtered_weight` | graph low-pass on local model weights |

Compatibility (read-only input aliases; new outputs use `graph_filtered_*`):

```text
spectral_filtered_* -> graph_filtered_*   # via canonical_aggregation_target() and config_io
```

New target update points:

```text
targets.py
CLI choices
config schema
suite variants
diagnostics
```

## Suite Variant Interface

Token contract:

```text
source/mode/target inferable from name
control counterpart available
deterministic result path
parser tests under tests/experiments/vision/
examples in extension-guide.md
```

## Diagnostics Contract

Minimum questions:

```text
graph density and degree distribution
real vs random/shuffled/identity
pre/post dominance, alignment, LOO
graph low-pass effect size
relation vs cluster vs graph-free dominance control
```

## Implementation Checklist

1. Write method profile or preset metadata.
2. Add source only if needed.
3. Add builder only if needed.
4. Guard invalid combinations with `require_graph_context`.
5. Add or update `GraphFLDesign`.
6. Add suite token after source/mode/target path exists.
7. Test shape, determinism, metadata, diagnostics, control comparability.
8. Run `python -m unittest discover -s tests`.
9. Run `python scripts/checks/diagnostic_suite_preflight.py`.
