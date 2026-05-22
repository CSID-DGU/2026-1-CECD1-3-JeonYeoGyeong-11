# Project Working Prompt

Use for new implementation or cleanup sessions.

```text
You are working in the Graph-FL Design Lab repository.

Goal:
- Test whether observed Graph-FL gains come from graph structure or simpler confounders:
  dominance, norm, smoothing, optimizer effects.
- Represent graph-FL methods as lifecycle components:
  client_state, relation, topology, aggregation, delivery, local_objective,
  state_store, diagnostics.
- Prioritize composability, diagnostics, controls, and claim boundaries over leaderboard accuracy.

Naming:
- Prefer canonical vision paths for new vision FL work.
- Keep general paths only as compatibility wrappers or artifact/config aliases.
- Treat spectral as an operator/backend word, not project identity.
- Prefer graph_filtered_*, graph_filter_strength, ours_graph_filtered_*, _graph_filter_only.

Interfaces:
- Method metadata: graphfl_lab/designs/
- Client-state extraction: graphfl_lab/graph/sources/, graphfl_lab/graph/signals/
- Graph construction: graphfl_lab/graph/registry.py, builders.py, similarity/, sparsification.py
- Control graphs and clustering: graphfl_lab/graph/controls.py, graphfl_lab/graph/clustering.py
- Aggregation targets: graphfl_lab/strategies/graphfl/targets.py
- Runtime: graphfl_lab/strategies/graphfl/strategy.py
- Diagnostics: graphfl_lab/diagnostics/, graphfl_lab/strategies/graphfl/diagnostics.py
- Suite/reporting: graphfl_lab/experiments/suites/vision/
- Vision orchestration: graphfl_lab/experiments/vision/
- CLI parser modules: graphfl_lab/cli/

New graph algorithm workflow:
1. Read README.md, docs/framework/graph_fl_experimental_design.md, docs/structure.md,
   docs/framework/interfaces.md, docs/framework/extension-guide.md.
2. Write method profile:
   client_state, relation estimator, topology, aggregation target,
   personalization site, exact/proxy/interface/out-of-scope status.
3. Reuse graph_source, graph_mode, aggregation_target, correction family,
   cluster controls, graph-free controls when possible.
4. Add graph source only for genuinely new representation.
5. Add graph builder only for genuinely new relation/topology.
6. Register builders with register_graph_builder(...).
7. Guard invalid combinations with require_graph_context(...).
8. Add/update GraphFLDesign preset and --graph-method profile.
9. Add suite tokens only after source/mode/target path exists.
10. Add diagnostics and tests:
    deterministic adjacency, valid shape, finite non-negative weights,
    metadata, graph stats, round/client diagnostics, matched control path.

Verification:
- python -m unittest discover -s tests
- python scripts/checks/diagnostic_suite_preflight.py
- python scripts/smoke/prior_work_proxy.py
- python scripts/checks/prior_work_proxy_parity.py --summary <summary.json>

Output:
- State changes, compatibility debt, and checks passed.
- Do not claim exact paper reproduction unless state, relation, topology,
  aggregation, delivery, local objective, and personalization all match.
```

## Read First

Read these before changing interfaces, experiment design, or public run paths.

```text
README.md
docs/framework/graph_fl_experimental_design.md
docs/framework/claim.md
docs/structure.md
docs/framework/interfaces.md
docs/framework/naming-and-compatibility.md
```

Prior-work proxy:

```text
docs/framework/prior-work-mapping.md
```
