# Project Working Prompt

Use this prompt when starting a new implementation or cleanup session in this
repository. It captures the current project goal, interface boundaries, and
verification expectations.

```text
You are working in the Graph-FL Design Lab repository.

Project goal:
- This is not a repo for claiming one new graph algorithm is universally better than FedAvg.
- The goal is to test whether observed graph-FL gains come from graph structure
  or from simpler confounders such as dominance, norm, smoothing, and optimizer effects.
- The implementation vehicle is a composable diagnostic framework for graph-based federated learning.
- A graph-FL method should be represented as replaceable lifecycle components:
  client_state, relation, topology, aggregation, delivery, local_objective,
  state_store, and diagnostics.
- Performance is secondary. Composability, runnable diagnostics, controls,
  and clear claim boundaries are primary.

Repository principles:
- Prefer canonical `vision` paths for new vision FL work.
- Keep `general` paths only as compatibility wrappers or artifact/config-path
  aliases. New vision configs belong under `configs/vision/`.
- Treat `spectral` as an operator/backend word, not as the project identity.
  New strategy runtime work belongs under `spectral_fl/strategies/graphfl/`.
- Prefer public graph spellings in new commands and configs:
  `graph_filtered_*`, `graph_filter_strength`, `ours_graph_filtered_*`,
  and `_graph_filter_only`.
- Use `--graph-method default_similarity_knn` as the representative built-in
  graph-FL path unless a task requires a paper-specific proxy.
- New graph algorithms should enter through interfaces, not by adding large
  branches inside `strategy.py`.
- If a rename touches public CLI names, config paths, import paths, or result
  filenames, treat it as a migration. Keep a wrapper and document the debt.
- If a change is even slightly risky, document it first in
  `docs/framework/cleanup-plan.md` or `docs/framework/naming-and-compatibility.md`.

Primary interfaces:
- Method metadata: `spectral_fl/designs/`
- Client-state extraction: `spectral_fl/graph/sources/` and `spectral_fl/graph/signals/`
- Graph construction: `spectral_fl/graph/registry.py`, `builders.py`,
  `similarity/`, and `sparsification.py`
- Control graphs and clustering: `spectral_fl/graph/controls.py`,
  `spectral_fl/graph/clustering.py`
- Aggregation target logic: `spectral_fl/strategies/graphfl/targets.py`
- Strategy runtime: `spectral_fl/strategies/graphfl/strategy.py`
- Diagnostics: `spectral_fl/diagnostics/` and
  `spectral_fl/strategies/graphfl/diagnostics.py`
- Suite tokens and reporting: `spectral_fl/experiments/suites/vision/`
- Vision run orchestration: `spectral_fl/experiments/vision/`
- Parser-only CLI modules: `spectral_fl/cli/`

Workflow for a new graph algorithm:
1. Read `README.md`, `docs/framework/experimental-design.md`,
   `docs/structure.md`,
   `docs/framework/interfaces.md`, and
   `docs/framework/extension-guide.md`.
2. Write the method profile:
   - What is the client_state?
   - What is the relation estimator?
   - What topology does it produce?
   - What aggregation target does it affect?
   - Is personalization server-side, client-side, local objective, or absent?
   - Is the implementation exact, proxy-supported, interface-target, or out-of-scope?
3. Reuse existing knobs if possible:
   `graph_source`, `graph_mode`, `aggregation_target`, correction family,
   cluster controls, graph-free controls.
4. Add a graph source only when the client representation is genuinely new.
5. Add a graph builder only when relation/topology construction is genuinely new.
6. Register new builders with `register_graph_builder(...)` and constrain
   incompatible combinations with `require_graph_context(...)`.
7. Add or update a `GraphFLDesign` preset and expose runnable profiles through
   `--graph-method`.
8. Add suite variant tokens only after the lower-level source/mode/target path exists.
9. Add diagnostics and tests:
   - deterministic adjacency for the same seed
   - adjacency shape and finite non-negative weights
   - metadata fields for method interpretation
   - graph stats and round/client diagnostics
   - matched control or graph-free comparison path
10. Update docs before claiming completion.

Verification:
- `python -m unittest discover -s tests`
- `python scripts/checks/diagnostic_suite_preflight.py`
- For executable composability smoke:
  `python scripts/smoke/prior_work_proxy.py`
- If using a prior-work proxy smoke summary:
  `python scripts/checks/prior_work_proxy_parity.py --summary experiments_current/prior_work_proxy_smoke/<stamp>/prior_work_proxy_summary.json`

Expected output style:
- State what changed, what remains compatibility debt, and which checks passed.
- Do not claim a prior paper is exactly reproduced unless the implementation
  matches the paper's actual state, relation, topology, aggregation, delivery,
  local objective, and personalization semantics.
```

## How To Use

For a fresh implementation session, read these first:

- `README.md`
- `docs/framework/experimental-design.md`
- `docs/framework/claim.md`
- `docs/structure.md`
- `docs/framework/interfaces.md`
- `docs/framework/naming-and-compatibility.md`

If the task is a prior-work proxy or paper-inspired assembly, also check
`docs/framework/prior-work-mapping.md`.
