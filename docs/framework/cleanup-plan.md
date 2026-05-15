# Repository Cleanup Plan

This document tracks repository cleanup work needed to keep the project usable
as a real Graph-FL Design Lab. The rule is simple: if a rename is safe, do it;
if it touches imports, configs, result files, or public commands, treat it as a
migration with aliases, tests, and a removal gate.

## Cleanup Principles

- New work should use `vision`, `framework`, `research`, `archive`, and
  `graphfl` names where they describe the real responsibility.
- `general` names are compatibility names only.
- `spectral` is not project identity. It may remain only as a compatibility
  spelling or when it specifically describes a graph Laplacian/spectral
  filtering operator.
- New graph algorithms should enter through `GraphFLDesign`, graph source,
  graph builder, aggregation target, lifecycle, and diagnostics interfaces.
- Risky renames must be staged: add canonical name, keep old alias, migrate
  internal references, update docs/configs, verify, then remove old alias later.

## Completed In Current Cleanup

| Item | Status |
|---|---|
| README rewritten around project philosophy, interfaces, run path, and verification | done |
| Active configs moved from `configs/general/` to `configs/vision/` | done |
| `configs/general/...` path alias added in config resolver | done |
| `result_vision_*` and `vision_suite_*` aliases added beside old output names | done |
| Strategy implementation moved to `spectral_fl/strategies/graphfl/` | done |
| `spectral_fl/strategies/spectral/` reduced to compatibility wrappers | done |
| `GraphFLDiagnosticStrategy` added as canonical runtime class | done |
| `SpectralConflictAwareStrategy` retained as compatibility alias | done |
| `graph_filtered_*` aggregation target aliases accepted by CLI/runtime | done |
| `--graph-filter-strength` accepted as CLI spelling | done |
| `graph_filter_strength` written/read as canonical config and result key | done |
| old `spectral_filter_strength` config/result key retained as alias | done |
| `ours_graph_filtered_*` suite tokens added and active configs migrated | done |
| old `ours_spectral_filtered_*` suite tokens retained as aliases | done |
| `graph_filter_only` suffix added for suite tokens | done |
| active configs no longer contain project-level `spectral` naming | done |

## Remaining Hard Renames

These names still need to change eventually. They remain only because changing
them carelessly can break experiments or historical result readers.

| Name | Desired canonical name | Risk |
|---|---|---|
| top-level package `spectral_fl` | `graphfl_lab` | very high: every import, Flower app entrypoint, package discovery, tests |
| compatibility result/meta key `spectral_filter_strength` | remove after reader policy is frozen | medium: historical JSON results, diagnostics, plots, suite summaries |
| compatibility suite tokens `ours_spectral_filtered_*` | remove after old result reuse policy is frozen | medium: historical configs, run tags, result grouping |
| compatibility suffix `_spectral_only` / `_speconly` | remove after old result reuse policy is frozen | low/medium: historical variant tokens |
| compatibility aggregation targets `spectral_filtered_*` | remove after lifecycle/design/result readers use `graph_filtered_*` | high: operator names, tests, historical results |

## Recommended Order

Do not start with the package root rename. It has the widest blast radius.

1. Completed: migrate result/meta key from `spectral_filter_strength` to
   canonical `graph_filter_strength` while keeping the old key as an alias.
2. Completed: add canonical suite tokens `ours_graph_filtered_*` while keeping
   old tokens as aliases.
3. Completed: update active configs and docs to prefer the new key and tokens.
4. Next: migrate remaining design/operator spellings that still expose
   `spectral_filtered_*` where the risk is acceptable.
5. Only after that, migrate the top-level Python package.

This order keeps experiment semantics stable before moving the import surface.

## Migration A: Result And Metadata Key

Goal: make `graph_filter_strength` the canonical result/config/diagnostic key
while retaining `spectral_filter_strength` as a compatibility alias.

Status: canonical key is implemented for CLI/config parsing, run metadata,
diagnostics, suite summaries, and active configs. The old key is still written
as a compatibility field for historical readers.

### A1. Add canonical field

- Add `graph_filter_strength` to run metadata, diagnostics config context, and
  filter diagnostics.
- Keep `spectral_filter_strength` next to it for old readers.
- Internally prefer `graph_filter_strength` for new code paths.

### A2. Reader fallback

- Update plot/report/suite readers to read `graph_filter_strength` first and
  fall back to `spectral_filter_strength`.
- Update any summary CSV/Markdown writers to emit the canonical key.
- Keep old column/key only if historical tooling still reads it.

### A3. CLI/config bridge

- CLI already accepts `--graph-filter-strength`.
- Add config support so `graph_filter_strength`, `graph-filter-strength`, and
  old `spectral_filter_strength` spellings resolve to the same value.
- Prefer `graph_filter_strength` in active configs.

### A4. Tests

- Add unit tests for new-key output and old-key fallback.
- Add config parsing tests for all accepted spellings.
- Run full tests and diagnostic preflight.

### Removal gate

Only remove `spectral_filter_strength` after:

- no active config writes it,
- report readers prefer the new key,
- historical reader behavior is explicitly frozen or archived,
- one full suite output has been generated with `graph_filter_strength`.

## Migration B: Suite Variant Tokens

Goal: make `ours_graph_filtered_*` the canonical suite token family while old
`ours_spectral_filtered_*` tokens remain accepted.

Status: canonical tokens are implemented, active configs use them, reporting
groups both families, and old tokens still resolve to their historical run
labels for result reuse.

### B1. Add parser aliases

- Add `ours_graph_filtered_dense`, `ours_graph_filtered_uniform`,
  `ours_graph_filtered_knn_kN`, `ours_graph_filtered_magnitude...`,
  `ours_graph_filtered_rbf...`, and `ours_graph_filtered_random_matched_kN`.
- Keep the same CLI extras as old tokens, except prefer
  `--aggregation-target graph_filtered_update`.
- Keep old tokens as aliases, not as the recommended spelling.

### B2. Decide run tag behavior

Recommended:

- New token produces new run tag, e.g. `ours_graph_filtered_knn_k2`.
- Old token keeps old run tag, e.g. `ours_spectral_filtered_knn_k2`, so
  `--reuse-existing-results` can still find historical outputs.
- Reporting groups both spellings under one semantic family where comparison
  pairs need to match.

This avoids silently changing where old commands look for results.

### B3. Update configs

- After parser aliases exist, update active `configs/vision/...` files to use
  `ours_graph_filtered_*`.
- Leave archive docs and old result names unchanged.

### B4. Reporting compatibility

- Update pair detectors and summary labels to recognize both old and new token
  families.
- Prefer graph-filtered labels in new Markdown/CSV summaries.

### B5. Tests

- Add parser tests proving old and new tokens produce equivalent flags.
- Add reporting tests proving old/new pair grouping still works.
- Run full tests and diagnostic preflight.

### Removal gate

Do not remove old tokens until:

- active configs no longer use them,
- docs no longer recommend them,
- result readers can load both old and new result names,
- at least one smoke suite has been generated with new tokens.

## Migration C: Top-Level Package Rename

Goal: move project imports from `spectral_fl` to `graphfl_lab`.

This is the highest-risk migration. It should be done only after A and B are
stable.

### C0. Confirm canonical package name

Recommended name: `graphfl_lab`.

Reason:

- short enough for imports,
- matches "Graph-FL Design Lab",
- avoids claiming to be every possible graph-FL library,
- avoids the old spectral-specific identity.

### C1. Alias-first preparation

- Add a new top-level package `graphfl_lab/` that re-exports the current
  `spectral_fl` modules.
- Update `pyproject.toml` package discovery to include both packages.
- Add smoke tests:
  - `import graphfl_lab`
  - `from graphfl_lab.strategies.graphfl import GraphFLDiagnosticStrategy`
  - old `spectral_fl` imports still work.

### C2. Entry point bridge

- Add or update Flower app entrypoints so both work:
  - canonical: `graphfl_lab.flower_app:server_app`
  - compatibility: `spectral_fl.flower_app:server_app`
- Keep script launchers working during the transition.

### C3. Internal import migration

- Migrate internal imports from `spectral_fl.*` to `graphfl_lab.*` in small
  batches by package:
  1. `designs`, `graph`, `diagnostics`, `lifecycle`
  2. `strategies`
  3. `experiments`
  4. `cli`, scripts, tests
- After each batch run targeted tests.
- Avoid mixing import directions in the same module.

### C4. Real package move

When internal imports mostly use `graphfl_lab`, move implementation files from
`spectral_fl/` to `graphfl_lab/`.

Then turn `spectral_fl/` into thin wrappers that import from `graphfl_lab`.

### C5. Docs and command output

- Update README, structure docs, project prompt, and extension examples.
- Update pyproject Flower component defaults to canonical entrypoints.
- Keep a compatibility section listing `spectral_fl` import support.

### C6. Tests

Required checks:

- `python -m unittest discover -s tests`
- `python scripts/checks/diagnostic_suite_preflight.py`
- `python run_vision_experiment.py --help`
- `python run_vision_suite.py --help`
- `python run_general_suite.py --help`
- a tiny `--engine print-flwr-run` smoke with canonical package entrypoints

### Removal gate

Do not remove `spectral_fl` wrappers until:

- no active source imports `spectral_fl` except compatibility wrappers,
- docs use `graphfl_lab`,
- tests cover both canonical and old import paths,
- historical scripts still have an explicit compatibility route.

## Migration D: Graph Source Helper Name

This is smaller than the package root but still should follow it or happen
near it.

| Current | Target |
|---|---|
| `spectral_fl/graph/sources/spectral.py` | `graphfl_lab/graph/sources/graphfl.py` or `graphfl_lab/graph/sources/strategy.py` |
| `graph_vectors_for_spectral` | `graph_vectors_for_graphfl` |

Plan:

1. Add canonical helper and module.
2. Keep old helper as alias.
3. Migrate imports in graph-FL strategy and baselines.
4. Add alias tests.
5. Remove old helper only after package migration is stable.

## Current P0/P1 Backlog

| Priority | Task | Why | Risk |
|---|---|---|---|
| Done | migrate `spectral_filter_strength` metadata to `graph_filter_strength` | public result schema still has old project language | medium |
| Done | add `ours_graph_filtered_*` suite tokens | configs and suite grammar still expose old project language | medium |
| Done | migrate active configs to graph naming | new experiment entry points should not show old project language | low |
| P1 | package alias `graphfl_lab` | top-level package still exposes old identity | high |
| P1 | real package move to `graphfl_lab/` | final cleanup of import identity | very high |
| P1 | graph source helper rename | helper name still says spectral | medium |
| P1 | migrate `spectral_filtered_*` operator internals to graph-canonical outputs | old aggregation target names still appear in lower-level lifecycle/design surfaces | high |
| P1 | decide removal date for `configs/general/...` alias | old config namespace is still supported | medium |
| P1 | decide removal date for `result_general_*` names | old outputs are still generated | medium |
| P2 | generated cache cleanup | source tree scan noise | low |

## Completion Criteria For Any Rename Pass

Each migration pass must satisfy:

1. Canonical name exists and is documented.
2. Old name remains as alias unless the pass explicitly removes it.
3. Active docs and configs prefer the canonical name.
4. Tests cover canonical name and old alias.
5. Full tests pass.
6. Diagnostic preflight passes.
7. Remaining compatibility debt is recorded in
   `docs/framework/naming-and-compatibility.md`.
