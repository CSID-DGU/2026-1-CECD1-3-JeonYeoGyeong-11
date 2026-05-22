# GraphFL Lab Cleanup Status

This file is the source of truth for the staged cleanup/rename execution state.
Git is the source of truth for code state; this file records intent,
checkpoint status, and the next safe action. If Git state and this file
disagree, rerun the relevant gate check and update this file from the result.

## Current Status

| Field | Value |
|---|---|
| current_gate | Gate 5d-prep stabilized; moving to local smoke/docs readiness |
| status | Gate 4c remote green remains pending; local commit-only Gate 5a-prep through Gate 5d-prep are committed; representative local smoke plus tiny vision and Cora experiment paths passed; optional deeper modularization is deferred |
| owner | codex |
| started_at | 2026-05-21 |
| last_verified | see `docs/maintenance/last_gate_check.json` |
| next_step | continue current-project docs/readiness work or broader suite preparation only when needed; do not claim Gate 4c/5/6 completion before golden/nightly evidence |

Only one Gate branch should be active at a time. In short: use a single Gate branch.
If parallel work is
unavoidable, update this file only immediately before or immediately after the
merge that changes the main branch state.

When an unexpected situation appears, record one line under "Findings And
Decisions" before changing code.

## Existing Unstaged Docs

These files were already modified before Gate 0 to absorb the
semantic/smoothing vocabulary into the current diagnostic-framework wording.
Do not overwrite them during cleanup without first reviewing the diff.

| Path | Handling |
|---|---|
| `docs/framework/claim.md` | preserved in commit `39db595` |
| `docs/framework/graph_fl_experimental_design.md` | preserved in commit `39db595` |
| `docs/framework/graph_fl_experimental_design_appendix.md` | preserved in commit `39db595` |
| `docs/research/prior-work-review.md` | preserved in commit `39db595` |

## Source Of Truth

| Artifact | Owns |
|---|---|
| `docs/maintenance/cleanup-status.md` | current gate, owner, next step, last verified state |
| `docs/maintenance/rename-inventory.md` | legacy name inventory and classification |
| `MIGRATION.md` | user and maintainer migration guidance |
| `docs/removed-materials.md` | tombstones, archived material, and release/tag references |
| `docs/maintenance/line-budget-allowlist.txt` | line-budget exceptions and expiry dates |
| `docs/maintenance/last_gate_check.json` | last machine-readable gate-check result |
| `tests/golden/README.md` | golden baseline policy and volatile field list |

## Gate Check Contract

Gate checks run through one cross-platform entrypoint:

```text
python scripts/dev/run.py gate-check <gate>
```

Exit code contract:

```text
0      pass
non-0  fail
```

Every run writes `docs/maintenance/last_gate_check.json` with these fields:

```text
gate
pass
failed_checks
verified_at
commit_sha
```

## Existing Plan Mapping

| Existing document item | Final Plan Gate | Note |
|---|---:|---|
| `spectral_fl` -> `graphfl_lab` package rename | Gate 3 | alias bridge first, real move later in same Gate as separate commits |
| remove `spectral_filter_strength` | Gate 6 | Gate 2 freezes reader/schema policy first |
| remove `ours_spectral_filtered_*` | Gate 6 | old tokens remain through deprecation |
| remove `_spectral_only` / `_speconly` | Gate 6 | old suffixes remain through deprecation |
| remove `spectral_filtered_*` | Gate 3 / Gate 6 | canonical internals migrate in Gate 3, compatibility removal in Gate 6 |
| naming order item 3: lower-level `spectral_filtered_*` outputs | Gate 3 | internal canonicalization before hard cleanup |
| naming order item 4: `graphfl_lab` package alias | Gate 3 | first package-rename commit |
| naming order item 5: real package move | Gate 3 | later commit after alias/import tests |

Gate 3 combines the previous alias-package stage and real package move stage
into one gated PR, but not one patch. It must proceed as separate sequential
commits: alias bridge, import batches, pyproject/Flower entrypoint, then real
move verification. The old `spectral_fl` shim remains until Gate 6.

## Gate Policies

### Gate 3 Batch Policy

Default import migration order is leaf-to-root:

```text
data/models/diagnostics/corrections
graph/designs/lifecycle
strategies
experiments
cli/scripts/tests
pyproject/Flower
```

If actual import dependencies require a different order, update
`rename-inventory.md` with the finding and decision before changing code.

### Real Move Verification

real move verification means rechecking, after alias bridge and import batches:

```text
canonical import
legacy import through shim
DeprecationWarning behavior
GRAPHFL_LAB_SILENCE_DEPRECATION behavior
sys.modules alias behavior
pickle round-trip compatibility
Flower bootstrap for canonical and legacy app paths
```

### Nightly Policy

There is no scheduled nightly workflow yet. Gate 4c introduces a workflow with
both `schedule` and `workflow_dispatch`.

Gate 4c completion requires one scheduled or manual nightly green run. Gate 6
entry requires seven consecutive nightly green runs after Gate 4c.

### Gate 5 Sub-Gate Policy

Gate 5 is behavior-preserving engineering cleanup, split into separate
sub-gates:

```text
Gate 5a reporting/result writer split
Gate 5b vision suite/runner orchestration split
Gate 5c suite variants parser/token/family/path helper split
Gate 5d graphfl strategy split
```

Gate 5b starts with monolithic `variants.py` unchanged. If variants dependency
blocks Gate 5b, record the finding here and in `rename-inventory.md`, then swap
Gate 5b and Gate 5c.

Golden baseline updates are allowed only in a separate PR with explicit reason
and impact scope.

## Deferred Internal Debt

The items below are intentionally deferred because the current local cleanup
already supports experiment-readiness work. Resume them only after smoke runs or
golden comparisons show that the boundary is actually painful.

| Area | Current state | Deferred work | Resume trigger |
|---|---|---|---|
| `graphfl_lab/strategies/graphfl/strategy.py` | about 600 lines; calculation, artifact writing, round outputs, projection, and graph construction helpers are extracted | split remaining `aggregate_fit` orchestration only if it becomes hard to test or change | repeated edits to the same round-flow block or a failed smoke that requires deeper isolation |
| `graphfl_lab/experiments/suites/vision/reporting.py` | about 619 lines; currently the largest remaining vision-suite file | split normalization, result table building, and golden/report comparison helpers | Gate 5 golden comparison work or report/schema changes touch this file again |
| `graphfl_lab/strategies/graphfl/diagnostics.py` | about 426 lines; mostly metric and spectral diagnostic math | split metric families only if tests need narrower fixtures | new diagnostic metric families or repeated changes to `build_round_log` / `build_fit_metrics` |
| `graphfl_lab/strategies/graphfl/aggregation.py` | about 311 lines; shared aggregation/math helper surface | defer further split to avoid moving stable math during smoke validation | new aggregation target families or conflicts between graph-free and graph-aware paths |

Deferred means "recorded, acceptable, and not blocking current smoke/docs work."
It does not mean "forgotten." Any future refactor of these areas remains
behavior-preserving unless a new Gate explicitly changes public CLI, config, or
result-schema contracts.

## Findings And Decisions

| Date | Finding | Decision |
|---|---|---|
| 2026-05-21 | Current repository has `.github/workflows/ci.yml` only; no scheduled nightly workflow. | Add nightly workflow in Gate 4c with `schedule` and `workflow_dispatch`. |
| 2026-05-21 | `tests/structure/test_boundaries.py` keeps top-level runners as thin facades. | Put unified runner dispatch logic inside package code; keep `run_experiment.py` thin. |
| 2026-05-21 | Current working tree has four unstaged docs changes. | Preserve and record them before Gate 1. |
| 2026-05-21 | Recursive filesystem scan can hit ignored/cache directories such as `.pytest_cache` with access errors. | Gate 1 inventory uses `git ls-files` for tracked sources plus explicit serialized asset globs that exclude virtualenv/cache directories. |
| 2026-05-21 | Creating branch `codex/graphfl-cleanup-gate-0` failed because the local Git refs cannot create the `codex/` directory. | Use `codex-graphfl-cleanup-gate-0` for this local branch while keeping the same Gate 0 scope. |
| 2026-05-21 | Gate 2 added additive-only result schema fields and config alias tracing. | New result/suite payloads get `result_schema_version`, `config_aliases_used`, and `unsupported_components`; missing version is read as `v0`. |
| 2026-05-21 | Gate 3 alias bridge can be verified before full import-batch migration, but calling it full Gate 3 would overstate completion. | Split local checks as `3a` for alias bridge; keep `3` fail-closed until full package migration and real move verification are done. |
| 2026-05-21 | Gate 3 import identity changes touch a broad set of source, script, and test files. | Treat this as `3b` canonical import batch; forbid new `spectral_fl` import tokens outside the shim, alias test, and archived scripts before attempting the real package move. |
| 2026-05-21 | Gate 3 real move leaves only `spectral_fl/__init__.py` as the legacy package shim. | Update line-budget paths and gate checks to `graphfl_lab/*`; verify legacy submodule imports through shim `__path__` for old pickle paths. |
| 2026-05-21 | `run_experiment.py` was still a Cora-only wrapper after Gate 3. | Gate 4a adds `graphfl_lab.cli.experiment_dispatcher`; missing `--track` keeps Cora behavior with `DeprecationWarning`, while `--track vision|cora` selects the canonical runner. |
| 2026-05-21 | Only single-run facades have a unified dispatcher equivalent in Gate 4b. | Rewire `run_vision_experiment.py` and `run_general_experiment.py` through dispatcher track helpers; keep suite/stress/count wrappers on package CLI modules until their unified equivalents exist. |
| 2026-05-21 | Existing CI compile step still referenced the pre-move package layout. | Update CI compile paths to `graphfl_lab` plus the `spectral_fl` shim and add a separate scheduled/manual nightly workflow. |
| 2026-05-21 | Pushing the Gate branch is required for GitHub nightly/manual-nightly verification, but external publication needs explicit user approval. | Do not push or trigger remote workflows yet; continue only local Gate 4c prep such as golden comparison tooling and keep Gate 4c fail-closed. |
| 2026-05-21 | User prefers continuing with local commits and no immediate push. | Leave Gate 4c completion pending; allow local Gate 5a prep commits that do not require remote workflow state and do not claim full Gate 5a completion. |
| 2026-05-21 | Vision suite orchestration contains reusable run-or-reuse subprocess logic. | Extract only the execution helper as Gate 5b-prep; keep result aggregation and public runner behavior unchanged. |
| 2026-05-21 | Gate 5b-prep is not a complete Gate 5b modularization. | Keep the golden/nightly prerequisite visible and do not claim full Gate 5b completion. |
| 2026-05-21 | Client-count and stress-grid wrappers still invoke subprocesses directly with the project root. | Reuse the new execution helper for those wrapper calls only; keep wrapper arguments, outputs, and skip/reuse behavior unchanged. |
| 2026-05-21 | `vision/suite.py` still mixes runner orchestration with per-run feature extraction, timing normalization, and ranking helpers. | Move those pure helpers to `experiments/suites/vision/features.py`; keep `suite.py` as the compatibility import surface during Gate 5b-prep. |
| 2026-05-21 | `vision/suite.py` still builds summary rows inline after collecting raw run rows. | Move summary-row aggregation to `experiments/suites/vision/summary.py` so runner orchestration and metric aggregation stay separately testable. |
| 2026-05-21 | Feature extraction already collects DI, N_eff, alignment, and LOO pre/post diagnostics, but suite summary aggregation did not surface them. | Add the missing summary aggregates as a corrective Gate 5b-prep contract so one suite result can compare the full diagnostic set. |
| 2026-05-21 | Suite metadata still lives inside the runner and does not describe the newly surfaced DI/N_eff/alignment/LOO summary fields. | Move metadata builders to `experiments/suites/vision/metadata.py` and update trace semantics text so result JSON documents the full diagnostic set. |
| 2026-05-21 | Gate 5b-prep reduced `vision/suite.py` to runner orchestration, while `variants.py` remains the largest Gate 5c target. | Start Gate 5c-prep with token/control/path helpers only; leave `parse_variant` branch order unchanged. |
| 2026-05-21 | Gate-check self-tests still verify earlier prep gates after moving `current_gate` forward. | Keep completed prep gate names visible in status text so older checks remain reproducible while Gate 5c-prep proceeds. |
| 2026-05-21 | `variants.py` still mixes token parsing with base command assembly. | Move only `build_base_cmd` to `experiments/suites/vision/variant_commands.py`; keep `variant_cmd` and `parse_variant` public imports stable. |
| 2026-05-21 | `parse_variant` still contains baseline/Fed* parsing before the Ours graph families. | Move only baseline/Fed* family parsing to `experiments/suites/vision/variant_families.py`; leave recursive Ours suffix handling and graph families in place. |
| 2026-05-21 | Current diagnostic protocol tokens (`real_graph`, matched controls, cluster-only, graph-free controls) are mixed into the general Ours parser. | Move only those diagnostic protocol tokens to `experiments/suites/vision/variant_diagnostics.py`; keep all older Ours graph families in `parse_variant`. |
| 2026-05-21 | Legacy residual reweight tokens are compatibility surface and still occupy a separate block inside `parse_variant`. | Move only those legacy residual tokens to `experiments/suites/vision/variant_legacy.py`; keep their exact CLI args unchanged until Gate 6. |
| 2026-05-21 | Basic Ours graph-mode tokens are stable and independent of source/aggregation-target families. | Move those basic graph-mode tokens to `experiments/suites/vision/variant_core.py`; keep source-specific graph families in `parse_variant`. |
| 2026-05-21 | After extracting core graph variants, `ours_default_graph` remained duplicated in `variants.py`. | Remove the duplicate local branch and route it through `parse_core_graph_variant` like the rest of the core graph family. |
| 2026-05-21 | Source-specific Ours variants (`weight`, `layerwise`, classifier-head, EMA, tail slices) are still mixed into the main parser. | Move those source-family tokens to `experiments/suites/vision/variant_sources.py`; keep target-only filtered families separate for the next step. |
| 2026-05-21 | Target-only filtered variants (`graph_filtered_*` and legacy `spectral_filtered_*`) remain as the last large family inside `parse_variant`. | Move those target-family tokens to `experiments/suites/vision/variant_targets.py`; preserve legacy spectral tokens until Gate 6. |
| 2026-05-21 | Recursive suffix handling (`fixed_tau`, `graph_filter_only`, `lp`, `serverm`) is now the last non-routing logic inside `variants.py`. | Move suffix handling to `experiments/suites/vision/variant_suffixes.py` with the parser callback injected to preserve recursive behavior. |
| 2026-05-21 | Gate 5d starts with `strategy.py` still owning client-update EMA copy/update logic. | Extract only the EMA calculation/copy helper to `strategies/graphfl/ema.py`; keep state assignment in `GraphFLDiagnosticStrategy`. |
| 2026-05-21 | `strategy.py` still owns the cached projection branch used before graph construction. | Extract projection calculation to `strategies/graphfl/projection.py`; keep projection matrix storage on `GraphFLDiagnosticStrategy`. |
| 2026-05-21 | `strategy.py` still owns client metric alias extraction and optional weighted means. | Extract metric readers to `strategies/graphfl/client_metrics.py`; keep aggregate_fit call sites unchanged apart from function names. |
| 2026-05-21 | `strategy.py` still owns diagnostic post-target flattening for update, EMA update, and weight targets. | Extract target flattening to `strategies/graphfl/diagnostic_targets.py`; keep `_diagnostic_post_flat_updates` as a wrapper around current strategy state. |
| 2026-05-21 | `aggregate_fit` still owns Flower fit-result ordering and conversion to client arrays. | Extract result collection to `strategies/graphfl/fit_results.py`; preserve stable CID ordering from baselines ordering. |
| 2026-05-21 | `strategy.py` still owns current-vs-EMA graph selection and source labeling. | Extract graph EMA selection to `strategies/graphfl/graph_state.py`; preserve existing `ema_graph` label when EMA is enabled outside warmup. |
| 2026-05-21 | `strategy.py` still owns graph-meta cluster-id normalization for client diagnostics. | Extract cluster-id normalization to `strategies/graphfl/graph_metadata.py`; preserve `-1` fallback for missing or mismatched cluster lists. |
| 2026-05-21 | `aggregate_fit` still owns inline module-trace run-context enrichment. | Extract trace enrichment to `strategies/graphfl/trace_context.py`; preserve default round/run_id/variant/seed insertion. |
| 2026-05-21 | `aggregate_fit` still owns diagnostic CSV row construction for round, graph, and client artifacts. | Extract row builders to `strategies/graphfl/artifact_rows.py`; keep CSV append calls in `GraphFLDiagnosticStrategy`. |
| 2026-05-21 | `aggregate_fit` still owns default counterfactual spec retargeting, runner setup, and artifact row construction. | Extract counterfactual artifact orchestration to `strategies/graphfl/counterfactual_artifacts.py`; keep CSV/JSONL append calls in `GraphFLDiagnosticStrategy`. |
| 2026-05-21 | `aggregate_fit` still owns config context field selection for logs and metrics. | Extract config context projection to `strategies/graphfl/config_context.py`; keep the explicit field list tested. |
| 2026-05-21 | `aggregate_fit` still owns round Laplacian/H_spec/spectral energy metric calculation. | Extract spectral metric bundle to `strategies/graphfl/spectral_metrics.py`; keep state assignment for H_spec EMA in `GraphFLDiagnosticStrategy`. |
| 2026-05-21 | `aggregate_fit` still owns graph-filtered conflict, tau resolution, and conflict-weight calculation. | Extract conflict metric bundle to `strategies/graphfl/conflict_metrics.py`; keep tau-signal EMA state assignment in `GraphFLDiagnosticStrategy`. |
| 2026-05-21 | `aggregate_fit` still owns the sequence that selects aggregation weights and then applies correction-family post-processing. | Extract round weight orchestration to `strategies/graphfl/round_weights.py`; preserve existing alpha_mode composition. |
| 2026-05-21 | `aggregate_fit` still owns local update construction and repeated flat/norm update-space arrays. | Extract update-space arrays to `strategies/graphfl/update_space.py`; preserve flattened delta matrix used by diagnostics and counterfactuals. |
| 2026-05-21 | `aggregate_fit` still owns seeded relation graph construction, pre-weight calculation, graph diagnostics, and EMA graph selection. | Extract round graph state to `strategies/graphfl/round_graph.py`; preserve graph seed formula and sample-weight normalization. |
| 2026-05-21 | `aggregate_fit` still owns round log/metric context dictionary assembly. | Extract context builders to `strategies/graphfl/round_context.py`; keep `build_round_log` and `build_fit_metrics` as the output contract. |
| 2026-05-22 | `aggregate_fit` still owned diagnostic artifact file writing after row/counterfactual helpers were extracted. | Move artifact append orchestration to `strategies/graphfl/diagnostic_artifacts.py`; keep CSV/JSONL filenames and row contracts unchanged. |
| 2026-05-22 | `aggregate_fit` still assembled round log contexts and scalar fit metrics inline after context helpers existed. | Move round output composition to `strategies/graphfl/round_outputs.py`; preserve `round_logs` and Flower metrics payloads. |
| 2026-05-22 | `aggregate_fit` still mixed graph-source vector selection with projection-space construction. | Move projected graph-space preparation to `strategies/graphfl/round_projection.py`; keep projection matrix state on the strategy. |
| 2026-05-22 | Existing graph-source tests exercise the private `_graph_vectors` wrapper. | Keep `_graph_vectors` as a compatibility wrapper around the extracted graph-source helper during Gate 5d-prep. |
| 2026-05-22 | Further line shaving would mostly target orchestration readability rather than immediate experiment readiness. | Defer optional deeper splits of `strategy.py`, `vision/reporting.py`, `diagnostics.py`, and `aggregation.py`; move next to local smoke checks and docs tightening. |
| 2026-05-22 | Representative local smoke checks passed: `gate-check 5d-prep`, focused graph/design/schema/golden/runner tests, and runner help checks for `run_experiment.py`, `run_vision_experiment.py`, and `run_graph_ablation.py`. | Keep Gate 4c remote/nightly completion pending; use these local checks as readiness evidence only, not full Gate completion. |
| 2026-05-22 | Diagnostic suite preflight initially reported `NEEDS_REVIEW` because its expected args still used legacy `spectral_filtered_update` while diagnostic variants now emit canonical `graph_filtered_update`. | Update preflight expectations to canonical `graph_filtered_update`; rerun preflight and related variant tests until green. |
| 2026-05-22 | `print-flwr-run` generated a valid Flower command for `default_similarity_knn`, but displayed unresolved parser defaults such as `aggregation-target="update"` before the app-side graph-method preset was applied. | Resolve graph presets during run-config generation while preserving explicit user overrides, so printed and submitted Flower configs match the intended graph design. |
| 2026-05-22 | Tiny representative vision run `configs/vision/smoke/default_similarity_knn.json` completed with `graph_method=default_similarity_knn`, `graph_mode=rbf_knn`, `aggregation_target=graph_filtered_update`, and schema fields `result_schema_version`, `config_aliases_used`, `unsupported_components`. | Treat this as local execution readiness evidence for the default graph pipeline; keep remote/nightly and broader suite evidence pending. |
| 2026-05-22 | `configs/cora/ablations/graph/graph_ablation_smoke.json` still had mojibake in its description. | Replace only the description text before using it as graph-dataset smoke evidence. |
| 2026-05-22 | Cora graph ablation smoke `configs/cora/ablations/graph/graph_ablation_smoke.json` completed for `fedavg`, `ours_knn`, and `ours_random`, producing per-run results plus `suite_cora_graph_ablation_smoke_summary.*`. | Treat this as graph-dataset local execution evidence; avoid interpreting the 1-round diagnostic-only numbers as research results. |
| 2026-05-22 | Cora graph ablation smoke completed, but its subprocess commands still invoked `run_experiment.py` without `--track cora` and triggered the Gate 4a deprecation warning. | Add `--track cora` to graph-ablation subprocess commands and cover the command contract with a focused test. |
| 2026-05-22 | Current docs such as `docs/structure.md` still presented `spectral_fl/...` as the active edit path after the package move. | Update current routing docs to `graphfl_lab/...`; keep `spectral_fl` references only where they describe deprecation compatibility or cleanup inventory. |
| 2026-05-22 | Gate-check self-test found `line-budget-allowlist.txt` still referenced non-existent `graphfl_lab/experiments/suites/vision/suite.py` instead of the actual orchestration file. | Align the protected path with `scripts/dev/run.py`: `graphfl_lab/experiments/vision/suite.py`. |
| 2026-05-22 | Result schema checks guarantee top-level contract fields, but they do not independently verify that performance, mechanism, and secondary diagnostic metrics are present together. | Add an evidence-bundle validator for single-run results and suite summaries, so missing DI/N_eff/alignment/LOO/spectral fields fail as a checkable contract. |
| 2026-05-22 | `docs/framework/experimental-design.md` is now only a bridge to the canonical `graph_fl_experimental_design.md`. | Remove the duplicate bridge from current framework docs and record the replacement in `docs/removed-materials.md`. |
| 2026-05-22 | Current docs `configs/README.md` and `docs/research/framework-design-notes.md` still described `spectral_fl` package paths as active implementation paths. | Update those current docs to `graphfl_lab`; keep `spectral_fl` mentions only for compatibility debt and archived/history records. |
| 2026-05-22 | The graph-source vector helper still exposed the old `graph_vectors_for_spectral` name as the active helper in strategy and baseline code. | Add canonical `graph_vectors_for_graphfl`, migrate active imports, and retain `graph_vectors_for_spectral` as a Gate 6 compatibility alias. |

## Closure Policy

After Gate 6 completes, mark this file `closed`. Keep only a tombstone link from
`docs/removed-materials.md`.
