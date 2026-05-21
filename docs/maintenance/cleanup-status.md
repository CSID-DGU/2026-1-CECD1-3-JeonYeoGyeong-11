# GraphFL Lab Cleanup Status

This file is the source of truth for the staged cleanup/rename execution state.
Git is the source of truth for code state; this file records intent,
checkpoint status, and the next safe action. If Git state and this file
disagree, rerun the relevant gate check and update this file from the result.

## Current Status

| Field | Value |
|---|---|
| current_gate | Gate 4a - unified runner dispatcher |
| status | Gate 3 real package move complete; Gate 4a dispatcher in progress |
| owner | codex |
| started_at | 2026-05-21 |
| last_verified | see `docs/maintenance/last_gate_check.json` |
| next_step | verify `run_experiment.py --track vision|cora`, missing-track deprecation, and thin wrapper boundary |

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

## Closure Policy

After Gate 6 completes, mark this file `closed`. Keep only a tombstone link from
`docs/removed-materials.md`.
