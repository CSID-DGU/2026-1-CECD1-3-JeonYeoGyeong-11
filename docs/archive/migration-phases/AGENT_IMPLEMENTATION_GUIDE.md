# Agent Implementation Guide

## Agent Quick Start

When starting or resuming implementation work, do this in order:

1. Read `CURRENT_STATUS.md` first and identify the active phase.
2. Read `README.md` only for global invariants and phase ordering.
3. Read exactly one phase document for the phase you are implementing.
4. Use that phase document's `Agent Task Card`, `Files to create`, `Files allowed to modify`, `Files not allowed to modify`, and `Step-by-step implementation order` as the working contract.
5. Inspect existing code before editing.
6. Implement the next incomplete step only; do not jump ahead to later phases.
7. Run the phase-specific tests listed in the phase document.
8. Run `python -m unittest discover -s tests`.
9. Update `CURRENT_STATUS.md` with completed work, tests, limitations, and the next step.

Reference docs such as `docs/research/design-pattern-survey.md` are normative for lifecycle decomposition, support-level classification, trace vocabulary, and representational coverage. They are not implementation checklists or exact-reproduction backlogs.

## Global Rules

1. Do not change public CLI behavior unless the current phase explicitly asks for it.
2. Do not remove compatibility facades.
3. Every phase must end with:
   - phase-specific tests
   - `python -m unittest discover -s tests`
4. Each lifecycle module must return `output + trace`.
5. The actual training path may update the server model.
6. Counterfactual paths must not update the server model.
7. Unsupported personalized methods must raise explicit errors or expose their `support_level`.
8. Do not silently fallback from unsupported behavior.
9. Do not mix lifecycle responsibilities in one module.
10. Preserve existing behavior unless a phase explicitly defines a migration.
11. When implementing a new graph-FL component, classify it by lifecycle role first: `ClientStateExtractor`, `RelationEstimator`, `TopologyOperator`, `AggregationOperator`, `DeliveryPolicy`, `StateStore`, or `LocalObjectiveHook`.
12. Do not implement a new graph-FL method as a monolithic strategy if it can be expressed by replacing lifecycle components.
13. If a method requires learned server-side graph modules, hypernetworks, generated personalized models, or local GNN architecture changes, mark the unsupported part as `interface-target` or `out-of-scope` instead of silently approximating it.
14. Any prior-work-inspired component must declare whether it is `core-supported`, `proxy-supported`, `interface-target`, or `out-of-scope`.
15. Treat design-space kind lists as representational vocabulary, not as a mandate to implement every listed kind in the current phase.
16. If a phase needs a file outside its allowed list, stop and update the phase note or ask for direction before editing it.
17. Prefer the smallest executable slice that satisfies the current phase completion criteria.

## Phase Protocol

For each phase:

1. Read `D:\jongseol\docs\archive\migration-phases\README.md`.
2. Read the current phase document.
3. Inspect existing files before editing.
4. Modify only files listed in `Files to create` or `Files allowed to modify`.
5. Do not modify files listed in `Files not allowed to modify`.
6. Add or update tests in the same phase.
7. Run phase-specific tests.
8. Run full tests.
9. Update `D:\jongseol\docs\archive\migration-phases\CURRENT_STATUS.md`.

## Scope Decision Rules

Use these rules when a phase note feels ambiguous:

| Situation | Agent action |
|---|---|
| A feature appears in the design-space vocabulary but not the phase scope | Record it as `interface-target` or leave it documented only. |
| A prior-work method is named | Implement only the proxy/interface behavior listed in the current phase. |
| A module cannot support a requested behavior yet | Return or document explicit `unsupported` with `support_level`; do not silently fallback. |
| A needed edit is outside `Files allowed to modify` | Do not edit it until the phase document is updated or the user approves the scope change. |
| Tests require broad rewrites outside the phase | Add a blocker to `CURRENT_STATUS.md` instead of expanding the phase silently. |

## Report Format

At the end of each phase, report:

- Summary
- Files changed
- Tests run
- Behavior changes
- Known limitations
- Next phase blockers

If `CURRENT_STATUS.md` does not exist, add it with a short template:

```text
# Current Status

## Current Phase

## Completed

## In Progress

## Blockers

## Last Tests Run

## Next Step
```
