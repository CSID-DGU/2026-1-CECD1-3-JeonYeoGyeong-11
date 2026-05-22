# Docs Index

Use `framework/` for current work. Use `research/` for background notes. Use
`archive/` only for prior direction or migration history.

| Document | Role |
|---|---|
| [README.md](README.md) (this file) | docs index and run-path checklist |
| [structure.md](structure.md) | **detailed** repo map, scripts, tests, edit routing |
| [removed-materials.md](removed-materials.md) | Gate 6 removals and read-only aliases |
| [../README.md](../README.md) | install, quick start, **abbreviated** repo layout |

Post-Gate-6 rename/cleanup is **complete** on `main` (2026-05-22). Active policy:
[naming-and-compatibility.md](framework/naming-and-compatibility.md).

## Layout

```text
docs/
├── README.md                         this index
├── structure.md                      detailed repository map (files + scripts + tests)
├── removed-materials.md              Gate 6 removals and remaining read-only aliases
├── framework/                        active framework documentation
│   ├── claim.md                      project claim and non-goals
│   ├── graph_fl_experimental_design.md   main experiment design
│   ├── graph_fl_experimental_design_appendix.md   metric definitions
│   ├── diagnostics.md                how to read diagnostic traces
│   ├── interfaces.md                 GraphFLDesign, graph_source, aggregation_target
│   ├── lifecycle.md                  lifecycle modules and contracts
│   ├── prior-work-mapping.md         exact vs proxy vs out-of-scope prior work
│   ├── extension-guide.md            add graph_source / builder workflow
│   ├── cleanup-plan.md               rename execution summary (closed)
│   ├── naming-and-compatibility.md   canonical names vs read-only aliases
│   ├── project-prompt.md             handoff prompt for agents/maintainers
│   └── experiment-results.md         how to store and cite experiment outputs
├── maintenance/                      gate checks and rename inventory (cleanup closed)
│   ├── cleanup-status.md             Gate 6 execution log (closed)
│   ├── gate-6-prep.md                Gate 6 checklist (all done)
│   ├── rename-inventory.md           legacy name inventory (historical + post-Gate-6 note)
│   └── last_gate_check.json          latest gate-check record
├── research/                         literature and design notes (non-normative)
└── archive/                          superseded direction and migration phases (historical)
```

## First Read

| Need | Document |
|---|---|
| project direction | [framework/claim.md](framework/claim.md) |
| experimental design | [framework/graph_fl_experimental_design.md](framework/graph_fl_experimental_design.md) |
| metric definitions | [framework/graph_fl_experimental_design_appendix.md](framework/graph_fl_experimental_design_appendix.md) |
| install and run | [../README.md](../README.md) |
| **file-level map** | [structure.md](structure.md) |
| edit routing | [structure.md](structure.md) (Change Routing) |
| add graph algorithm | [framework/interfaces.md](framework/interfaces.md), [framework/extension-guide.md](framework/extension-guide.md) |
| prior-work proxy boundary | [framework/prior-work-mapping.md](framework/prior-work-mapping.md) |
| diagnostic interpretation | [framework/diagnostics.md](framework/diagnostics.md) |
| compatibility / removals | [framework/naming-and-compatibility.md](framework/naming-and-compatibility.md), [removed-materials.md](removed-materials.md) |
| handoff prompt | [framework/project-prompt.md](framework/project-prompt.md) |

## Current Run Path

| Step | Command |
|---|---|
| Unit tests | `python -m unittest discover -s tests` |
| Suite preflight | `python scripts/checks/diagnostic_suite_preflight.py` |
| Vision smoke | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| Result bundle | `python scripts/checks/result_evidence_bundle.py <result.json> --kind single-run` |
| Cora ablation smoke | `python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json` |
| Gate check | `python scripts/dev/run.py gate-check 5d-prep` |

## Document Policy

| Area | Rule |
|---|---|
| `framework/` | active claim, experiment design, interface docs |
| `research/` | literature review and design notes; may mention legacy names historically |
| `archive/` | frozen migration/previous direction; **not** current run policy |
| `maintenance/` | gate inventory and closed cleanup log |
| root `README.md` | install, run, abbreviated layout → detail in `structure.md` |
