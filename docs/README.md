# Docs Index

Use `framework/` for current work. Use `research/` for background notes. Use `archive/` only for prior direction or migration history.

## Layout

```text
docs/
├── README.md
├── structure.md
├── framework/
│   ├── claim.md
│   ├── graph_fl_experimental_design.md
│   ├── graph_fl_experimental_design_appendix.md
│   ├── diagnostics.md
│   ├── interfaces.md
│   ├── lifecycle.md
│   ├── prior-work-mapping.md
│   ├── extension-guide.md
│   ├── cleanup-plan.md
│   ├── naming-and-compatibility.md
│   ├── project-prompt.md
│   └── experiment-results.md
├── research/
└── archive/
```

## First Read

| Need | Document |
|---|---|
| project direction | [framework/claim.md](framework/claim.md) |
| experimental design | [framework/graph_fl_experimental_design.md](framework/graph_fl_experimental_design.md) |
| metric definitions | [framework/graph_fl_experimental_design_appendix.md](framework/graph_fl_experimental_design_appendix.md) |
| install and run | [../README.md](../README.md) |
| edit routing | [structure.md](structure.md) |
| add graph algorithm | [framework/interfaces.md](framework/interfaces.md), [framework/extension-guide.md](framework/extension-guide.md) |
| prior-work proxy boundary | [framework/prior-work-mapping.md](framework/prior-work-mapping.md) |
| diagnostic interpretation | [framework/diagnostics.md](framework/diagnostics.md) |
| compatibility names | [framework/naming-and-compatibility.md](framework/naming-and-compatibility.md), [framework/cleanup-plan.md](framework/cleanup-plan.md) |
| handoff prompt | [framework/project-prompt.md](framework/project-prompt.md) |

## Current Run Path

```powershell
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json
python scripts/dev/run.py gate-check 5d-prep
```

## Document Policy

| Area | Rule |
|---|---|
| `framework/` | active claim, experiment design, interface docs |
| `research/` | literature review and design notes |
| `archive/` | previous direction and migration phases |
| root `README.md` | install, run, repository map |
