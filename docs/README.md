# Docs Index

Use `framework/` for current work. Use `research/` for background notes. Use
`archive/` only for prior direction or migration history.

현재 문서는 세 갈래로 구성된다. `framework/`는 주장의 형태와 실험 설계를
담고, `structure.md`는 실제 파일과 실행 경로를 찾는 지도 역할을 한다. 발표나
공유가 필요할 때는 `capstone-prior-work-positioning.md`와
`demos/graphfl-assembly-scratch.html`이 프로젝트의 의미와 조립식 구조를 가장
빠르게 보여준다.

| Document | Role |
|---|---|
| [README.md](README.md) (this file) | docs index and run-path checklist |
| [structure.md](structure.md) | **detailed** repo map, scripts, tests, edit routing |
| [removed-materials.md](removed-materials.md) | Gate 6 removals and read-only aliases |
| [../README.md](../README.md) | install, quick start, **abbreviated** repo layout |
| [capstone-prior-work-positioning.md](capstone-prior-work-positioning.md) | 선행연구 흐름에서 프로젝트가 자연스럽게 나오는 이유와 현재 구현 범위 |

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
| project story / prior work position | [capstone-prior-work-positioning.md](capstone-prior-work-positioning.md) |
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

All commands are repository-relative. Use the Python interpreter from the active
environment; the docs avoid machine-specific paths.

| Step | Command |
|---|---|
| Unit tests | `python -m unittest discover -s tests` |
| Suite preflight | `python scripts/checks/diagnostic_suite_preflight.py` |
| Vision smoke | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| Result bundle | `python scripts/checks/result_evidence_bundle.py <result.json> --kind single-run` |
| Cora ablation smoke | `python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json` |
| Gate check | `python scripts/dev/run.py gate-check 6` |

| Run Type | Role |
|---|---|
| single vision run | one graph design, one config, one result JSON |
| diagnostic suite | multiple variants under one environment, used for real/control/graph-free comparison |
| Cora ablation | graph-structured input path smoke and summary validation |
| result evidence check | confirms schema fields needed for attribution are present |

New configs should keep the chosen `track`, `dataset`, `graph_source`,
`graph_mode`, `aggregation_target`, `correction_family`, controls, and
diagnostics explicit. This makes the result readable later without relying on
local notes or unstated defaults.

## Presentation Demo

| Artifact | Purpose |
|---|---|
| [demos/graphfl-assembly-scratch.html](demos/graphfl-assembly-scratch.html) | graph_source, relation cue, graph_mode, aggregation target, control, diagnostic을 블록처럼 조립해 실제 config JSON 흐름으로 보여주는 발표용 데모 |

이 데모는 repo 실행 축과 맞춘 시각적 조립 화면이다. palette의 주요 블록은 현재
`graphfl_lab`의 실행 축을 따르며, custom dataset/config/graph component 이름을
직접 넣어 새 graph 후보를 같은 구조에 얹는 흐름을 보여준다. 비교 열은 같은
환경에서 real graph, controls, graph-free correction을 나란히 놓는 방식으로
자라며, 발표에서는 “그래프가 효과가 있었는가”와 “그렇다면 어떤 구성요소 때문인가”를
한 화면에서 설명할 수 있다.

## Document Policy

| Area | Rule |
|---|---|
| `framework/` | active claim, experiment design, interface docs |
| `research/` | literature review and design notes; may mention legacy names historically |
| `archive/` | frozen migration/previous direction; **not** current run policy |
| `maintenance/` | gate inventory and closed cleanup log |
| root `README.md` | install, run, abbreviated layout → detail in `structure.md` |
