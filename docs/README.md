# Documentation Index

`docs/`는 repository 기준 공식 문서와 HTML demo를 담는다. Markdown 문서는 root에 모으고, demo artifact는 `docs/demos/` 아래에 둔다.

## Canonical Docs

| Document | 답하는 질문 | 핵심 근거 |
|---|---|---|
| `framework.md` | Graph-FL gain을 어떤 component와 metric으로 분해하는가 | lifecycle, `GraphFLDesign`, metric formula, artifact fields |
| `evidence.md` | framework가 왜 정당한가 | experiment case matrix, 18 graph modes parity, 8,640 design-space checks, 4 / 4 extension checks |
| `research.md` | prior work와 어떤 위치 관계를 갖는가 | FedAMP, SFL, pFedGraph, FedAGA mapping |
| `repository.md` | code/config/script/test/docs가 어디에 있는가 | repository tree, package layout, change routing |
| `maintenance.md` | public surface와 compatibility 기준은 무엇인가 | canonical surface, alias table, golden baseline policy |
| `history.md` | 이전 실험 관찰이 현재 claim으로 어떻게 이어졌는가 | Phase 1-3 observation, migration phases |
| `demos/graphfl-assembly-scratch.html` | Graph-FL assembly를 시각적으로 확인하는 demo는 어디에 있는가 | HTML demo artifact |

## Reading Order

1. `framework.md`: 현재 claim, lifecycle, component contract.
2. `evidence.md`: framework 정당성을 뒷받침하는 실험과 수치.
3. `research.md`: prior work positioning과 design pattern survey.
4. `repository.md`: code, script, config, tests 위치와 change routing.
5. `maintenance.md`: compatibility와 repository policy.
6. `history.md`: 과거 실험 관찰과 migration phase 결정.
7. `demos/graphfl-assembly-scratch.html`: Graph-FL assembly demo 확인.

## Verification

Repository root에서 실행한다.

| Check | Command |
|---|---|
| unit tests | `python -m unittest discover -s tests` |
| diagnostic preflight | `python scripts/checks/diagnostic_suite_preflight.py` |
| vision smoke | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| Cora ablation smoke | `python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json` |
| Evidence smoke | `python scripts/validation/graph_evidence_report.py --profile smoke --include-external --out-dir <out-dir>` |
