# Legacy Analysis Scripts

이 directory는 현재 claim을 만들기 전 phase reports와 transition history를 재현하기 위한 historical analysis scripts를 보관한다.

## Role

| Surface | Role |
|---|---|
| scripts | older phase report reproduction |
| current summary | `docs/history.md` |
| generated output | caller-selected output directory |

## Moved Scripts

| Script | Historical Role |
|---|---|
| `phase1_diagnostics_report.py` | Phase 1 diagnostic metric summary |
| `phase2_graph_informativeness.py` | Phase 2 graph/control comparison |
| `phase2_graph_source_sanity_suite.py` | source sanity and artifact path check |
| `phase2_5_smoothing_failure.py` | smoothing-only effect check |
| `phase3_dominance_aware.py` | dominance-aware baseline check |
| `graph_preset_smoke_test.py` | graph preset smoke check |
| `pathology_graph_case_study.py` | pathology graph case study |
| `pathology_graph_case_smoke.py` | pathology graph smoke case |
