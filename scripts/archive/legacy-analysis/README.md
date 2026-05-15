# Legacy Analysis Scripts

These scripts belong to the project direction before the lifecycle design framework.

They are preserved for reproducing older phase reports and transition history, but new framework code should not depend on them. Prefer the current diagnostic suite, lifecycle modules, and future `GraphFLDesign`-based tooling for new experiments.

Default report paths point to `docs/archive/legacy-phase-reports/` so running
one of these scripts does not recreate old `PHASE*.md` files in the repository
root.

Moved scripts:

- `phase1_diagnostics_report.py`
- `phase2_graph_informativeness.py`
- `phase2_graph_source_sanity_suite.py`
- `phase2_5_smoothing_failure.py`
- `phase3_dominance_aware.py`
- `graph_preset_smoke_test.py`
- `pathology_graph_case_study.py`
- `pathology_graph_case_smoke.py`
