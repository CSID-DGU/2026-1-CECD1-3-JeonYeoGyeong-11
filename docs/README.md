# Docs Index

## Canonical Docs

| 문서 | 용도 |
|---|---|
| `framework/overview.md` | project claim, research position, experiment design |
| `framework/metrics.md` | metric 정의, 수식, diagnostic 해석 |
| `framework/components.md` | lifecycle, interface, extension guide, prior-work mapping |
| `framework/evidence.md` | framework-quality evidence와 provenance |
| `maintenance/migration-and-compatibility.md` | migration, compatibility, gate-check contract |
| `archive/README.md` | archive summary |

## Folder 역할

| Folder | 용도 |
|---|---|
| `framework/` | 현재 claim, metric, component, evidence |
| `research/` | prior work와 design note |
| `maintenance/` | migration, compatibility, gate 기록 |
| `archive/` | 이전 방향과 migration phase 기록 |

## 실행 확인

Repository root 기준 command:

| Check | Command |
|---|---|
| unit tests | `python -m unittest discover -s tests` |
| diagnostic preflight | `python scripts/checks/diagnostic_suite_preflight.py` |
| vision smoke | `python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json` |
| Cora ablation smoke | `python run_graph_ablation.py --config configs/cora/ablations/graph/graph_ablation_smoke.json` |
| evidence smoke | `python scripts/validation/graph_evidence_report.py --profile smoke --include-external --out-dir <out-dir>` |

## 문서 작성 규칙

| 규칙 | 기준 |
|---|---|
| 언어 | 한국어 중심, 핵심 기술 명사만 English |
| 경로 | repository root 기준 relative path |
| claim | 주장과 근거 artifact를 함께 기록 |
| report | row-level CSV/JSON을 primary evidence로 둠 |
