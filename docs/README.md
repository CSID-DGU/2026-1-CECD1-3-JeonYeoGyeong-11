# Docs Index

이 디렉터리는 현재 프레임워크 문서, 연구 노트, 보관 자료를 분리한다.
새 작업을 시작할 때는 `framework/`를 먼저 보고, 과거 판단이나 migration
이력이 필요할 때만 `archive/`를 본다.

## Layout

```text
docs/
├── README.md                         # 문서 index
├── structure.md                      # 저장소 구조와 변경 위치 규칙
│
├── framework/                        # 현재 프로젝트를 실행하고 확장하기 위한 문서
│   ├── claim.md                      # 현재 연구 claim과 boundary
│   ├── diagnostics.md                # 진단 지표 해석 규칙
│   ├── interfaces.md                 # 조립식 graph algorithm 인터페이스
│   ├── lifecycle.md                  # graph-FL lifecycle decomposition
│   ├── prior-work-mapping.md         # 선행연구 exact/proxy/interface 경계
│   ├── extension-guide.md            # graph source/builder 추가 방법
│   ├── cleanup-plan.md               # 정리 우선순위와 risky rename backlog
│   ├── naming-and-compatibility.md   # 남겨둔 옛 이름과 정리 계획
│   ├── project-prompt.md             # 이후 작업자/agent용 전체 프롬프트
│   └── experiment-results.md         # 기존 실험 결과와 claim boundary
│
├── research/                         # 문헌 검토와 설계 노트
│   ├── prior-work-review.md
│   ├── framework-design-notes.md
│   └── design-pattern-survey.md
│
└── archive/                          # 낮은 우선순위의 보관 자료
    ├── previous-direction/
    ├── legacy-phase-reports/
    └── migration-phases/
```

## First Read

| 목적 | 문서 |
|---|---|
| 처음 설치하고 실행 | [../README.md](../README.md)의 `Quick Start` |
| 저장소 어디를 고칠지 판단 | [structure.md](structure.md) |
| 프로젝트 claim 파악 | [framework/claim.md](framework/claim.md) |
| 새 graph algorithm 추가 | [framework/interfaces.md](framework/interfaces.md), [framework/extension-guide.md](framework/extension-guide.md) |
| 조립식 부품 구조 빠르게 파악 | [../README.md](../README.md)의 `Assembly Model` |
| 선행연구 proxy 경계 확인 | [framework/prior-work-mapping.md](framework/prior-work-mapping.md) |
| 진단 지표 해석 | [framework/diagnostics.md](framework/diagnostics.md) |
| 남은 이름/경로 정리 계획 | [framework/naming-and-compatibility.md](framework/naming-and-compatibility.md), [framework/cleanup-plan.md](framework/cleanup-plan.md) |
| 다음 작업자에게 전체 맥락 전달 | [framework/project-prompt.md](framework/project-prompt.md) |

## Current Run Path

실행 명령과 가장 작은 smoke 경로는 repository root의
[README.md](../README.md)에 둔다.

현재 기본 확인 순서:

```powershell
python -m unittest discover -s tests
python scripts/checks/diagnostic_suite_preflight.py
python run_vision_experiment.py --config configs/vision/smoke/default_similarity_knn.json
```

## Document Policy

| 구분 | 규칙 |
|---|---|
| `framework/` | 현재 코드와 직접 맞물리는 문서만 둔다. |
| `research/` | 문헌 검토, 설계 판단, 배경 노트를 둔다. |
| `archive/` | 현재 우선순위에서 내려간 과거 방향과 migration phase를 둔다. |
| root `README.md` | 처음 보는 사람이 실행 경로와 구조를 잡는 문서로 유지한다. |

오래된 문서를 삭제하기 위험하면 `archive/`로 옮기고, 현재 문서에서는 링크나
compatibility note만 남긴다.
