# Agent Implementation Guide Archive

## 목적

Migration phase 진행 시 agent가 따르던 작업 규칙을 보관한다.

## Workflow

| Step | 내용 |
|---:|---|
| 1 | `CURRENT_STATUS.md`에서 현재 phase 확인 |
| 2 | 해당 phase 문서의 scope 확인 |
| 3 | allowed file과 verification 기준 확인 |
| 4 | 작은 단위로 구현 |
| 5 | test 결과와 상태 업데이트 |

## 핵심 규칙

| Rule | 기준 |
|---|---|
| lifecycle boundary | state, relation, topology, aggregation 역할 분리 |
| compatibility | public surface 변경은 phase scope에 맞춤 |
| trace | unsupported 상태와 support level을 명시 |
| diagnostics | counterfactual path는 server model update와 분리 |
| design | graph-FL method를 component 조합 중심으로 표현 |

## 현재 연결

현재 작업 기준은 `docs/framework/project-prompt.md`와 `docs/structure.md`를 따른다.
