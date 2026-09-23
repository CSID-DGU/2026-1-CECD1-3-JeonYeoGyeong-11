# 프로젝트 문서

이 디렉터리가 현재 설계와 협업의 기준이다. **별도 내부 ZIP 없이 저장소만으로 시작할 수 있다.** 과거 초안·전체 검토 이력·요청문 모음은 로컬 보관하며 현행 구현 지시로 사용하지 않는다.

## 처음 읽을 것

1. [개발 안내](development.md): 설치·검사·공통 import·현재 구현 범위.
2. [팀 시작 안내](team/start.md): 담당별 첫 작업과 독립 진행 방법.
3. 자신의 [A 플랫폼](team/tasks/A.md) / [B 데이터·모델](team/tasks/B.md) / [C FL·통합](team/tasks/C.md) 카드.

처음부터 아래 모든 문서를 읽을 필요는 없다. 담당 작업에 필요한 계약·설계 절을 찾아 읽고, 이어가기에서는 변경 diff와 남은 작업을 확인한다.

## 현재 설계

| 문서 | 다루는 내용 | 주 독자 | 이전 문서명 |
| --- | --- | --- | --- |
| [아키텍처](design/architecture.md) | 그림·주문 흐름·중앙/판매자 데이터 경계 | 모두 | ARCHITECTURE |
| [인터페이스 계약](design/interfaces.md) | 주문/event·카탈로그·추천·FL 제출/배포 | 연결 담당 | CONTRACTS |
| [모델 경계](design/model.md) | NLP·관계·공유 가중치·개인화·동시성 | B, C | MODEL_BOUNDARY |
| [데이터](design/data.md) | 출처 한계·전처리·텍스트·합성 fixture | B | DATA |
| [평가](design/evaluation.md) | 시점·두 콜드스타트·기준선·지표 | B | EVALUATION |
| [독립 모델 실험](design/model-lab.md) | 화면 없는 학습·checkpoint·초기 release | B, C | LAB_WORKFLOW |
| [모델 비교](design/comparison.md) | T-G/R-G/T-P/R-P·동일 snapshot 시연 | A, B | COMPARISON |
| [채택한 설계 결정](design/decisions.md) | D0017~D0019 및 초기 뼈대 선택 이유 | 필요한 경우 | 현행 결정 요약 |
| [열린 구현 항목](design/open-questions.md) | 미확정 부분·담당·완료 증거 | 해당 담당 | 현행 검토 항목 |

본문의 CONTRACTS/MODEL_BOUNDARY 등 약칭은 위 공개 문서를 가리킨다. JSON 필드/타입은 [스키마와 검사](contracts.md), Python 호출은 `commerce/packages/contracts/ports.py`와 `types.py`를 기준으로 한다. 기술 설명과 코드가 충돌하면 관련 소비자가 함께 수정한다.

## 팀 운영

- [분업·런타임·완료 기준](team/working-agreement.md): 소유 경로·DB·포트·환경변수·게이트.
- [Git 협업](team/git-workflow.md): 브랜치·변경 확인·작은 PR·검토·병합.
- [개발 도구 공통 지침](../AGENTS.md): 에이전트도 같은 계약·소유권·검사를 사용한다.

## 구현 상태와 공개 범위

계약 검증과 초기 공통 인터페이스/health 뼈대가 있다. 문서에 적힌 주문·NLP·학습·추천·FL·보호 집계는 목표 설계이며 대부분 아직 미구현이다. 완료 여부는 코드·게이트·PR 결과로 판단한다.

코드·현행 설계·합성 계약 예제·점포 ID만 있는 작은 기준 roster를 공개한다. 원자료·고객별 배정/파생물·모델 파일·DB·인증 정보는 제외한다. 논문 PDF와 분석 결과물(`fedcommerce/out/`)도 제외한다.

B가 어댑터를 처음부터 재작성하지 않도록 이전 탐색 분석 스크립트만 [fedcommerce/](../fedcommerce/README.md)에 함께 둔다. 현행 계약을 따르지 않는 참고 자료이며 게이트 대상이 아니다. 원자료 없이는 실행되지 않는다.
