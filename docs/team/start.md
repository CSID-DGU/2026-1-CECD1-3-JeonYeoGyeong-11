# 팀 작업 시작

판매자별 거래로 추천하는 커머스 플랫폼을 만든다. 거래 저장·추천·학습은 판매자 환경에서 수행하고, 중앙은 공개 상품 정보와 보호 집계·공통 모델 배포를 맡는다. 공통 모델과 판매자별 후반부 개인화를 구분하며, 기본 실험은 한 PC의 여러 논리적 판매자로 실행한다. 세부 기준은 [아키텍처](../design/architecture.md)와 [모델 경계](../design/model.md)에 있다.

현재는 계약·기동 뼈대 단계다. 아래 첫 작업도 최신 코드·PR에서 완료 여부를 확인한 뒤 남은 부분부터 진행한다.

## 0. 필요한 만큼 읽고 시작하기

공통 진입점은 [AGENTS.md](../../AGENTS.md) → 이 안내 → **자기 역할 카드 하나**다. 역할 카드의 첫 작업에서 이번에 끝낼 작은 동작을 고른 뒤 아래 표의 관련 절과 실제 코드를 읽는다. 링크를 따라 모든 설계 문서를 미리 읽거나 다른 역할 카드 전체를 읽을 필요는 없다.

역할은 사용자의 지정과 역할 카드 2행의 계정으로 확인한다. 둘 다 없거나 서로 충돌하면 사용자에게 확인하고, 임의로 다른 담당의 구현을 시작하지 않는다. 카드가 `미선점`이면 작업 PR을 열기 전에 [작업 규칙](working-agreement.md) §1의 선점 PR을 연다.

- 공통 절차: [분업 표](working-agreement.md) §1에서 소유권을 확인하고 [Git 규약](git-workflow.md)의 세션 시작 절차를 수행한다. PR·리뷰·병합 시에는 그에 해당하는 절을 확인한다.
- 작업별 참고: 설치는 [개발 안내](../development.md), DB·환경변수·게이트 변경은 작업 규칙 §3~4, 계약 변경은 해당 schema·Python 타입·동작 명세·소비자 코드를 함께 확인한다. 각 역할 카드의 나머지 링크는 해당 기능을 구현할 때 읽는다.
- 이어가기: 최신 PR의 다음 행동·담당·검사 SHA와 관련 diff부터 본다. 필요한 기준 문서가 바뀌었거나 의미가 불확실하면 해당 절을 다시 읽는다. 긴 전체 요약이나 별도 상태 문서를 새로 만들지 않는다.
- 착수 보고는 `이번 동작 / 수정할 소유 경로 / 사용할 계약·stub / 완료 검사`를 짧게 적으면 된다. 관련 [미확정 항목](../design/open-questions.md)과 실제 구현 상태도 확인하고, 결정이 필요한 부분만 보류한다. 카드의 전체 완료 목록을 한 번에 구현하지 않는다.

## 1. 같은 코드 준비

초기 코드는 `main`에 있다. 각 에이전트는 자기 머신에 별도 clone을 만들고 자기 역할의 작업 브랜치에서 시작한다.

```text
git clone https://github.com/CSID-DGU/2026-1-CECD1-3-JeonYeoGyeong-11.git
cd 2026-1-CECD1-3-JeonYeoGyeong-11
git switch -c commerce/a/first-order origin/main
```

마지막 브랜치는 자신의 역할/작업명으로 바꾼다. 분기 기준은 현재 HEAD가 아니라 `origin/main`이다(아직 병합되지 않은 PR 위에서 일할 때는 [Git 협업](git-workflow.md)의 '의존 PR 쌓기'). 다른 브랜치가 체크아웃된 폴더에서 HEAD로 분기하면 남의 커밋이 PR에 딸려 간다. PR의 base는 항상 `main`이다. 기존 작업 폴더를 쓰는 경우 미커밋 변경을 먼저 확인하고 `git fetch origin` 뒤에 분기한다.

설치·최초 검사는 [개발 안내](../development.md)를 따른다. Git 작성자 이름·이메일은 각자 본인 것으로 설정한다. 별도 ZIP이나 비공개 문서는 필요 없다.

## 2. 역할과 첫 작업

역할별 GitHub 계정은 각 역할 카드 2행에 한 번만 기록한다(실명 없음, 절차는 [작업 규칙](working-agreement.md) §1).

| 역할 | 첫 작업에 필요한 참조 | 먼저 보여줄 결과 | 이후 상대에게 확인할 것 |
| --- | --- | --- | --- |
| A | [A 카드](tasks/A.md), [merchant 진입점](../../commerce/services/merchant_api/README.md)·현재 코드, [인터페이스](../design/interfaces.md) §1~3부터. 추천 연결 시 §4 | 합성 주문 한 건의 상태·catalog/purchase_event 예제와 작은 검사 | B의 영속 반영·상품 전달 순서, C 기동 설정 |
| B | [B 카드](tasks/B.md), [어댑터](../../commerce/packages/data_adapters/README.md)·[추천 runtime](../../commerce/packages/recommender/README.md)·현재 코드, [데이터](../design/data.md)의 해당 입력 절. NLP 구현 시 [모델](../design/model.md) §1~2·7, 학습 전 손실·평가 규칙 | 두 출처/live 공통 입력·text builder 예제 | A 상품 매핑, C manifest·학습 반환 |
| C | [C 카드](tasks/C.md), [FL client 진입점](../../commerce/packages/fl_client/README.md)·현재 코드, [인터페이스](../design/interfaces.md) §1·5~8. 보호 조사 시 [아키텍처](../design/architecture.md) §2~3 | 더미 tensor 라운드의 작은 예제와 보호 후보의 실행 가능성 | B 실제 tensor 계약, A lifespan/실행 환경 |

위 결과는 먼저 피드백하기 좋은 단위다. 더 넓은 범위를 맡았다면 독립적으로 가능한 구현을 연속 진행해도 된다. 작은 예제 성공을 a1/b1/c1 전체 완료로 보고하지 않는다.

## 3. 진행 원칙

- 리뷰·담당자 수정·에이전트 병합은 [Git 규약](git-workflow.md)을 따른다. 해당 규칙을 별도 문서에 복사하지 않는다.
- 약 2주 완성이 목표지만 날짜별 진도나 전원 대기를 강제하지 않는다. 가능한 날에는 여러 단위를 진행하고 결과가 생기면 비동기로 피드백한다.
- 첫 완성은 최소 화면·주문 왕복·두 출처의 작은 고정 표본·텍스트/관계/개인화 비교·두 콜드스타트·보호 FL다. 전체 roster, 다수 seed, 고급 UI, 클라우드/자동 스케줄러는 후순위다.
- A는 B 대신 계약 stub, B는 C 대신 단일 판매자 로컬 실험, C는 B 대신 더미 trainer로 독립 진행할 수 있다. 실제 연결 검사는 나중에 별도로 한다.
- 인터페이스/모델 의미/보호 조건 변경이 필요한 부분만 보류하고 근거·추천안·소비자 영향을 남긴다. 응답이 없다는 것을 승인으로 해석하지 않는다.
- 실데이터 유래 중앙 FL는 보호 경로 검증 전 실행하지 않는다. 실험 규모를 줄여도 데이터 경계·시점 누출 방지·이벤트 정확성은 유지한다.
- [열린 구현 항목](../design/open-questions.md)은 해당 구현 전에 해결한다. 이것 때문에 전원 작업을 멈출 필요는 없다.

## 4. 다음 사람이 이어받을 때

인계 형식은 [PR 템플릿](../../.github/PULL_REQUEST_TEMPLATE.md) 하나다. 템플릿의 각 절을 짧게 채우고, '다음 작업 / 인계'의 첫 줄 `현재 상태 / 다음 행동 / 다음 담당 계정 / 관련 SHA·Issue`는 [Git 협업](git-workflow.md)의 상태 표기를 쓴다. 팀원이 결과를 읽고 필요한 검사를 재현한다. 코드 생성과 검증 완료는 구분한다.

새 에이전트에 전달할 정보도 역할·작업 범위·기준 PR/SHA와 그 PR의 인계면 충분하다. 공통 규칙은 루트 AGENTS.md를 사용한다. Git 업로드와 병합은 [Git 규약](git-workflow.md)을 따른다.

## 5. 약어

역할 카드와 설계 문서에 나오는 ID다. 뜻만 알면 되는 경우 여기서 멈추고, 구현에 필요할 때 정의 문서의 해당 절을 읽는다.

| 약어 | 뜻 | 정의 |
| --- | --- | --- |
| a1·b1·b2·c1 | 역할별 업무 게이트 | [작업 규칙](working-agreement.md) §4 |
| G1, g2·g3·g4 (G2~G4) | 통합 게이트. G1은 `contracts` | [작업 규칙](working-agreement.md) §4 |
| D0017~D0019 | 채택한 설계 결정 | [결정 요약](../design/decisions.md) |
| OQ01~ | 구현 전에 확정할 열린 항목 | [열린 구현 항목](../design/open-questions.md) |
| text_only · text_relation | 두 모델 variant. 텍스트만 / 텍스트와 로컬 구매 관계 | [모델 비교](../design/comparison.md) §2 |
| T-G·R-G·T-P·R-P | 2×2 비교 결과. T=text_only, R=text_relation, G=공통 모델, P=후반부 개인화 | [모델 비교](../design/comparison.md) §2 |
| z | 고정 NLP 인코더가 만든 상품 텍스트 벡터 | [모델 경계](../design/model.md) §3 |
| E-G0 | 텍스트만 기준선의 소형 첫 실행. NLP·비용·입력 품질을 잰다 | [평가](../design/evaluation.md) §1 |
| R1 · R2 | 전체 roster 실험. 두 출처 플랫폼 학습 / 보류 판매자 이전. 후순위 | [평가](../design/evaluation.md) §5 |
| A-0 · A-few | 신규 판매자. 학습 미참여로 공통 모델만 / 소수 step 개인화 | [평가](../design/evaluation.md) §3 |
| C-new · C0 | 신상품. 학습에서 보류한 상품 / 관계만 0으로 가린 보조 분석(신상품 증거 아님) | [평가](../design/evaluation.md) §3 |
| `CONTRACTS.md §8`, `R6`, `SC4`, `MODEL_BOUNDARY.md` | 스키마 설명·코드 주석에 남은 옛 문서명과 규칙 번호. 뜻은 인터페이스·모델 경계 문서에 있다 | [계약 문서](../contracts.md) '기존 설계 참조' |
