# A 작업 카드 — 플랫폼·주문
담당 계정: 미선점 · 선점은 [작업 규칙](../working-agreement.md) §1 · 실행 기준 [D0017~D0019](../../design/decisions.md) · 약어는 [팀 시작 안내](../start.md) §5

## 목표와 소유
구매자/판매자 화면과 판매자 로컬 주문을 만들고 B의 실제 추천을 화면에 연결한다.
소유: commerce/apps/buyer/, commerce/apps/seller/, commerce/services/central_api/, commerce/services/merchant_api/.
첫 읽기는 [팀 시작 안내](../start.md) §0·2를 따른다. 아래는 기능별 참고이며 전부 선독할 목록이 아니다: 소유권·환경은 [작업 규칙](../working-agreement.md), 배치·데이터 경계는 [아키텍처](../../design/architecture.md), 주문·추천 연결은 [인터페이스 계약](../../design/interfaces.md) §1~4.
지속 작업: [Git 협업](../git-workflow.md)에 따라 작업별 브랜치와 Draft PR을 사용한다. 시작 때 B의 이벤트/추천 계약 변경과 C의 실행 훅 변경을 확인한다.

## 첫 작업
아래는 피드백을 거쳐 나눠 수행할 초기 순서다. 먼저 팀 시작 안내의 A 첫 작업부터 확인하고 맡은 범위 안에서 이어간다. 1번의 화면·HTTP 인증·공개 목록은 [열린 항목](../../design/open-questions.md)의 OQ13~15, OQ19(사람 결정 포함)에 걸린다. 그 결정 전에는 주문 상태 전이·transaction·outbox 같은 도메인 계층부터 만든다.

1. 합성 판매자·고객·상품으로 공개 목록 → 판매자 구매 화면 → 주문 requested를 만든다.
2. 판매자 accepted/completed/cancelled, 권한·상태 버전·idempotency를 구현한다.
3. 완료 주문+purchase_event+outbox를 한 DB transaction으로 저장한다.
4. 자기 test double로 전달·재전달·중복 성공 경로를 만든다. B 구현이 오면 runtime API로 연결한다. 저장소의 `UnimplementedRuntime`은 항상 예외를 내므로 성공 경로에 쓸 수 없다. 주입 방법과 금지 사항은 [개발 안내 §상대 모듈을 대체하는 방법](../../development.md#상대-모듈을-대체하는-방법)을 따른다.
5. catalog upsert도 증가 source_seq로 전달한다. 신규 상품 텍스트가 B NLP로 들어가는 것을 확인한다.

첫 산출물: 실행 가능한 화면·API, 합성 seed, 주문 상태/이벤트 예제, a1 selfcheck.
완료 전 B와 event ID·정상 반환의 영속성·source_seq 의미를 함께 확인한다.

## 경계와 통합
- A는 orders.sqlite만 쓴다. 특징·cache·모델은 B API만 사용한다.
- 입력/출력은 catalog_item, purchase_event, recommendation_request/recommendation.
- 가격과 상태는 서버가 결정한다. 구매 화면을 중앙 origin에서 렌더하지 않는다.
- C run_local.py를 수정하지 않고 앱 lifespan에 FL client start/stop 연결점을 제공한다.
- [독립 모델 실험](../../design/model-lab.md)의 초기 release를 읽어 서비스 시작 즉시 추천한다. 전체 사전학습을 앱 startup에 넣지 않는다.
- 고객 계정/주문/추천은 판매자 origin, host-only 쿠키와 CSRF. 중앙에는 공개 snapshot만.
- 같은 완료 요청 재시도, DB commit 직후 종료, B 반영 뒤 ack 전 종료를 복구한다.

## 완료 판정
a1 → B 연결 뒤 g2. g2는 실제 B 추천·특징 반영 1회·장애 복구·중앙 로그/DB 비잔류를 확인한다.
NLP/FL 미구현 stub을 사용한 경우 결과에 명시하고 g2/g3 통과라고 쓰지 않는다.

## 모델 비교 화면
- [모델 비교](../../design/comparison.md)과 [인터페이스 계약](../../design/interfaces.md) §4.1을 읽고 판매자 origin에서 네 결과를 표시한다. 일반 추천 연결 후 추가하며 B 준비 전에는 로컬 ComparisonResult stub으로 화면을 만든다.
- B compare_local을 한 번 호출한다. A가 모델별 요청을 따로 보내거나 이력/특징을 직접 계산하지 않는다.
- T-G/R-G/T-P/R-P의 순위·상품·비교 시점·기준 버전과 unavailable 이유를 보여준다. 없는 P를 G로 채우지 않는다. raw score를 확률이나 모델 간 우열로 표시하지 않는다.
- 일반 서비스 모델 선택은 B predict_local의 로컬 인자를 사용한다. 기존 strict JSON 요청에 variant/mode를 임의 추가하지 않는다.
- 통제 이벤트는 한 번만 주문/원장에 반영한다. 자유 사용 결과와 동일 snapshot 비교를 구분하고 비교 내용이 중앙에 남지 않는지 g2로 확인한다.

## 이후 협업
B가 막히면 입력 품질·합성 fixture, C가 막히면 e2e 시나리오를 인계받을 수 있다. 경로 소유권을 [작업 규칙](../working-agreement.md)에서 먼저 바꾼다. 새 결제 기능·중앙 통합 주문·완료 후 취소는 첫 작업에 추가하지 않는다.


공통 시작점: [팀 시작 안내](../start.md). 최초에 담당 카드와 해당 계약을 읽고, 이어갈 때는 관련 diff·검사·남은 작업만 확인한다.
