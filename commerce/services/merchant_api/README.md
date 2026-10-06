# 판매자 서비스 · A

- `main.py`: FastAPI lifespan·health·주문/카탈로그 JSON 라우트(`/sellers/...`) + 구매자·판매자 화면 HTML 라우트(`/buyer/{seller_id}/...`, `/seller/{seller_id}/...`, `commerce/apps/buyer|seller/templates` 렌더). 화면은 로그인과 폼별 CSRF 토큰을 확인하고, 주문 가격은 폼이 아니라 서버의 카탈로그 가격으로 정한다. JSON `/sellers/...` 라우트는 아직 호출자 인증이 없다(OQ13·OQ15 대기, 아래 참고).
- `context.py`: 판매자별 B runtime 하나를 만들고 같은 객체·작업 실행기를 C client에 주입한다. `merchant_db_path`(orders.sqlite 위치)도 여기서 들고 있다.
- `jobs.py`: FL/개인화가 함께 쓰는 단일 background 실행기. 중복 제출은 JobBusyError, 완료/실패 후 다시 제출할 수 있다.
- `orders_db.py`: orders.sqlite 스키마·행 단위 접근. 이 파일만 SQL을 실행한다.
- `orders_service.py`: 주문 상태 전이(requested→accepted→completed, ·→cancelled), idempotency, purchase_event/outbox를 한 트랜잭션으로 커밋한 뒤 B ingest 호출, catalog_item upsert(source_seq 증가). `get_recommendations_for_display`는 구매자 홈 화면의 추천 슬롯이다 — 밀린 카탈로그 전달을 먼저 재시도하고, 후보는 B의 활성 카탈로그에 맡긴 채(`candidate_item_ids=None`, B에 아직 안 간 상품이 NOT_FOUND로 슬롯 전체를 막지 않게) `runtime.predict_local`을 부른다. B가 답하지 못하면(stub의 `FeatureNotImplemented`나 그 밖의 실패) `commerce/evaluation/metrics/ranking.py`의 P-TopFreq(비모델 기준선, `MOCK_MODEL_VERSION`)로 대체하므로 추천기 장애가 구매자 화면을 막지 않는다. 배지는 `recommendation_label`이 정한다: A 기준선이면 "임시", B 자체 fallback(`fallback_reason`: 공유 모델 없음·판매 이력 부족·첫 방문 고객)이면 그 사유를 띄우고, 실제 모델 순위면 배지가 없다. `get_comparison_for_display`는 판매자 "모델 비교" 화면(`/seller/{seller_id}/compare`)용으로 B `compare_local`을 한 번만 부르고, T-G·R-G·T-P·R-P의 순위·기준 버전·비교 시점과 unavailable 사유를 그대로 보여준다. P칸을 G로 채우거나 점수를 모델 간에 비교하지 않는다(comparison.md §5).
- `social_db.py`/`social_service.py`: Layer 6 소셜/부가기능(모듈지도 M25 DM, M26·M27 피드/숏폼, M29 공동구매, M30 시세) — 전부 orders.sqlite의 로컬 테이블만 쓰고 cross-role 계약은 없다. `join_group_buy`는 목표 수량 도달 즉시 성사시켜 참여자별 주문을 만들고(M29→M6), 마감 지난 미달 건만 `settle_due_group_buys`가 지연 평가로 `failed` 처리한다. 정산(M10)·배송(M7)·리뷰(M9)는 아직 범위 밖이라 성사해도 대금 정산 레코드는 없다.
- `cart_db.py` + `orders_service`의 `add_to_cart`·`set_cart_quantity`·`get_cart`·`checkout_cart`: 구매자 장바구니(orders.sqlite의 `cart_items`, 계약 없음). 결제는 담긴 상품을 카탈로그 현재 가격으로 `place_order` 1건(여러 상품)으로 만들고, `checkout_key`를 idempotency_key로 써서 중복 제출에도 주문이 한 번만 생긴다. `tests/test_cart.py`.
- `accounts_db.py`/`accounts_service.py`: 구매자 회원가입/로그인과 이 매장 자체의 판매자(직원) 계정. `signup_seller`는 `nts_client.verify_business_registration`으로 **국세청 사업자등록 진위확인**을 통과해야 계정을 만든다. 비밀번호는 PBKDF2-SHA256(20만 회)로 저장, 평문 보관 없음.
- `nts_client.py`: 공공데이터포털 "사업자등록정보 진위확인" API 클라이언트. `NTS_SERVICE_KEY`가 설정되면 실제 API(`POST api.odcloud.kr/.../validate`)를 호출하고, 없으면 사업자등록번호 형식만 확인하는 데모 mock으로 대체한다(`VerificationResult.mode`가 "real"/"mock"을 구분). 실제 키로 라이브 호출을 검증한 적은 없다(이 환경엔 키가 없음) — `get_recommendations_for_display`와 같은 "mock 지금, 실제 키로 전환" 구조.
- `session.py`: stdlib(`hmac`/`hashlib`)만으로 서명한 세션 쿠키와 세션에 묶인 CSRF 토큰. `MERCHANT_SECRET`이 있으면 그 값에서 키를 만들어 재시작해도 로그인이 유지되고, 없으면 프로세스마다 랜덤 키다(재시작하면 로그아웃).
- `selfcheck.py`: a1 게이트. 호출자 인증(OQ13·15)과 B 실제 연결(g2)은 의도적으로 남겨두고 그 외 상태전이·타 판매자 거부·평가지표 손계산 예제를 확인한다.
- `tests/fakes.py`: A 소유 테스트 더블(`FakeRecommenderRuntime`). B의 `UnimplementedRuntime`을 고치지 않고 자체 성공 경로를 만든다([개발 안내](../../../docs/development.md#상대-모듈을-대체하는-방법)).
- `tests/test_social.py`: DM 스레드 순서, 피드 최신순, 공동구매 즉시/지연 성사·실패·중복참여 거부, 시세 upsert 커버.
- `tests/test_accounts.py`: 회원가입/로그인, 비밀번호 비평문 저장, NTS mock/real 분기, 세션 서명·변조·만료 커버.
- 시작 시 재전달: 앱 lifespan이 `orders_service.deliver_all_pending`으로 밀린 outbox(상품 → 구매 이벤트 순)를 한 번 다시 보낸다(interfaces.md §2 재시작 후 재전달). seed가 runtime 없이 쓴 주문도 이때 B로 간다. B가 실패해도 기동은 막지 않는다. 판매자 개요는 B가 실제 모델로 답하는지·자체 fallback인지·미연결인지와 outbox 전달 현황을 보여준다.
- `tests/test_screens.py`: HTTP로 화면을 검사한다 — CSRF 없는/위조/이전 세션 토큰 거부, 폼 가격 조작 무시, A 기준선·B fallback·실제 모델의 배지, 추천기 오류 시 화면 유지, 비교 화면 네 칸과 P칸 미대체, `MERCHANT_SECRET`의 재시작 유지.
- `seed_demo_data.py`: 데모용 시드 스크립트 — 판매자 계정 1개, 상품 6종, 고객 8명, 지난 30일에 걸쳐 분산된 주문 20여 건, DM 5건, 피드 4건, 공동구매 2건(진행중 1·성사 1), 상품 3종의 14일치 시세를 채운다. 실행: 앱과 같은 환경변수로 `MERCHANT_ID=merchant-1 FEATURE_DB_PATH=commerce/deploy/var/merchant_1/features.sqlite python -m commerce.services.merchant_api.seed_demo_data` (`MERCHANT_DB_PATH`가 있으면 그 경로, 없으면 `FEATURE_DB_PATH` 옆 `orders.sqlite` — 앱과 같은 `context.merchant_db_path_from_env` 규칙. `MERCHANT_ID`가 없으면 만들지 않고 멈춘다). 재실행해도 중복 생성하지 않는다(계정·주문 idempotency_key로 존재 확인). `tests/test_seed_demo_data.py`가 멱등성·중복 키 없음을 검사한다. `--bulk N`을 붙이면 상품 28종과 **취향이 있는** 고객 N명(아침 장보기·수산물·커피·과일채소·고기 집밥 다섯 패턴, 단골 재구매와 약간의 탐색), 60일치 주문을 더 만든다(`seed_bulk`). 기본 seed가 쓰는 내용은 바뀌지 않고 키가 모두 `bulk-`로 시작해서 이미 seed한 DB 위에 다시 돌려도 된다. 완료 주문의 purchase_event 시각도 과거 완료 시각으로 맞춘다(아직 B에 전달되지 않은 행만).
- `simulate_activity.py`: 실행 중인 앱에 대한 실시간 데모 트래픽. seed 고객이 로그인 → 홈의 추천을 보고(기본 60%는 추천 상품) 바로 주문하거나 장바구니로 주문하고, 판매자가 수락·완료해 구매 이벤트가 B로 간다. 브라우저와 같은 폼·CSRF로 HTTP만 쓴다. 실행: `python -m commerce.services.merchant_api.simulate_activity --base http://127.0.0.1:8101 --seller merchant-1 --interval 3` (Ctrl+C로 중지). `tests/test_simulate_activity.py`.

작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: event/outbox는 B ingest가 정상 반환한 뒤에만 전달 완료로 표시한다. 현재 B stub은 항상 미구현 예외를 내므로 delivered로 처리하면 안 된다.
A는 주문 DB, B는 특징 DB·모델만 수정한다. 웹 worker는 판매자당 1개이며 학습은 `context.jobs.submit(...)`으로 실행한다. 호출 예제는 [개발 안내](../../../docs/development.md)를 따른다.

**OQ13의 화면 쪽은 구현됨, 결정은 아직.** `/seller/...` 전체(가입/로그인 제외)는 `seller_accounts` 로그인(국세청 인증 완료)을, 주문·공동구매참여·쪽지 같은 구매자 행동은 고객 로그인을 요구한다. 로그인한 사용자의 모든 POST 폼은 그 세션 쿠키에 묶인 CSRF 토큰을 확인한다(host-only·HttpOnly·SameSite=Lax 쿠키와 함께, 인터페이스 계약 §6의 "호스트별 쿠키와 CSRF"). 남은 것: ① JSON `/sellers/...` 라우트의 호출자 인증과 OQ13의 오류 코드 결정 ② OQ15(주문 URL에 order_id 노출) ③ 가입·로그인 폼은 세션이 없어 SameSite=Lax에만 기댄다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: `MERCHANT_ID`, `FEATURE_DB_PATH`, `MODEL_DIR`, `MERCHANT_DB_PATH`, `FL_ENABLED`, `FL_MODE`, `FL_MODEL_VARIANT`, `NTS_SERVICE_KEY`, `MERCHANT_SECRET`
- 구현 시 추가: `COORDINATOR_URL`, `FL_CLIENT_TOKEN`

`MERCHANT_SECRET`은 선택값이다. 지금은 세션 쿠키·CSRF 서명 키를 만드는 데만 쓰며, 판매자마다 다른 충분히 긴 임의 문자열을 넣는다. 비워두면 재시작할 때마다 모든 사용자가 로그아웃된다. 저장소나 로그에 남기지 않는다.

`NTS_SERVICE_KEY`는 선택값이다. 비워두면 판매자 가입이 사업자등록번호 형식만 확인하는 mock 인증으로 통과한다. 실제 국세청 진위확인을 받으려면 [공공데이터포털](https://www.data.go.kr)에서 "사업자등록정보 진위확인 및 상태조회 서비스"를 신청해 발급받은 serviceKey를 이 값에 넣는다.

`MERCHANT_DB_PATH`는 아직 `run_local.py`(C 소유)가 넘기지 않으므로 `os.environ.get`과 `FEATURE_DB_PATH` 옆 기본 경로로 읽는다. 필수값으로 바꾸는 것은 C의 런처 PR이 먼저 병합된 뒤다([작업 규칙](../../../docs/team/working-agreement.md) §3).
