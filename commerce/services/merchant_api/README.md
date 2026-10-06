# 판매자 서비스 · A

- `main.py`: FastAPI lifespan·health·주문/카탈로그 JSON 라우트(`/sellers/...`) + 구매자·판매자 화면 HTML 라우트(`/buyer/{seller_id}/...`, `/seller/{seller_id}/...`, `commerce/apps/buyer|seller/templates` 렌더). 호출자 인증은 없다(OQ13·OQ15 대기, 아래 참고).
- `context.py`: 판매자별 B runtime 하나를 만들고 같은 객체·작업 실행기를 C client에 주입한다. `merchant_db_path`(orders.sqlite 위치)도 여기서 들고 있다.
- `jobs.py`: FL/개인화가 함께 쓰는 단일 background 실행기. 중복 제출은 JobBusyError, 완료/실패 후 다시 제출할 수 있다.
- `orders_db.py`: orders.sqlite 스키마·행 단위 접근. 이 파일만 SQL을 실행한다.
- `orders_service.py`: 주문 상태 전이(requested→accepted→completed, ·→cancelled), idempotency, purchase_event/outbox를 한 트랜잭션으로 커밋한 뒤 B ingest 호출, catalog_item upsert(source_seq 증가). `get_recommendations_for_display`는 구매자 홈 화면의 추천 슬롯이다 — `runtime.predict_local`을 먼저 시도하고 `FeatureNotImplemented`면 `commerce/evaluation/metrics/ranking.py`의 P-TopFreq(비모델 기준선)로 대체한다. B가 실제로 구현되면 `predict_local`이 예외를 그만 내는 순간 자동으로 실제 추천으로 바뀌고, 호출부(main.py)는 고칠 필요가 없다. `model_version`이 `MOCK_MODEL_VERSION`("mock-p-topfreq-v1")이면 화면에 "임시" 배지를 띄운다(`recommendation_is_mock`).
- `social_db.py`/`social_service.py`: Layer 6 소셜/부가기능(모듈지도 M25 DM, M26·M27 피드/숏폼, M29 공동구매, M30 시세) — 전부 orders.sqlite의 로컬 테이블만 쓰고 cross-role 계약은 없다. `join_group_buy`는 목표 수량 도달 즉시 성사시켜 참여자별 주문을 만들고(M29→M6), 마감 지난 미달 건만 `settle_due_group_buys`가 지연 평가로 `failed` 처리한다. 정산(M10)·배송(M7)·리뷰(M9)는 아직 범위 밖이라 성사해도 대금 정산 레코드는 없다.
- `accounts_db.py`/`accounts_service.py`: 구매자 회원가입/로그인과 이 매장 자체의 판매자(직원) 계정. `signup_seller`는 `nts_client.verify_business_registration`으로 **국세청 사업자등록 진위확인**을 통과해야 계정을 만든다. 비밀번호는 PBKDF2-SHA256(20만 회)로 저장, 평문 보관 없음.
- `nts_client.py`: 공공데이터포털 "사업자등록정보 진위확인" API 클라이언트. `NTS_SERVICE_KEY`가 설정되면 실제 API(`POST api.odcloud.kr/.../validate`)를 호출하고, 없으면 사업자등록번호 형식만 확인하는 데모 mock으로 대체한다(`VerificationResult.mode`가 "real"/"mock"을 구분). 실제 키로 라이브 호출을 검증한 적은 없다(이 환경엔 키가 없음) — `get_recommendations_for_display`와 같은 "mock 지금, 실제 키로 전환" 구조.
- `session.py`: stdlib(`hmac`/`hashlib`)만으로 서명한 세션 쿠키. 프로세스마다 랜덤 시크릿이라 재시작하면 모든 세션이 끊긴다(의도적 한계, 아래 참고).
- `selfcheck.py`: a1 게이트. 호출자 인증(OQ13·15)과 B 실제 연결(g2)은 의도적으로 남겨두고 그 외 상태전이·타 판매자 거부·평가지표 손계산 예제를 확인한다.
- `tests/fakes.py`: A 소유 테스트 더블(`FakeRecommenderRuntime`). B의 `UnimplementedRuntime`을 고치지 않고 자체 성공 경로를 만든다([개발 안내](../../../docs/development.md#상대-모듈을-대체하는-방법)).
- `tests/test_social.py`: DM 스레드 순서, 피드 최신순, 공동구매 즉시/지연 성사·실패·중복참여 거부, 시세 upsert 커버.
- `tests/test_accounts.py`: 회원가입/로그인, 비밀번호 비평문 저장, NTS mock/real 분기, 세션 서명·변조·만료 커버.
- `seed_demo_data.py`: 데모용 시드 스크립트 — 판매자 계정 1개, 상품 6종, 고객 8명, 지난 30일에 걸쳐 분산된 주문 20여 건, DM 5건, 피드 4건, 공동구매 2건(진행중 1·성사 1), 상품 3종의 14일치 시세를 채운다. 실행: `MERCHANT_DB_PATH=... python -m commerce.services.merchant_api.seed_demo_data`. 재실행해도 중복 생성하지 않는다(계정·주문 idempotency_key로 존재 확인). `tests/test_seed_demo_data.py`가 멱등성·중복 키 없음을 검사한다.

작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: event/outbox는 B ingest가 정상 반환한 뒤에만 전달 완료로 표시한다. 현재 B stub은 항상 미구현 예외를 내므로 delivered로 처리하면 안 된다.
A는 주문 DB, B는 특징 DB·모델만 수정한다. 웹 worker는 판매자당 1개이며 학습은 `context.jobs.submit(...)`으로 실행한다. 호출 예제는 [개발 안내](../../../docs/development.md)를 따른다.

**OQ13의 일부는 이제 해결됨, 전부는 아님.** `/seller/...` 전체(가입/로그인 제외)는 이제 `seller_accounts` 로그인(국세청 인증 완료)을 요구하고, 주문·공동구매참여·쪽지 같은 구매자 행동은 로그인을 요구한다 — "누구나 임의의 seller_id로 행동할 수 있던" 원래 문제는 닫혔다. 아직 없는 것: ① 폼마다 CSRF 토큰(지금은 `SameSite=Lax` 쿠키만, 인터페이스 계약 §6이 요구하는 "호스트별 쿠키와 CSRF"의 뒷부분) ② 세션이 서버 재시작을 못 버팀(프로세스별 랜덤 시크릿) ③ OQ15(주문 URL에 order_id 노출)는 여전히 미결정. 실제 서비스에 연결하기 전엔 이 세 가지를 OQ13 결정과 함께 다시 본다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: `MERCHANT_ID`, `FEATURE_DB_PATH`, `MODEL_DIR`, `MERCHANT_DB_PATH`, `FL_ENABLED`, `FL_MODE`, `FL_MODEL_VARIANT`, `NTS_SERVICE_KEY`
- 구현 시 추가: `MERCHANT_SECRET`, `COORDINATOR_URL`, `FL_CLIENT_TOKEN`

`NTS_SERVICE_KEY`는 선택값이다. 비워두면 판매자 가입이 사업자등록번호 형식만 확인하는 mock 인증으로 통과한다. 실제 국세청 진위확인을 받으려면 [공공데이터포털](https://www.data.go.kr)에서 "사업자등록정보 진위확인 및 상태조회 서비스"를 신청해 발급받은 serviceKey를 이 값에 넣는다.

`MERCHANT_DB_PATH`는 아직 `run_local.py`(C 소유)가 넘기지 않으므로 `os.environ.get`과 `FEATURE_DB_PATH` 옆 기본 경로로 읽는다. 필수값으로 바꾸는 것은 C의 런처 PR이 먼저 병합된 뒤다([작업 규칙](../../../docs/team/working-agreement.md) §3).
