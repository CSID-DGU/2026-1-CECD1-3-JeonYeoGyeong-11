# 판매자 서비스 · A

- `main.py`: FastAPI lifespan·health·주문/카탈로그 JSON 라우트(`/sellers/...`) + 구매자·판매자 화면 HTML 라우트(`/buyer/{seller_id}/...`, `/seller/{seller_id}/...`, `commerce/apps/buyer|seller/templates` 렌더). 호출자 인증은 없다(OQ13·OQ15 대기, 아래 참고).
- `context.py`: 판매자별 B runtime 하나를 만들고 같은 객체·작업 실행기를 C client에 주입한다. `merchant_db_path`(orders.sqlite 위치)도 여기서 들고 있다.
- `jobs.py`: FL/개인화가 함께 쓰는 단일 background 실행기. 중복 제출은 JobBusyError, 완료/실패 후 다시 제출할 수 있다.
- `orders_db.py`: orders.sqlite 스키마·행 단위 접근. 이 파일만 SQL을 실행한다.
- `orders_service.py`: 주문 상태 전이(requested→accepted→completed, ·→cancelled), idempotency, purchase_event/outbox를 한 트랜잭션으로 커밋한 뒤 B ingest 호출, catalog_item upsert(source_seq 증가). `get_recommendations_for_display`는 구매자 홈 화면의 추천 슬롯이다 — `runtime.predict_local`을 먼저 시도하고 `FeatureNotImplemented`면 `commerce/evaluation/metrics/ranking.py`의 P-TopFreq(비모델 기준선)로 대체한다. B가 실제로 구현되면 `predict_local`이 예외를 그만 내는 순간 자동으로 실제 추천으로 바뀌고, 호출부(main.py)는 고칠 필요가 없다. `model_version`이 `MOCK_MODEL_VERSION`("mock-p-topfreq-v1")이면 화면에 "임시" 배지를 띄운다(`recommendation_is_mock`).
- `selfcheck.py`: a1 게이트. 호출자 인증(OQ13·15)과 B 실제 연결(g2)은 의도적으로 남겨두고 그 외 상태전이·타 판매자 거부·평가지표 손계산 예제를 확인한다.
- `tests/fakes.py`: A 소유 테스트 더블(`FakeRecommenderRuntime`). B의 `UnimplementedRuntime`을 고치지 않고 자체 성공 경로를 만든다([개발 안내](../../../docs/development.md#상대-모듈을-대체하는-방법)).

작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: event/outbox는 B ingest가 정상 반환한 뒤에만 전달 완료로 표시한다. 현재 B stub은 항상 미구현 예외를 내므로 delivered로 처리하면 안 된다.
A는 주문 DB, B는 특징 DB·모델만 수정한다. 웹 worker는 판매자당 1개이며 학습은 `context.jobs.submit(...)`으로 실행한다. 호출 예제는 [개발 안내](../../../docs/development.md)를 따른다.

**의도적으로 아직 없는 것.** `main.py`의 라우트는 URL의 `seller_id`를 그대로 믿는다 — 어떤 호출자가 어떤 판매자로 행동할 수 있는지 확인하는 절차가 없다. 이건 누락이 아니라 OQ13(인증 실패 오류 코드)·OQ15(주문 화면 URL에 order_id를 넣어도 되는지)가 아직 사람 결정 대기 상태([열린 구현 항목](../../../docs/design/open-questions.md))라 A 카드 지시대로 화면·인증보다 도메인 계층(상태전이·transaction·outbox)을 먼저 만든 것이다. 그 결정이 나기 전에는 이 라우트를 실제 구매자/판매자 브라우저에 연결하지 않는다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: `MERCHANT_ID`, `FEATURE_DB_PATH`, `MODEL_DIR`, `MERCHANT_DB_PATH`, `FL_ENABLED`, `FL_MODE`, `FL_MODEL_VARIANT`
- 구현 시 추가: `MERCHANT_SECRET`, `COORDINATOR_URL`, `FL_CLIENT_TOKEN`

`MERCHANT_DB_PATH`는 아직 `run_local.py`(C 소유)가 넘기지 않으므로 `os.environ.get`과 `FEATURE_DB_PATH` 옆 기본 경로로 읽는다. 필수값으로 바꾸는 것은 C의 런처 PR이 먼저 병합된 뒤다([작업 규칙](../../../docs/team/working-agreement.md) §3).
