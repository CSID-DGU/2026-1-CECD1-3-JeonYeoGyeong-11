# 판매자 서비스 · A

- `main.py`: FastAPI lifespan·health. 현재 업무 API는 없다.
- `context.py`: 판매자별 B runtime 하나를 만들고 같은 객체·작업 실행기를 C client에 주입한다.
- `jobs.py`: FL/개인화가 함께 쓰는 단일 background 실행기. 중복 제출은 JobBusyError, 완료/실패 후 다시 제출할 수 있다.

작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: event/outbox는 B ingest가 정상 반환한 뒤에만 전달 완료로 표시한다. 현재 B stub은 항상 미구현 예외를 내므로 delivered로 처리하면 안 된다.
A는 주문 DB, B는 특징 DB·모델만 수정한다. 웹 worker는 판매자당 1개이며 학습은 `context.jobs.submit(...)`으로 실행한다. 호출 예제는 [개발 안내](../../../docs/development.md)를 따른다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: `MERCHANT_ID`, `FEATURE_DB_PATH`, `MODEL_DIR`, `FL_ENABLED`, `FL_MODE`, `FL_MODEL_VARIANT`
- 구현 시 추가: `MERCHANT_DB_PATH`, `MERCHANT_SECRET`, `COORDINATOR_URL`, `FL_CLIENT_TOKEN`

필수 값은 C의 `commerce/deploy/run_local.py`가 그 값을 넘기는 PR이 먼저 병합된 뒤에 읽는다([작업 규칙](../../../docs/team/working-agreement.md) §3).
