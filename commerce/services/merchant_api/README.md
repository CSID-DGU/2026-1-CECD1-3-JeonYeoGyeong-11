# 판매자 서비스 · A

- `main.py`: FastAPI lifespan·health. 현재 업무 API는 없다.
- `context.py`: 판매자별 B runtime 하나를 만들고 같은 객체·작업 실행기를 C client에 주입한다.
- `jobs.py`: FL/개인화가 함께 쓰는 단일 background 실행기. 중복 제출은 JobBusyError, 완료/실패 후 다시 제출할 수 있다.

작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: event/outbox는 B ingest가 정상 반환한 뒤에만 전달 완료로 표시한다. 현재 B stub은 항상 미구현 예외를 내므로 delivered로 처리하면 안 된다.
A는 주문 DB, B는 특징 DB·모델만 수정한다. 웹 worker는 판매자당 1개이며 학습은 `context.jobs.submit(...)`으로 실행한다. 호출 예제는 [개발 안내](../../../docs/development.md)를 따른다.
