# 판매자 FL client · C

진입점: `lifecycle.create_client(runtime, jobs, config)`.
A가 제공한 B runtime과 동일한 jobs 객체를 사용한다. `await start()` / `await stop()`은 앱 lifespan에서 호출한다.
현재 기본 enabled=false로 네트워크/학습을 하지 않는다. enabled=true는 두 mode 모두 FeatureNotImplemented로 시작을 거부한다.
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. B 호출은 주입된 jobs에서 실행하고, 실데이터 유래 평문 제출을 허용하지 않는다. 보호 프로토콜 선정과 메시지 계약은 별도 작업이다.
