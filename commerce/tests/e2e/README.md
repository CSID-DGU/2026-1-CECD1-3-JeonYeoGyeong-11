# 통합 검사 · C

현재 `test_scaffold.py`는 객체 연결·작업 중복 거부·실패 처리·미구현 표시만 검증한다. 항상 지킬 경계(`ScaffoldInvariants`)와 아직 없는 C 기능(`NotYetImplemented`)을 나눠 두며, 다른 역할의 구현 상태는 고정하지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
실행: `python -m commerce.tools.gate scaffold`.

`test_c1_*.py`와 `dummy_round.py`는 c1용이다. 생성 tensor만 쓰는 가짜 B runtime(`DummyRuntime`)과 in-process HTTP로 검사한다. 실제 B runtime과 A의 `build_context`는 거치지 않는다. 실행: `python -m commerce.tools.gate c1`(남은 항목이 있어 종료 코드 3).

- `test_c1_round.py`: 집계 core, `aggregate_uniform`, registry 디스크 저장, 초기 release import, npz 규칙.
- `test_c1_service.py`: cohort 고정, 라운드 원장, 재시작 폐기.
- `test_c1_auth.py`: 판매자 토큰.
- `test_c1_http.py`: coordinator 경로, 인증, 오류 형식, 전송 한도.
- `test_c1_client.py`: 판매자 client 함수로 HTTP 라운드를 끝까지 돌리고 신규 판매자가 설치하는 흐름, 거부 사례.

실제 주문/추천 연결·FL 수치·보호 검증은 향후 g2/g3/g4에서 확인한다. scaffold나 c1 성공으로 그 게이트를 통과 처리하지 않는다.
