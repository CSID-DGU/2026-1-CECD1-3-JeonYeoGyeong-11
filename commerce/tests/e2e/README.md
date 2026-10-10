# 통합 검사 · C

현재 `test_scaffold.py`는 객체 연결·작업 중복 거부·실패 처리·미구현 표시만 검증한다. 항상 지킬 경계(`ScaffoldInvariants`)와 아직 없는 C 기능(`NotYetImplemented`)을 나눠 두며, 다른 역할의 구현 상태는 고정하지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
실행: `python -m commerce.tools.gate scaffold`.

`test_c1_*.py`와 `dummy_round.py`는 c1용이다. 생성 tensor만 쓰는 가짜 B runtime(`DummyRuntime`)과 in-process HTTP로 검사한다. 실제 B runtime과 A의 `build_context`는 거치지 않는다. 실행: `python -m commerce.tools.gate c1`(남은 항목이 있어 종료 코드 3).

- `test_c1_round.py`: 집계 core, `aggregate_uniform`, registry 디스크 저장, 초기 release import, npz 규칙.
- `test_c1_service.py`: cohort 고정, 라운드 원장, 재시작 폐기.
- `test_c1_auth.py`: 판매자 토큰.
- `test_c1_http.py`: coordinator 경로, 인증, 오류 형식, 전송 한도.
- `test_c1_client.py`: 판매자 client 함수로 HTTP 라운드를 끝까지 돌리고 신규 판매자가 설치하는 흐름, 거부 사례.
- `test_c1_synthetic.py`: synthetic_plaintext 루프(설치 후 참여, 라운드당 한 번)와 합성 입력 확인(D0025). 확인 파일과 다른 내용·생성기 밖 이벤트 한 건·해시를 못 내는 runtime은 제출하지 않는다.

`test_g3_flow.py`·`selfcheck_g3.py`는 g3용이다. B의 실제 `SellerRuntime`(CI 크기 `TINY_ARCHITECTURES`·`FakeText`)과 B의 g3 시나리오로 다음을 한 번에 돈다.
1. 판매자 4곳이 base-0을 설치한다.
2. 한 판매자가 개인화한다.
3. 고정 cohort 3곳이 한 라운드를 돌아 새 base가 나온다.
4. 모두 새 base를 설치하고, 옛 개인화는 붙지 않으며 새 base에서 다시 만든다.
5. cohort 밖·상품 비중복의 네 번째 판매자가 새 base로 점수화한다.
6. 통제 구매로 신상품의 점수가 바뀐다.

coordinator는 같은 프로세스 HTTP다. 실행: `python -m commerce.tools.gate g3`(종료 코드 3). 별도 프로세스로 같은 흐름을 도는 것은 `commerce.deploy.fl_demo`이며 CI에서는 돌리지 않는다.

실제 주문/추천 연결(g2)과 보호 검증(g4)은 향후 확인한다. scaffold나 c1 성공으로 그 게이트를 통과 처리하지 않는다.
