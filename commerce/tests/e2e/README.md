# 통합 검사 · C

현재 `test_scaffold.py`는 객체 연결·작업 중복 거부·실패 처리·미구현 표시만 검증한다. 항상 지킬 경계(`ScaffoldInvariants`)와 아직 없는 C 기능(`NotYetImplemented`)을 나눠 두며, 다른 역할의 구현 상태는 고정하지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
실행: `python -m commerce.tools.gate scaffold`.
실제 주문/추천 연결·FL 수치·보호 검증은 향후 g2/g3/g4에서 확인한다. scaffold 성공으로 업무 게이트를 통과 처리하지 않는다.
