# 로컬 추천 runtime · B

시작: `runtime.py`의 `open_runtime`. 공통 명세는 `commerce.packages.contracts.ports.RecommenderRuntime`이다.
현재 모든 업무 메서드는 FeatureNotImplemented를 낸다. 데이터·모델 파일을 만들지 않고 저장/학습 성공을 반환하지 않는다.
실제 구현은 별도 클래스로 만들고 `open_runtime`이 그것을 돌려주게 한다. `UnimplementedRuntime`은 scaffold가 검사하는 기준 stub이므로 고치거나 지우지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
작업 순서는 [B 카드](../../../docs/team/tasks/B.md)를 따른다. 실제 checkpoint를 고르기 전 fixture의 벡터/shape를 실제 값으로 간주하지 않는다.
A/C는 같은 runtime을 사용한다. API 시그니처/반환 객체 변경은 공통 ports/types와 호출자·검사를 함께 수정한다. 실험 core는 웹 앱을 import하지 않는다.
