# 로컬 추천 runtime · B

시작: `runtime.py`의 `open_runtime`. 공통 명세는 `commerce.packages.contracts.ports.RecommenderRuntime`이다.
현재 모든 업무 메서드는 FeatureNotImplemented를 낸다. `text_encoder.py`는 frozen 텍스트 인코더(평균 pooling·L2·text_artifact_hash)이고, `z_cache.py`는 판매자 로컬 z cache(SQLite, key = artifact·preprocessing·텍스트 해시)다. runtime 연결은 아직 없다. b2 selfcheck는 작은 무작위 BERT로 CI에서 돈다.
`examples.py`는 "이전 방문 → 다음 방문 상품 집합" 예제를, `replay.py`는 Instacart 진행률 replay(판매자가 target 시점에 볼 수 있는 구매)를 만든다. 학습·평가·서비스가 같은 코드를 쓴다.
`model.py`는 text_only 모델(6개 공유 그룹, 상품 ID·bias 없음)과 OQ01 손실, `training.py`는 판매자별 배치 학습·고정 검증 손실·전체 후보 점수다. text_relation과 manifest·export는 아직 없다. 데이터·모델 파일을 만들지 않고 저장/학습 성공을 반환하지 않는다.
실제 구현은 별도 클래스로 만들고 `open_runtime`이 그것을 돌려주게 한다. `UnimplementedRuntime`은 scaffold가 검사하는 기준 stub이므로 고치거나 지우지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
작업 순서는 [B 카드](../../../docs/team/tasks/B.md)를 따른다. 실제 checkpoint를 고르기 전 fixture의 벡터/shape를 실제 값으로 간주하지 않는다.
A/C는 같은 runtime을 사용한다. API 시그니처/반환 객체 변경은 공통 ports/types와 호출자·검사를 함께 수정한다. 실험 core는 웹 앱을 import하지 않는다.
