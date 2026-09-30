# 로컬 추천 runtime · B

시작: `runtime.py`의 `open_runtime`. 공통 명세는 `commerce.packages.contracts.ports.RecommenderRuntime`이다.
현재 runtime의 모든 업무 메서드는 FeatureNotImplemented를 낸다. 데이터·모델 파일을 만들지 않고 저장/학습 성공을 반환하지 않는다.
모델 core는 runtime과 별도로 있고 아직 runtime에 연결되지 않았다. 학습·평가·서비스가 같은 코드를 쓴다.
- `text_encoder.py`: frozen 텍스트 인코더(평균 pooling·L2·text_artifact_hash). `z_cache.py`: 판매자 로컬 z cache(SQLite, key = artifact·preprocessing·텍스트 해시)
- `examples.py`: "이전 방문 → 다음 방문 상품 집합" 예제. `replay.py`: Instacart 진행률 replay(판매자가 target 시점에 볼 수 있는 구매)
- `relations.py`: 판매자 로컬 상품 관계 snapshot(고객·장바구니·방향별 시간)
- `model.py`: 두 variant(text_only 6개·text_relation 9개 공유 그룹, 상품 ID·bias 없음)와 OQ01 손실. `training.py`: 판매자·snapshot별 배치 학습·고정 검증 손실·전체 후보 점수
- `harex.py`: D0022 비교용 HAREX(GCI)식 공통 뼈대(1층 Transformer)와 네 상품 표현 T_hx·R_hx·T_lm·R_lm. hx 단어 토큰 표는 판매자 로컬이라 공유·집계하지 않는다

manifest·export·개인화는 아직 없다. b2 selfcheck는 작은 무작위 BERT로 CI에서 돈다.
실제 구현은 별도 클래스로 만들고 `open_runtime`이 그것을 돌려주게 한다. `UnimplementedRuntime`은 scaffold가 검사하는 기준 stub이므로 고치거나 지우지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
작업 순서는 [B 카드](../../../docs/team/tasks/B.md)를 따른다. 실제 checkpoint를 고르기 전 fixture의 벡터/shape를 실제 값으로 간주하지 않는다.
A/C는 같은 runtime을 사용한다. API 시그니처/반환 객체 변경은 공통 ports/types와 호출자·검사를 함께 수정한다. 실험 core는 웹 앱을 import하지 않는다.
