# 판매자 화면 · A

판매자용 상품/주문 관리 템플릿·정적 파일 위치다. 현재 화면은 미구현이다.

시작: `commerce/services/merchant_api/main.py`와 `context.py`. 작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: 완료 주문의 event/outbox 영속화는 A가 담당하고, 모델 학습 설정은 B/C 인터페이스를 사용한다.

B 구현 전 성공 경로가 필요하면 자기 test double을 주입한다. `UnimplementedRuntime`은 모든 메서드가 예외를 내므로 쓸 수 없다: [개발 안내 §상대 모듈을 대체하는 방법](../../../docs/development.md#상대-모듈을-대체하는-방법).
