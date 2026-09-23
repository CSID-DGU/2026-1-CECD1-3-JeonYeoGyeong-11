# 판매자 화면 · A

판매자용 상품/주문 관리 템플릿·정적 파일 위치다. 현재 화면은 미구현이다.

시작: `commerce/services/merchant_api/main.py`와 `context.py`. 다음 작업은 상품 생성과 주문 상태 전이이며, 완료 주문의 event/outbox 영속화는 A가 담당한다. 모델 학습 설정은 B/C 인터페이스를 사용한다.
