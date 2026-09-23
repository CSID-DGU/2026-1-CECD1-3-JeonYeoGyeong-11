# 구매자 화면 · A

판매자 서버가 렌더할 구매자용 템플릿·정적 파일 위치다. 현재 화면은 미구현이다.

시작: `commerce/services/merchant_api/main.py` → 공통 [개발 안내](../../../docs/development.md).
다음 작업: 합성 상품/주문 화면을 판매자 origin에서 연결한다. 추천은 B runtime API로 호출하고 특징·모델 DB를 직접 읽지 않는다.

B 구현 전 성공 경로가 필요하면 자기 test double을 주입한다. `UnimplementedRuntime`은 모든 메서드가 예외를 내므로 쓸 수 없다: [개발 안내 §상대 모듈을 대체하는 방법](../../../docs/development.md#상대-모듈을-대체하는-방법).
