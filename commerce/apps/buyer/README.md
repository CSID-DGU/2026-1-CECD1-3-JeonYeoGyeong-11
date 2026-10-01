# 구매자 화면 · A

판매자 서버가 렌더하는 구매자용 홈/상품상세/주문/챗봇 Jinja2 템플릿이다. 색상·카드·탭 UI는 팀의 기존 React 프로토타입(JYGfile)과 같은 시각 언어로 맞췄고, `../static/style.css`를 공유한다.

- `templates/layout.html`: 앱 전환 탭(고객용/판매자용) + 홈·주문내역·챗봇 탭 네비게이션. 장바구니에 여러 상품을 담는 기능은 아직 계약(schema)에 없어 "주문내역"(주문 히스토리)으로 대체했고, 상품 상세에서 바로 1건 주문하는 흐름으로 단순화했다.
- `templates/home.html`: 카탈로그 상품 그리드 + 상단 "추천 상품" 슬롯. 추천은 `orders_service.get_recommendations_for_display`가 만들며, B 미구현 상태에서는 P-TopFreq(인기순) 기준선으로 대체되고 화면에 "임시" 배지가 붙는다(`recommendation_is_mock`). B가 실제로 붙으면 이 화면은 고칠 필요 없이 실제 추천으로 자동 전환된다.
- `templates/product.html`: 상품 상세 + 바로 주문 폼(`POST /buyer/{seller_id}/orders`).
- `templates/order_confirmation.html`: 주문 접수 확인 + 처리 흐름(주문접수→매장 로컬 기록→매장 로컬 학습→공유 모델 반영, 뒤 2단계는 B/C 연동 전이라 안내 문구만).
- `templates/orders.html`: `customer_id_local` 기준 주문 내역 조회.
- `templates/chat.html`: 정적 예시 대화(백엔드 미연동).

라우트는 `commerce/services/merchant_api/main.py`의 `/buyer/{seller_id}/...` 경로들이다. **호출자 인증이 없다** — main.py의 모듈 docstring과 OQ13/OQ15 참고. 실제 구매자 브라우저에 연결하기 전에 그 결정이 먼저 나야 한다.

시작: `commerce/services/merchant_api/main.py` → 공통 [개발 안내](../../../docs/development.md).
작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: 화면은 판매자 origin에서 렌더하고, 추천은 B runtime API로 호출하며 특징·모델 DB를 직접 읽지 않는다.

B 구현 전 성공 경로가 필요하면 자기 test double을 주입한다. `UnimplementedRuntime`은 모든 메서드가 예외를 내므로 쓸 수 없다: [개발 안내 §상대 모듈을 대체하는 방법](../../../docs/development.md#상대-모듈을-대체하는-방법).
