# 구매자 화면 · A

판매자 서버가 렌더하는 구매자용 홈/상품상세/주문/챗봇 Jinja2 템플릿이다. 디자인은 플랫폼 이름(임시 "오이OO", `main.py`의 `BRAND_NAME` 한 곳에서 바꾼다)에 맞춘 은은한 오이 초록 계열이다. 색·모서리·간격은 `../static/style.css` 맨 위의 토큰(CSS 변수)으로만 정하고 템플릿에는 inline style을 쓰지 않는다(진행률 막대 폭만 예외). 글꼴은 Pretendard(OFL)를 jsDelivr CDN에서 불러오고, 실패하면 시스템 글꼴로 대체된다. 로고 `../static/logo.svg`와 아이콘은 직접 그린 SVG이고, 상품 사진이 없어 썸네일은 카테고리별 이모지(`main.py`의 `_product_emoji`) 또는 상품명 첫 글자로 보인다. 구매자 화면은 반응형으로, PC에서는 상단 메뉴와 4열 카드, 폰(720px 이하)에서는 왼쪽 썸네일 목록·하단 탭 바·상세 화면 하단 고정 주문 바로 바뀐다(마켓 앱에서 흔한 배치이며 다른 서비스의 로고·색·아이콘은 쓰지 않았다).

- `templates/layout.html`: 데모용 구매자/판매자 전환 띠 + 상단 바(PC 메뉴) + 하단 탭 바(폰). 로그인하면 상단에 장바구니 링크가 보인다.
- `templates/home.html`: 카탈로그 상품 그리드 + 상단 "추천 상품" 슬롯. 추천은 `orders_service.get_recommendations_for_display`가 만들며, B가 답하지 못하면 P-TopFreq(인기순) 기준선으로 대체되고, B가 자기 fallback(공유 모델 없음·판매 이력 부족·첫 방문 고객)을 돌려주면 그 사유가 배지로 붙는다(`recommendation_label`). 실제 모델 순위일 때만 배지가 없다.
- `templates/product.html`: 상품 상세 + 수량과 "장바구니 담기"(`POST /buyer/{seller_id}/cart`)·"바로 주문하기"(`POST /buyer/{seller_id}/orders`) — 상품 ID와 수량만 보내고 가격은 서버가 카탈로그에서 정한다 + 시세 그래프(M30, `price_history`가 2건 이상일 때만 표시).
- `templates/cart.html`: 장바구니. 수량 변경(0이면 삭제), 판매 중지된 상품은 따로 표시하고 주문에서 뺀다. "주문하기"(`POST /buyer/{seller_id}/checkout`)는 담긴 상품 전부를 주문 1건(여러 상품)으로 만든다. 화면을 그릴 때마다 새 `checkout_key`를 넣어서 두 번 눌러도 주문은 한 번만 생긴다.
- `templates/order_confirmation.html`: 주문 접수 확인 + 처리 흐름(주문접수→매장 로컬 기록→매장 로컬 학습→공유 모델 반영, 뒤 2단계는 B/C 연동 전이라 안내 문구만).
- `templates/orders.html`: `customer_id_local` 기준 주문 내역 조회.
- `templates/chat.html`: 정적 예시 대화(백엔드 미연동).

**Layer 6 소셜/부가기능** (모듈지도 M25/M26·M27/M29; cross-role 계약 없음, `social_service.py` 참고):
- `templates/messages.html` (M25 DM): 이 판매자와의 1:1 쪽지 — 거래문의 챗봇(`chat.html`, mock)과는 별개의 실제 저장 기능.
- `templates/feed.html` (M26/M27): 매장 소식·숏폼 소개 읽기 전용. 랭킹은 아직 최신순 로컬 휴리스틱뿐 — `ports.py`에 피드 랭킹용 B 연결점이 없어서 상품 추천(M16)과 달리 "B 연결 대기" 자리가 없다.
- `templates/group_buys.html` (M29): 진행중 공동구매 목록 + 참여 폼. 목표 수량 도달 즉시 성사(마감 전이라도)되어 바로 일반 주문으로 전환된다.

라우트는 `commerce/services/merchant_api/main.py`의 `/buyer/{seller_id}/...` 경로들이다. 둘러보기는 로그인 없이 되고, 주문·쪽지·공동구매 참여·주문내역은 고객 로그인을 요구한다. 로그인한 상태의 POST 폼은 `csrf_token` hidden 필드를 함께 보낸다. 남은 결정은 [merchant_api README](../../services/merchant_api/README.md)의 OQ13 절 참고.

시작: `commerce/services/merchant_api/main.py` → 공통 [개발 안내](../../../docs/development.md).
작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: 화면은 판매자 origin에서 렌더하고, 추천은 B runtime API로 호출하며 특징·모델 DB를 직접 읽지 않는다.

B 구현 전 성공 경로가 필요하면 자기 test double을 주입한다. `UnimplementedRuntime`은 모든 메서드가 예외를 내므로 쓸 수 없다: [개발 안내 §상대 모듈을 대체하는 방법](../../../docs/development.md#상대-모듈을-대체하는-방법).
