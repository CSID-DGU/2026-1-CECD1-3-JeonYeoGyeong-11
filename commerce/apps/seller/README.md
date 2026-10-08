# 판매자 화면 · A

판매자용 개요/상품관리/주문관리 Jinja2 템플릿이다. 디자인은 플랫폼 이름(임시 "오이OO", `main.py`의 `BRAND_NAME` 한 곳에서 바꾼다)에 맞춘 은은한 오이 초록 계열이다. 색·모서리·간격은 `../static/style.css` 맨 위의 토큰(CSS 변수)으로만 정하고 템플릿에는 inline style을 쓰지 않는다(진행률 막대 폭만 예외). 글꼴은 Pretendard(OFL)를 jsDelivr CDN에서 불러오고, 실패하면 시스템 글꼴로 대체된다. 로고 `../static/logo.svg`와 아이콘은 직접 그린 SVG이고, 상품 사진이 없어 썸네일은 카테고리별 이모지(`main.py`의 `_product_emoji`) 또는 상품명 첫 글자로 보인다. 판매자 화면은 PC에서 왼쪽 세로 메뉴, 960px 이하에서 가로 스크롤 메뉴로 바뀐다.

- `templates/layout.html`: 앱 전환 탭(고객용/판매자용) + 개요·상품관리·주문관리 탭 네비게이션.
- `templates/overview.html`: `orders.sqlite`에서 직접 센 등록 상품 수·처리 대기/완료 주문 수. ML 예측 지표(재방문 확률 등)는 B 추천 런타임이 아직 미구현(`UnimplementedRuntime`)이라 넣지 않았다.
- `templates/products.html`: 상품 등록 폼 + 목록. `display_price_minor`는 화면 전용 필드로, `catalog_item.v1`에는 없다(계약에 가격 필드가 없어서 로컬 DB 컬럼으로만 둔 것; `orders_service.register_catalog_item` 참고).
- `templates/orders.html`: 주문 목록 + 수락/완료 처리 폼. `status_version`을 hidden input으로 넘겨 조건부 갱신(동시 완료 시 1회만 성공)에 그대로 태운다.
- `templates/compare.html` (A 카드 "모델 비교 화면"): 고객 한 명을 골라 B `compare_local`을 한 번 호출하고 T-G·R-G·T-P·R-P 네 칸의 순위·상품·기준 모델 버전·비교 시점·특징 스냅샷과 unavailable 사유를 보여준다. 점수는 모델 간 척도가 달라 표시하지 않고, 없는 P칸을 G로 채우지 않는다. B가 비교를 못 하는 동안(stub)은 "비교할 수 없음"만 표시한다.

**Layer 6 소셜/부가기능** (모듈지도 M25/M26·M27/M29/M30; `social_db.py`/`social_service.py`가 로직을 담당하며 cross-role 계약은 없음 — orders.sqlite 안의 로컬 테이블만 쓴다):
- `templates/messages.html`, `message_thread.html` (M25 DM): 고객별 쪽지함 + 답장.
- `templates/feed.html` (M26/M27 피드·숏폼): 소식 작성(영상은 소개 글로 대체, 별도 미디어 저장소 미구현).
- `templates/group_buys.html` (M29 공동구매): 캠페인 생성 + 진행률. **목표 수량 도달 즉시 성사**(마감 전이라도) — `social_service.join_group_buy`가 즉시 정산해 참여자별로 일반 주문(M6)을 만든다. 마감 지났는데 미달이면 `failed`. 정산(M10)은 아직 없어서 성사해도 대금 정산 레코드는 안 만든다.
- `templates/prices.html` (M30 시세): 일별 가격 입력 + 최근 7일 테이블. 외부 시세 API(KAMIS 등) 연동 없이 판매자가 직접 입력.

라우트는 `commerce/services/merchant_api/main.py`의 `/seller/{seller_id}/...` 경로들이다(JSON API인 `/sellers/{seller_id}/...`와는 별도). 가입·로그인을 뺀 모든 화면은 판매자 로그인을 요구하고, 모든 POST 폼은 `csrf_token` hidden 필드를 함께 보낸다(새 폼을 만들 때도 넣어야 403이 나지 않는다). 남은 결정은 [merchant_api README](../../services/merchant_api/README.md)의 OQ13 절 참고.

시작: `commerce/services/merchant_api/main.py`와 `context.py`. 작업 순서는 [A 카드](../../../docs/team/tasks/A.md)를 따른다. 경계: 완료 주문의 event/outbox 영속화는 A가 담당하고, 모델 학습 설정은 B/C 인터페이스를 사용한다.

B 구현 전 성공 경로가 필요하면 자기 test double을 주입한다. `UnimplementedRuntime`은 모든 메서드가 예외를 내므로 쓸 수 없다: [개발 안내 §상대 모듈을 대체하는 방법](../../../docs/development.md#상대-모듈을-대체하는-방법).
