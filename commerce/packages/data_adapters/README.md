# 데이터 어댑터 · B

Instacart·Dunnhumby 어댑터, Instacart 판매자 배정, live 입력 변환, 상품 텍스트 builder가 있다. b1 검사는 `python -m commerce.tools.gate b1`이다.

- `instacart.py`: prior 주문 → purchase_event.v1, 판매자별 관측 상품 → catalog_item.v1. 배정(user_id → client_id)은 입력이다. 수량은 null, 시간은 고객별 누적 상대일이다.
- `dunnhumby.py`: completejourney R 파일 → purchase_event.v1·catalog_item.v1. [데이터](../../../docs/design/data.md) §2 순서로 정제하고, 가구를 2~39주 basket이 가장 많은 점포 하나에 두며(동점은 작은 store_id) 그 점포의 거래만 남긴다(OQ06). 300 basket 이상 점포가 roster다. 시각은 파일의 tzone(America/New_York)으로 읽고 월요일 시작 주로 나눈다. 상품명이 없으므로 title은 product_type(없으면 product_category)인 대체 제목, description은 package_size, path는 [department, product_category]다.
- `rds.py`: R의 직렬화 형식(.rds·.rda) 중 data frame에 필요한 부분을 R 없이 읽는다. lock 밖 패키지를 쓰지 않는다(OQ12).
- `assignment.py`: 가상 판매자 배정. 이전 Dirichlet 라벨 분할과 같은 절차지만 표본 선정·라벨·크기에 train 주문(floor(0.7n)까지)만 쓴다. 목표 크기(client_id → 크기)는 Dunnhumby 점포 roster(`load_dunnhumby(...).roster`, 점포 → 2~39주 basket 수)에서 온다. 지금까지의 실행은 대리값(`e_g0.STAND_IN_TARGETS`)을 썼다. 배정표는 고객별 파생물이라 Git 제외 경로에 둔다.
- `baskets.py`: 모든 출처의 purchase_event.v1 → 공통 `LocalBasket`. 출처별 partition·시간 종류·order_rank·수량 규칙([인터페이스](../../../docs/design/interfaces.md) §2)을 검사한다. `customer_visits`는 고객 한 명의 방문 순서·간격·검열 비트(30일 cap)를 만든다.
- `text.py`: [데이터](../../../docs/design/data.md) §3의 builder. catalog_item은 출처 규칙으로 읽어 원자료 경로와 같은 텍스트를 낸다(OQ03의 실험실 쪽).
- `synthetic.py`: 원자료 값을 쓰지 않은 합성 fixture 생성기. 설정은 `tests/fixtures/config.json`, 결과는 같은 폴더에 커밋한다.
- `tests/rds_writer.py`: Dunnhumby 모양의 합성 R 파일 생성기(테스트 안에서 임시 폴더에 만든다, ID는 990000 이상).

`rosters/dunnhumby_rb.csv`는 기존 기준의 점포 ID 101개만 가진 재현용 목록이다. 고객·거래·배정표는 포함하지 않는다. 실제 데이터·고객별 파생물은 Git에 넣지 않는다.
이전 탐색 분석 스크립트가 [fedcommerce/](../../../fedcommerce/README.md)에 있다. 현행 계약을 따르지 않는 참고 자료이므로 여기 어댑터는 계약에 맞춰 새로 쓴다.
공통 Python 타입은 `commerce.packages.contracts`, 모델 호출 진입점은 `commerce.packages.recommender.runtime`이다. 학습·평가 실행 코드는 `commerce/evaluation/`에 둔다.

기준과 한계: [데이터 설계](../../../docs/design/data.md), [B 작업 카드](../../../docs/team/tasks/B.md).
