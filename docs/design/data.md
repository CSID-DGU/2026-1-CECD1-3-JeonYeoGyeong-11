# 데이터와 전처리
갱신: 2026-09-24 · D0017 · 소유: B

## 1. 데이터가 대표하는 상황

독립 판매자가 공통 플랫폼에 참여하는 시뮬레이션이다. 공개 데이터의 원출처가 서로 독립적인 실제 기업이라는 증거로 설명하지 않는다. Dunnhumby의 점포 분할과 Instacart의 가상 분할을 명시한다. 서로 다른 스키마를 공통 로컬 이벤트로 바꿔도 원자료에 없는 수량·절대시각·사업자 정보가 생기지는 않는다.

아래 수치는 이전 분석 기록에서 인계한 값이다. 이번 문서 정리에서 원자료 전체를 재계산하지 않았다. 공개 코드의 성능·재현 결과로 주장하지 않는다. B가 출처·버전·취득 경로·로컬 파일 해시와 정제 결과를 다시 확인해 재현 기록을 남긴다.

| 항목 | Dunnhumby Complete Journey | Instacart |
| --- | --- | --- |
| 이전 원자료 측정 | 거래 1,469,307행 / 점포 457 / basket 155,848 / household 2,469 | orders 3,421,083 / prior orders 3,214,874 / prior lines 32,434,489 |
| 상품 master | 92,331, 일반 상품명 없음 | 49,688, product_name 있음 |
| 시각 | 거래 절대시각, 이전 범위 2017-01-01~2018-01-01 | order_number와 days_since_prior_order, 간격 30일 cap |
| 수량 | 관측됨 | 미관측 |
| 판매자 | store를 독립 판매자로 가정 | retailer ID 없음, 고객 단위 가상 분할 |
| 기존 기준 roster | 101개 점포 | 100개 가상 client, 8,172명 / 129,586 prior orders |
| 한계 | 동일 유통계열의 점포를 독립 주체로 시뮬레이션 | 실제 서로 다른 소매업체의 자료로 주장할 수 없음 |

**판매자 간 차이에 대한 이전 기록.** 이전 탐색의 로컬 기록(`fedcommerce/DESIGN.md`, 저장소에 없음, 재현 전)에 따르면 DH 점포 간 이질성의 80~87%는 가구를 점포 하나에 묶은 데서 온다. 가구 무작위 분할과 비교하면 1.0~1.5배에 그친다. IC 가상 client의 이질성은 Dirichlet α=0.25 분할로 넣은 값이다(`fedcommerce/src/instacart_match.py`). 따라서 DH에서 FL이 local_only보다 좋아도 서로 다른 판매자 연합의 증거로 해석하지 않는다. IC의 FL·개인화 결과는 α에 달려 있다고 함께 적는다.

Instacart는 prior만 사용한다. 공식 train/test를 이 프로젝트의 시간 분할과 혼동하지 않는다. 이전 가상 표본은 전체 206,209 고객 중 약 3.96%다. 전수 실험으로 표현하지 않는다.

### 출처와 로컬 준비

데이터셋의 원출처, 배포판, 실제로 받은 파일을 구분한다. 아래 출처 안내는 2026-09-24에 확인했으며 과거 로컬 파일의 다운로드 날짜·정확한 revision까지 복원한 것은 아니다.

| 구분 | 출처·취득 안내 | 이 프로젝트에서의 취급 |
| --- | --- | --- |
| The Complete Journey 원출처 안내 | [dunnhumby Source Files](https://www.dunnhumby.com/source-files/) | 2년·2,500가구로 소개하는 원자료 안내. 아래 R 배포판과 동일 파일·기간이라고 가정하지 않는다 |
| 현재 Dunnhumby 분석의 배포판 | [completejourney 공식 문서](https://bradleyboehmke.github.io/completejourney/)·[관리자 저장소](https://github.com/bradleyboehmke/completejourney) | 기존 분석 기록은 이 R 배포판을 사용했다고 명시한다. 배포판은 출처를 84.51°로 밝히며 1년·2,469가구를 설명한다 |
| Instacart 2017 | [기존 공식 대회 주소](https://www.kaggle.com/c/instacart-market-basket-analysis/data)·[현재 Kaggle Market Basket Analysis 데이터 페이지](https://www.kaggle.com/competitions/basket-analysis/data) | 기존 기록에는 미러 검증이 있으나 정확한 취득 미러 URL·버전은 미확인이다. 이번 확인에서 기존 주소의 본문을 읽지 못했고 현재 페이지의 다운로드 가능 여부·이용 조건도 확인하지 못했다. B가 계정으로 접근해 실제 취득 경로를 기록한다 |

Complete Journey는 `transactions_sample`을 전체 데이터 대신 사용하지 않는다. 관리자 문서의 [get_transactions()](https://bradleyboehmke.github.io/completejourney/reference/get_transactions.html)와 [products](https://bradleyboehmke.github.io/completejourney/reference/products.html)를 기준으로 준비하고, 사용한 패키지 버전·저장소 commit 또는 실제 다운로드 URL을 고정한다. 공식 CSV 원본을 받았다면 기존 R 파일과의 변환·기간 차이를 확인하고 이전 분석 수치를 그대로 재사용하지 않는다.

이전 탐색 스크립트용 경로는 `fedcommerce/data/transactions.rds`, `fedcommerce/data/products.rda` 및 `fedcommerce/data/instacart/`다. Instacart에서 사용하는 파일은 `orders.csv`, `order_products__prior.csv`, `products.csv`, `aisles.csv`, `departments.csv`이며 `order_products__train.csv`는 현행 학습 입력에 쓰지 않는다. 스크립트의 상대 경로 기준 작업 디렉터리는 `fedcommerce/`다. 새 어댑터·실험 산출물의 경로는 §6을 따른다.

B는 실제 사용 파일별로 `출처 URL / 배포판·revision / 취득일 / 파일명·크기·SHA-256 / 스키마·행 수 / 이용 조건 확인 근거`를 Git 제외 실행 기록에 남긴다. 미확인 값은 미확인으로 표시하고 행 수가 같다는 이유만으로 같은 파일이라고 판정하지 않는다. 출처 인용과 원자료 재배포 허용은 별개이며, 이 안내는 원자료의 Git 업로드 허가가 아니다. 조원 간 파일 해시·전처리 설정을 맞춘 후 재현 완료를 보고한다.

## 2. 정제와 판매자 배정

Dunnhumby 순서: product_category 유효 → quantity>0 → sales_value>=0 → COUPON/MISC ITEMS 제외 → household를 가장 많은 basket을 가진 점포 하나에 배정(동점 숫자 store_id 오름차순) → 300 basket 이상 점포 선택.
제외 순서를 바꿔 나온 102개 결과와 기존 101개 roster를 섞지 않는다. 기존 보고의 최종 basket은 104,011개이며 B가 새 어댑터 실행으로 재현/차이를 설명한다.

**판매자 배정과 표본 선정에는 train 구간만 쓴다.** validation/test 구간의 방문이 배정 라벨이나 선정 기준에 들어가면 판매자 구성이 평가 정답과 상관되고, local_only와 개인화가 유리해진다.
- IC: 이전 스크립트는 고객의 전체 prior 주문으로 주력 aisle(배정 라벨)을 계산했다(`instacart_match.py` 70~76행). 각 고객의 floor(0.7n) 이전 주문만으로 다시 계산한다.
- DH: 주이용 점포 배정과 300 basket 기준을 2~39주로 계산한다. 아래 기준 roster는 전체 기간으로 만든 것이므로, B는 새 계산 결과와의 차이를 보고한다.

기준 roster: [dunnhumby_rb.csv](../../commerce/packages/data_adapters/rosters/dunnhumby_rb.csv). 공개하는 이 파일은 store_id 101개만 포함하며 고객·거래·배정표는 포함하지 않는다.
파일 SHA-256: bb1d8e9c503b966be93a915b2d4d0c031ea4e7bbec34d8de0d1c9abe339965a5
이번 정리에서 파일 존재와 해시는 확인했다. 내용 생성의 원자료 재현은 b1 작업이다.

기존 Instacart 배정을 만든 스크립트는 [fedcommerce/](../../fedcommerce/README.md)의 `src/instacart_clients.py`·`src/instacart_match.py`이며 저장소에 있다. 그 출력 `fedcommerce/out/instacart/client_assignment_matched.csv`는 고객별 파생물이라 clone에 포함되지 않는다. B가 재현 가능한 분할 생성 절차를 구현하기 전에는 실제 데이터 재현이 미완료다. 원자료 없이 공통 코드·합성 검사는 시작할 수 있다.
client_id는 0~99가 아니라 기존 점포 ID를 가져온 정수값이다. 숫자로 정렬하고 이를 고객/상품의 교차 출처 동일성으로 해석하지 않는다. B는 파일 해시와 배정 seed/알고리즘을 실행 기록에 남긴다. 고객 한 명의 주문은 한 client에만 둔다.

| 출처 | seller_id | customer_id_local | item_id_local | basket_id_local |
| --- | --- | --- | --- | --- |
| Dunnhumby | dh-store-{store_id} | dh-hh-{household_id} | dh-p-{product_id} | dh-b-{basket_id} |
| Instacart | ic-client-{client_id} | ic-user-{user_id} | ic-p-{product_id} | ic-o-{order_id} |
| live | A가 발급한 seller ID | 판매자별 내부 고객 ID | 판매자별 상품 ID | 주문 ID |

ID가 우연히 같아도 출처를 조인하지 않는다. 고객을 전 판매자 통합 ID로 만들지 않는다. 원자료의 같은 basket·상품 행은 로컬에서 하나로 합치며 Dunnhumby 수량은 정수 관측값 합계, Instacart는 null을 유지한다.

Instacart 첫 주문의 누적 상대일은 0, 이후 days_since_prior_order를 누적한다. 30은 실제 30일 이상일 수 있어 누적값은 실제 기간의 하한이다. 결측/검열 비트를 보존한다. 고객 간 상대일 10을 같은 실제 날짜로 비교하지 않는다.

## 3. NLP 텍스트 생성

공통 함수 build_product_text는 출처 어댑터와 live catalog_item 모두에서 B가 구현한다.

| 입력 | 기본 텍스트 구성 |
| --- | --- |
| Dunnhumby products | [CAT] product_category [TYPE] product_type [SIZE] package_size |
| Instacart products+aisles+departments | [NAME] product_name [AISLE] aisle [DEPT] department |
| live catalog_item | [NAME] title_text + 존재하면 [DESC] description_text + category_path가 있으면 [CAT] 경로 |

원문 title은 표시용으로 보존하고 인코더 입력은 B builder가 별도로 만든다. product_text라는 임의 필드를 JSON 계약에 추가하지 않는다. Dunnhumby의 대체 제목을 실제 고유 상품명으로 보고하지 않는다.

- NFC·공백 정리, 결측 필드는 생략. 모든 필드가 비면 오류/격리하고 빈 벡터를 성공으로 반환하지 않는다.
- 한국어와 영어, 규격의 소수·분수·단위를 보존한다. 비ASCII 삭제 금지. 소문자화는 선택 인코더 규칙에 따른다.
- null을 문자열 "nan"/"None"으로 만들지 않는다.
- 원문·정규화 후·token truncation 후의 중복률, 결측률, 한국어 보존을 측정한다.
- 마커는 기본 tokenizer의 일반 문자열로 처리한다. 새 special token을 frozen 모델에 임의 추가하지 않는다.
- 정확한 checkpoint/revision, tokenizer, pooling, 길이·차원은 [모델 경계](model.md) §2의 B 산출물이다.
- 규격만 다른 상품, 한국어 유기농 우유/일반 우유, 무가당/가당 등 fixture를 포함한다.

구매 데이터로 fit하는 사전·통계가 생기면 training split 안에서만 만들고 로컬 상태로 보관한다. 공통 preprocessing artifact에 고객 통계를 넣지 않는다.

## 4. 서비스 거래와 과거 데이터의 연결

서비스에서는 모든 판매자가 같은 purchase_event 스키마를 사용한다. 과거 데이터 두 종류는 어댑터를 거쳐 같은 basket 표현으로 변환한다. 서비스 주문 상태는 A, 원자료 정제는 B가 소유한다.

B 내부 학습 레코드는 seller/customer/basket, item 집합, 관측 시간·순서·검열 정보와 source를 가진다. 실제 모델은 수량 대신 구매 여부를 쓴다. 상대시간·절대시간 원본 차이는 내부 메타데이터에 남기고 다른 고객의 가상 날짜를 만들지 않는다.

A 완료 이벤트의 durable outbox와 B 멱등 처리는 [인터페이스 계약](interfaces.md) §2. 완료 전/취소 주문은 학습 입력이 아니다. 이벤트 재생과 상품 수정은 feature_epoch와 캐시 무효화를 동반한다.

## 5. 원자료 없이 시작하는 fixture

A/C는 합성 상품·계정·주문으로 시작한다. B는 소형 두 출처 어댑터 fixture와 live fixture를 제공한다. 실데이터 행을 조금 잘라 합성이라고 부르지 않는다.

합성 통합은 학습 판매자 3개와 전혀 다른 상품 목록을 가진 보류 판매자 1개를 포함한다. 보류 판매자는 다음 두 상태를 구분한다.
- 로컬 과거 구매는 있으나 공유 학습에 참여하지 않음: 신규 판매자 개인화 점수화.
- 로컬 구매도 없음: 정직한 fallback 확인. 이것만으로 모델의 신규 판매자 성능을 입증하지 않음.

판매자마다 미구매 신상품을 남기고, 공동구매 관계를 생성하는 통제 이벤트를 추가한다. 상품 목록의 의미상 중복과 ID 중복을 각각 확인한다. 예제 수·seed는 fixture 생성 설정에 기록한다.

## 6. 파일 반출과 완료 조건

원자료, 고객별 배정표, 전처리 구매 이벤트, 관계·캐시·개별 업데이트·실데이터 모델은 기본 Git 제외다. 공개할 수 있는 schema·직접 만든 합성 fixture·집계 보고서만 별도 검토한다. 라이선스 확인과 공개 범위 기록은 B가 맡는다. 검토는 생성기를 새로 만들거나 생성 규칙을 바꿀 때 한 번 받는다([Git 협업](../team/git-workflow.md) 첫 표).

직접 만든 합성 fixture는 커밋 대상이므로 자기 소유 경로의 `tests/fixtures/` 아래 두고, 생성기와 seed·개수 설정을 함께 커밋해 같은 결과를 다시 만들 수 있게 한다. 실행 산출물은 commerce/evaluation/data/, cache/, runs/, outputs/ 또는 commerce/deploy/var/ 아래에 두며 이들은 Git 제외다. 기존 fedcommerce/out/도 공개 안전성을 확인하기 전에는 기본 제외한다. 구입 이력이 담긴 파일을 새 폴더로 옮겼다고 반출 허용이 되지 않는다.

b1 완료: 소형 입력으로 정제 순서·ID·수량 null·시간 종류·중복·한국어 보존 검사. 별도로 로컬 원자료 재현 보고(행 수/제외 사유/roster/해시)를 만든다. 합성 fixture 통과와 원자료 재현 완료를 따로 보고한다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](open-questions.md)를 확인한다.
