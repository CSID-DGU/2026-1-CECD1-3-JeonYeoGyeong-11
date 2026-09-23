# 이전 탐색 분석 · B 참고용

이 디렉터리는 **커머스 구현 코드가 아니다.** 현행 프로젝트 코드는 `commerce/`이고 현행 설계는 [docs/](../docs/README.md)다. 여기 있는 것은 판매자 분할·이질성·어휘 정규화를 확인하려고 앞서 돌린 분석 스크립트이며, B가 데이터 어댑터를 만들 때 **재작성하지 않아도 되도록** 출발점으로 공개한다.

## 그대로 믿으면 안 되는 것

- 현행 계약(`commerce/packages/contracts/`)을 따르지 않는다. 컬럼 이름·ID 규칙이 [데이터 설계](../docs/design/data.md) §2의 `dh-store-{store_id}` 형식과 다르다.
- 여기서 나온 수치는 이전 실행 기록이다. [데이터 설계](../docs/design/data.md)가 "B가 출처·버전·취득 경로·로컬 파일 해시와 정제 결과를 다시 확인해 재현 기록을 남긴다"고 정한 이유다. 이 코드를 돌렸다는 사실만으로 재현을 보고하지 않는다.
- 게이트·CI가 검사하지 않는다. `b1`을 구현할 때 현행 계약에 맞춰 `commerce/packages/data_adapters/`에 새로 쓰고, 이 스크립트는 참고로만 본다.

## 원자료는 저장소에 없다

스크립트가 읽는 `data/transactions.rds`, `data/products.rda`, `data/instacart/*.csv`와 `out/` 결과물은 모두 `.gitignore` 대상이라 clone에 포함되지 않는다. 각자 원자료를 직접 취득해 같은 경로에 두어야 실행된다. 공개하는 것은 스크립트와 합성 fixture 두 개(`data/synthetic/product.csv`, `data/synthetic/_ground_truth.csv`)뿐이다.

출처 링크·배포판 구분·필요 파일·취득 기록은 [데이터 설계 §1 출처와 로컬 준비](../docs/design/data.md#출처와-로컬-준비)를 따른다. 기존 Dunnhumby 분석은 `completejourney` R 배포판 기준이며, 원출처의 CSV를 그대로 같은 파일로 취급하지 않는다. 탐색 스크립트는 `fedcommerce/`를 작업 디렉터리로 실행한다. 기존 Instacart 파일의 정확한 미러 주소는 아직 확인되지 않았다.

## B의 1번 작업에 가까운 순서

| 파일 | 하는 일 |
| --- | --- |
| `src/schema.py` | 원자료 컬럼 확인 |
| `src/make_synthetic.py` | 합성 fixture 생성 |
| `src/profile_stores.py`, `src/select_and_noniid.py` | Dunnhumby 점포 프로파일과 선정 |
| `src/instacart_clients.py`, `src/instacart_match.py` | Instacart 가상 client 분할·정합 |
| `src/vocab_norm.py`, `src/vocab_overlap.py` | 상품 텍스트 정규화와 출처 간 어휘 중복 |
| `src/heterogeneity.py`, `src/dirichlet_alpha.py` | 판매자 간 분포 이질성 측정 |

나머지는 이 위에 얹은 추가 분석이다. 자세한 배경은 [B 작업 카드](../docs/team/tasks/B.md)와 [데이터 설계](../docs/design/data.md)를 따른다.
