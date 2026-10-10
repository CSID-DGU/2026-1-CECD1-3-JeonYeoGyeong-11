# 화면 없는 모델 실험 · B

`encoder_probe.py`는 frozen 인코더 후보를 같은 입력으로 비교한다. `text_collisions.py`는 출처별로 텍스트·token이 다른 상품과 같은 비율을 판매자 카탈로그 안과 전체에서 잰다(OQ04). 결과와 선택은 [NLP 인코더 선정](../../docs/design/nlp-encoder.md)에 있다. `e_g0.py`는 E-G0(텍스트만 기준선의 소형 첫 실행)이다. D0022 비교 실행기는 다음과 같다.
- `harex_compare.py`: 네 상품 표현과 관계 셔플 대조를 local_only로 판매자마다 검증 손실이 멈출 때까지 학습·평가한다. `--holdout-frac`이면 C-new다.
- `fl_lab.py`: 같은 variant의 federated_lab_sim(비보호 FL 시뮬레이션, D0020). 라운드 수를 고정하고 마지막 라운드가 결과다. 집계는 C core가 들어오기 전까지 같은 규칙의 임시 균등 평균이다. 공유 가중치를 기록 옆에 저장한다.
- `cold_start.py`: 저장된 공유 가중치로 학습에 참여하지 않은 보류 판매자(A-0)를 로컬 갱신 없이 평가한다.
- `gci_protocol.py`: GCI 논문의 평가 조건(상품 단위 5개 창, 무작위 분할). `--protocol gci`로 두 실행기에 쓰는 추가 범위 재현이며 본 결과를 대신하지 않는다.
- `gci_original.py`: GCI 원 모델(상품명 생성 + Jaccard 매칭, glocal FL)을 같은 조건에서 재현한다. 단독 학습·FL·데이터 통합 학습을 논문 Table 4처럼 낸다. 추가 범위다.
- `a_few.py`: 보류 판매자에서 모델 비교의 2×2(T-G·R-G = A-0, T-P·R-P = A-few)를 실제 판매자 runtime으로 평가한다. 고객마다 마지막 방문이 정답이고 그 전 방문만 원장·개인화에 쓴다. 기본은 보류 판매자 11~20번(1~10번은 개인화 설정을 고르는 데 썼다).
- `d0022_report.py`: 실행 기록을 모아 결과표와 1차 대비의 판매자 단위 paired bootstrap 구간을 Markdown으로 낸다.
- `release_bundle.py`: lm 계열 FL 실행의 공유 가중치를 서비스 첫 release 폴더(release.json·manifest.json·weights.npz와 출처 provenance.json)로 만든다. 실행의 텍스트 artifact·전처리가 지금 코드와 다르면 거부한다.
- `service_check.py`: 보류 판매자를 실제 판매자 runtime에 넣어(카탈로그·마지막 방문 전까지의 이벤트) release 설치 전후의 추천을 마지막 방문으로 채점하고 지연을 잰다. 서비스 경로 점검이며 D0022 표가 아니다.

채점은 A의 `metrics/ranking.py`로 한다. 그 전까지 쓰던 B의 임시 `scoring.py`는 같은 정의였고(무작위 순위 500개·기준선·평균에서 소수 12자리까지 같음, #37 교차검증), A 모듈이 들어온 뒤 지웠다. `metrics/`(평가 지표와 모델이 아닌 기준선)는 A 소유이고, 나머지 실행기는 B 소유다([작업 규칙](../../docs/team/working-agreement.md) §1). `commerce.packages.recommender`의 같은 모델 core를 사용하며 FastAPI 앱에 의존하지 않는다.
작업 순서는 [B 카드](../../docs/team/tasks/B.md)를 따른다. 손실 함수와 예제별 특징 cutoff는 본 학습 전에 확정한다([열린 구현 항목](../../docs/design/open-questions.md)).
산출물은 이 디렉터리의 Git 제외 `data/`, `cache/`, `runs/`, `outputs/`에 둔다. 집계 core는 C에게서 제공받으며 그 API는 선정 후 공동 확정한다. 모델 학습이 서비스 startup에서 실행되게 하지 않는다.
