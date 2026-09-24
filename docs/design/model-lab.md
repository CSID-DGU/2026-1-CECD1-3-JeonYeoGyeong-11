# 독립 모델 실험과 플랫폼 초기 모델
갱신: 2026-09-24 · 소유: B(모델·실험), C(FL 실행·모델 등록) · 실행 기준 D0018·D0019
상태: 구현할 경계와 절차. 아래 실행기는 아직 없다.

## 1. 서비스 요구사항
서비스를 시연할 때마다 추천 모델을 처음부터 학습하지 않는다.
플랫폼 화면 개발과 별도로 모델을 학습·평가하고, 검증된 가중치를 플랫폼의 초기 버전으로 사용한다.
플랫폼 안에서는 추천을 즉시 제공하고 새 거래를 쌓은 뒤 별도 스케줄로 FL를 이어간다.

## 2. 두 개발 경로

~~~mermaid
flowchart LR
  Data[과거 데이터·합성 데이터] --> Adapter[B 어댑터]
  Adapter --> Lab[B 독립 모델 실험]
  Lab --> Local[판매자별 로컬 학습]
  Local --> FL[C 보호 집계 모듈]
  FL --> Local
  FL --> Check[B 평가·내보내기]
  Check --> Release[검증된 초기 모델]
  UI[A 구매·판매 화면] --> App[서비스 플랫폼]
  Release --> App
  App --> Inference[B 즉시 추천]
  App --> NewData[판매자별 새 거래]
  NewData --> Next[별도 FL 작업]
  Next --> NewRelease[다음 모델 버전]
  NewRelease --> App
~~~

모델 실험은 구매자 화면·판매자 화면·중앙 공개 카탈로그·주문 API에 의존하지 않는다.
B의 core 모델/손실/특징 모듈은 FastAPI 앱을 import하지 않는다. 오프라인 어댑터와 서비스 runtime이 **같은 모델·특징 계산 코드**를 호출한다.
C는 집계 모듈과 HTTP 서버를 분리하여 실험 실행기가 화면 없이 같은 집계 규칙을 사용할 수 있게 한다.

text_only와 text_relation을 같은 조건에서 각각 학습한다. 각 공통 모델에서 판매자별 query_proj/scorer만 추가로 학습하여 T-G/R-G/T-P/R-P를 평가한다. [comparison.md](comparison.md)의 네 결과는 두 공통 모델과 그 로컬 개인화 결과이며, 네 개의 별도 FL 실험을 뜻하지 않는다.

## 3. 어떤 부분을 처음부터 학습하는가

| 구성 | 초기 상태 | 비용 절감 |
| --- | --- | --- |
| 텍스트 encoder/tokenizer | 선정한 사전학습 모델, frozen | 언어모델 사전학습을 새로 하지 않음. 상품별 z 캐시 |
| 관계·fusion·basket·sequence·scorer | 첫 프로젝트 학습에서는 새로 학습 | 소형 E-G0 → 실험 학습 → checkpoint 재사용 |
| 판매자별 관계·이력·캐시 | 각 판매자의 관측으로 재구성 | 희소 관계와 snapshot. 공통 release에 넣지 않음 |
| 서비스 시작 모델 | 실험에서 승인한 release | 앱 시작 시 load. from-scratch 학습 강제 금지 |
| 판매자 개인화 | 해당 variant의 공통 base 복사본 | query_proj/scorer만 학습. 충분한 과거 데이터와 검증이 있을 때 적용 |

HAREX 공개 논문을 참고했다고 HAREX의 학습된 가중치를 확보한 것은 아니다. 모델 입출력이 달라 직접 호환된다고도 가정하지 않는다.
NLP 가중치 외의 추천 가중치는 실제 학습이 필요하다. 사전 학습은 그 비용을 서비스 시작 이전에 수행하는 것이며 비용 자체를 없애지는 않는다.

## 4. 실험 모드와 보호 경계

| 모드 | 역할 | 완료/주장 |
| --- | --- | --- |
| local_only | 한 판매자의 모델 학습·평가, 판매자별로 독립 반복 | G4 전에도 가능. 여러 판매자 데이터 합동 학습이 아님 |
| federated_synthetic | 합성 client를 별도 runtime으로 학습 후 집계 | C/B 인터페이스·수치 확인. 실거래 비노출 증명이 아님 |
| federated_lab_sim | 공개 데이터. 한 프로세스에서 판매자별 상태를 분리해 학습하고 C 집계 core로 합침 | G4와 무관하게 R1/R2 소형 표본에 사용. "비보호 FL 시뮬레이션"으로 표시(D0020) |
| federated_protected | 데이터는 seller별, C에는 보호된 제출 | G4 통과 후. 같은 실험을 보호 모듈로 다시 돌린 결과는 추가로 보고. A 화면 완성은 불필요 |

한 PC에서 여러 client를 순차 실행해도 로컬 학습→집계→재배포 규칙을 유지하면 FL 알고리즘의 시뮬레이션이 가능하다. 다만 같은 PC 관리자에 대한 데이터 격리 증명은 아니다.
기본 실험은 한 PC의 논리적 판매자들로 구성하고 여러 PC 배치는 선택적 시연으로 둔다([운영 가정](architecture.md)). 순차 실행에서도 같은 라운드의 모든 판매자는 **동일한 공통 base**에서 각각 출발한다. 앞 판매자가 학습한 모델을 다음 판매자의 초기값으로 넘기지 않는다. 판매자별 데이터·optimizer·개인화·캐시 상태를 분리하고, 선택한 cohort의 유효 업데이트를 모두 확보한 뒤 한 번 집계한다.
공개 데이터라는 이유로 서비스의 synthetic_plaintext 경로에 실데이터를 연결하지 않는다. 실험실 시뮬레이션은 서비스·HTTP를 거치지 않는 별도 경로다(D0020). 여러 판매자 원자료를 합쳐 한 optimizer로 학습한 결과는 FL 결과로 보고하지 않는다.
보호 모듈 G4는 합성 입력으로 화면 없이 독립 검증할 수 있다. 서비스 전체 비잔류 확인은 A 연결 후 별도로 한다. 최종 완료에는 둘 다 필요하다.

## 5. 실행·중단·재개
B 소유 commerce/evaluation/의 실행기는 dataset/split/roster/seed/model_variant/architecture_version/NLP artifact/모드/학습 budget/resume 경로를 명시한 설정을 읽는다. model_variant는 불변 architecture config에 포함되며, variant별 checkpoint·optimizer·결과 경로를 분리한다.
C 소유 FL 모듈을 호출하고 완료 round만 checkpoint로 확정한다. 설정과 결과 경로는 Git 제외 영역이다.
CPU와 GPU는 실행 장치 선택이며 모델 의미·입력 규격을 바꾸지 않는다. E-G0에서 실제 처리량과 메모리를 측정한 뒤 장치를 정한다.

- frozen z는 텍스트/encoder 설정 변경 때만 재계산한다.
- 구매 관계는 sparse 저장과 snapshot으로 재사용한다. 전체 상품 수의 제곱 크기 dense 행렬을 필수로 만들지 않는다.
- 관계 MLP가 학습되므로 최종 l/e까지 모든 라운드에서 영구 고정하지 않는다.
- global checkpoint는 완료 집계 weights와 round/config 상태를 보관한다. 로컬 optimizer·RNG·원장은 판매자에만 둔다.
- 라운드마다 optimizer를 초기화하는 현행 규칙을 유지한다. 중간 실패한 보호 라운드는 완료 global checkpoint에서 새 round로 다시 시작한다. 마스크/nonce를 재사용하지 않는다.
- 반복 실험은 같은 split·seed·예산으로 비교한다. FL와 local-only에 서로 다른 사전학습 이점을 주지 않는다.
- 개인화는 완료된 공통 checkpoint의 복사본에서 수행하고 결과를 판매자별 로컬 경로에 저장한다. 공통 checkpoint를 덮어쓰거나 다음 FL 초기값으로 사용하지 않는다.
- 개인화 설정과 학습/검증 prefix를 기록한다. 기본 오프라인 평가는 고정 적응 prefix로 개인화한 뒤 평가하며, 온라인 재학습 평가는 별도 프로토콜로 구분한다.

### 로컬 계산 비용과 느린 참여자

로컬 학습은 판매자의 계산 자원을 사용한다. 브라우저 단말 성능과 구분하며, 현재 설계만으로 일반 매장 PC에서 충분히 빠르다고 단정하지 않는다. 한 PC 실험에서는 여러 판매자의 계산이 같은 자원에 모이므로 작은 고정 표본·순차 실행부터 확인한다.

- 구매마다 전체 모델을 학습하지 않는다. 거래·특징 반영, 추천, 개인화, FL 라운드는 별도 작업이다. 구체적인 트리거·빈도는 OQ08에서 확정한다.
- frozen NLP와 텍스트 캐시, 제한 이웃·희소 관계, 로컬 step 상한, 후반부 개인화, 초기 release 재사용을 적용한다. 서빙과 학습 객체를 분리해도 CPU/GPU·메모리 경합이 없어지는 것은 아니다.
- B는 판매자 한 곳의 전처리/NLP·추천·로컬 학습·개인화 시간과 최대 메모리를 작은 표본에서 측정한다. A/B/C는 서비스 연결 후 학습 유무에 따른 주문·추천 응답 시간을 비교한다. OS·CPU/GPU·메모리·상품/이력 규모·모델 variant·캐시 상태·학습 설정을 함께 기록하고 실행 예산을 정한다. 개별 실제 거래 유래 지표를 중앙에 수집하지 않는다.
- 첫 FL는 고정 cohort 전원 완료 또는 전체 폐기다([집계 계약](interfaces.md)). 느린 판매자는 deadline까지 기다리고, 미완료가 있으면 새 공통 모델을 배포하지 않고 이전 정상 모델로 서비스한다. 지각 업데이트의 다음 라운드 이월은 첫 범위에 없다.
- 순차 시뮬레이션에서 뒤에 실행되는 판매자의 대기 시간을 그 판매자 컴퓨터가 느린 증거로 해석하지 않는다. C는 실제 로컬 계산 시간과 실행기 대기 시간을 구분하고, cohort 순차 실행 시간을 고려한 deadline을 설정에 기록한다. 인위적 지연을 넣으면 지연 모사라고 명시한다. 실제 여러 장비의 성능·통신 실험으로 보고하지 않는다.

## 6. 실험 → 서비스 release
1. B가 variant별 model config, 동일 preprocessing/NLP artifact, 공통 base weights, shared_model_manifest를 확정한다. 개인화 weights는 중앙 release에 포함하지 않는다.
2. architecture_version은 B 패키지의 **불변 config 등록 항목**과 일대일 대응한다. dropout/pooling/정렬처럼 shape가 같아도 의미가 다른 설정도 포함한다. 같은 번호의 config를 바꿔 checkpoint를 재사용하지 않는다.
3. C가 지정된 로컬 import 작업으로 config 등록값·manifest·해시·tensor·출처를 검증하고 model_release를 만든다. 외부 사용자가 임의 모델 파일을 업로드하는 API는 만들지 않는다.
4. variant별 중앙 registry에는 검증된 release를 원자 등록하고 각각의 latest로 지정한다. 첫 실행은 두 FL 실험을 순차 실행한다. 학습 완료 release와 random init을 구분해 운영 기록에 남긴다.
5. 각 판매자는 기존 latest/manifest/weights 경로로 받아 B install_release에 전달한다.
6. frozen text artifact와 config는 같은 버전의 B 패키지/설치 절차로 준비한다. weights에 없다는 이유로 임의 default 인코더를 선택하지 않는다.
7. 판매자는 자기 카탈로그·로컬 과거 이력으로 z/관계/cache를 만든다. 다른 판매자의 DB·상품 벡터를 복제하지 않는다.
8. 고정 합성 입력으로 실험 경로와 서빙 경로의 점수/순위를 비교한다. 장치별 수치 허용오차는 B가 정해 기록한다.

배포 묶음은 공통 weights/manifest/release이며 로컬 상태를 싣지 않는다. 실험용 설정·학습 provenance는 로컬 실행 기록으로 보관한다.
비교 시연용으로는 짝지은 두 공통 release의 식별자를 판매자 로컬 비교 설정에 고정한다. 개인화 결과를 재사용하려면 seller·variant·base 해시가 일치해야 하며, 해당 판매자의 로컬 경로에서만 읽는다.
사전학습 출처가 단일 판매자이면 그렇게 표시한다. FL로 만든 release와 혼동하지 않는다. 최종 서비스의 신규 판매자 평가 대상은 사전학습에도 포함하지 않는다.

## 7. 서비스 시작과 시연
앱 시작은 release load → 로컬 특징 준비 → 추천 가능 상태다. 학습 시작을 기다리지 않는다.
구매 완료는 로컬 event/feature_epoch를 갱신한다. 추천 표현은 바뀔 수 있지만 이것만으로 FL가 실행된 것은 아니다.
개인화는 별도 로컬 작업으로 실행한다. 준비 전에는 공통 모델로 서비스하고, 검증을 통과한 현재 base의 개인화 결과만 원자적으로 적용한다. 정상 서비스의 auto와 비교 화면의 명시적 personalized 모드는 [모델 경계](model.md)를 따른다.
정해진 시점의 FL 라운드가 새 shared weights를 배포하면 base_model_version이 바뀐다. 옛 개인화 후반부는 새 base에 붙이지 않고 새 버전에서 다시 학습한다. 시연에서는 준비한 release로 시작하고 짧은 추가 라운드로 이 차이를 보여준다.
실제 추천의 model_version은 공통 release 또는 로컬 개인화 식별자다. 구매에 따른 feature_epoch, FL의 base_model_version, 개인화의 personalization_revision을 구분한다.
두 variant는 같은 판매자 특징 원장을 읽고 이벤트는 한 번만 반영한다. 서비스의 네 결과 비교는 B compare_local이 동일 snapshot을 고정하는 절차를 따른다.

서비스 실패/재시작에도 마지막 정상 release를 유지한다. 신규 판매자가 자기 이력까지 전혀 없으면 fallback을 표시한다.

## 8. 분업 완료 조건
B: 화면 없이 두 variant 학습/평가/재개/export, 공통 base와 개인화 분리, 2×2 및 local-only 비교, 같은 snapshot의 compare_local.
C: 화면 없는 FL orchestration, G4 보호 모듈 검증, variant별 초기 release import, round 재개와 개인화 미제출 확인.
A: 준비된 release로 즉시 추천하고 outbox로 새 거래 전달, 네 결과/불가 사유 표시. 사전학습 프로그램을 앱 startup에 넣지 않음.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](open-questions.md)를 확인한다.
