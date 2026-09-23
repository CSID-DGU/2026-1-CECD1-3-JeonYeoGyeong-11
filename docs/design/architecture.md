# 아키텍처 — 판매자 로컬 서비스와 보호 집계

갱신: 2026-09-23 · D0017~D0019 · 소유: A · 목표 설계이며 구현 완료 표시가 아니다.

## 1. 전체 구성

~~~mermaid
flowchart LR
  Buyer[구매자] --> Public[중앙 공개 카탈로그]
  Public -->|판매자 접속 위치| Buyer
  Buyer --> MerchantWeb[판매자 origin의 구매 화면]
  Seller[판매자] --> SellerWeb[판매자 운영 화면]
  subgraph Local[판매자마다 독립한 로컬 환경]
    MerchantWeb --> OrderAPI[A 주문·계정·상품 API]
    SellerWeb --> OrderAPI
    OrderAPI --> Orders[(A 주문 DB·이벤트 outbox)]
    Orders --> Delivery[A 재전달 루프]
    Delivery --> Features[(B 특징 DB)]
    Features --> Serving[B 로컬 추천]
    Serving --> MerchantWeb
    Features --> Trainer[B 학습 복사본]
    Trainer --> Client[C 판매자 FL client]
  end
  subgraph Central[중앙]
    Public
    Coordinator[C FL 조정]
    Aggregate[C 보호 집계]
    Registry[집계 모델·manifest]
  end
  OrderAPI -->|공개 상품 snapshot만| Public
  Client -->|보호된 제출| Aggregate
  Coordinator -->|라운드 안내| Client
  Aggregate --> Registry
  Registry -->|검증 후 B 설치| Serving
~~~

A는 플랫폼·주문, B는 특징·모델, C는 FL·배포를 소유한다. 실제 컴포넌트 경로·포트·환경변수는 [작업 규칙](../team/working-agreement.md) §3을 따른다.

위 그림은 서비스 실행 구조다. 사전 모델 학습은 화면과 별도의 [실험 경로](model-lab.md)로 수행하고 검증한 release를 초기 모델로 설치한다(D0018). 서비스 startup에 처음부터의 모델 학습을 넣지 않는다.

## 2. 중앙이 알아도 되는 것

| 정보 | 중앙 허용 | 위치·처리 |
| --- | --- | --- |
| 판매자 접속 위치·공개 상품 텍스트·공개 상품 ID | 허용 | 공개 카탈로그. 판매 실적·재고·고객 연결은 제외 |
| 고객 계정·고객별 주문·바스켓·추천 | 금지 | 해당 판매자 DB·화면에만 |
| 상품별 관계·고객 시퀀스·임베딩 캐시 | 금지 | 판매자 특징 DB |
| 개인화 가중치·optimizer·개인화 검증 결과 | 금지 | 해당 판매자의 모델 저장소 |
| 개별 평문 모델 업데이트·개별 검증 손실 | 최종 경로에서 금지 | 판매자 FL client가 보호한 뒤 제출 |
| 집계 결과·공동 모델·집계 지표 | 허용 | 모델 registry, 운영 로그 |
| 참여 seller·라운드·버전·성공/실패 | 최소 허용 | 운영 메타데이터. 활동 정보가 일부 보인다는 한계 명시 |

보호 집계는 여러 입력의 합을 서버가 얻고 개별 입력을 숨기는 방식이다. 참여자가 많다는 사실, HTTPS, 업데이트 삭제만으로 이를 대신하지 않는다. [Secure Aggregation 원 논문](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/)의 보장도 선택 프로토콜·참여/이탈·공모 가정 아래 성립한다.

개별 gradient에서 학습 데이터를 복원한 연구가 있으므로 업데이트를 원본과 무관한 안전한 값으로 취급하지 않는다. 이는 모든 FL 모델에서 같은 복원이 된다는 뜻은 아니다. [Deep Leakage from Gradients](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)

## 3. 개발 모드와 최종 모드

| 모드 | 입력 | 중앙이 보는 것 | 완료 의미 |
| --- | --- | --- | --- |
| synthetic_plaintext | 생성한 합성 상품·고객·주문만 | 개별 delta·metrics를 일시 처리 | G3 배선 확인. 보호 완료 아님 |
| protected | 실제 서비스·과거 거래 유래 학습 | 보호된 제출, 허용 집계만 | G4 프로토콜 검증 후 활성화 |

FL client는 신뢰된 데이터 설정에서 입력 출처를 확인한다. 실제 데이터 로컬 참조를 합성 모드에 연결하면 실패한다. “source=synthetic” 문자열만 바꾸어 우회하는 구조를 만들지 않는다.

C는 첫 작업에서 검증된 보호 집계 구현을 조사·작동 확인한다. 암호·mask 복구를 임의로 직접 설계하지 않는다. 구현 라이브러리·버전, 서버/참여자 위협 가정, 최소 참여 수, 이탈 규칙, 재시도와 라운드 식별을 결정 기록에 남기고 보호 프로토콜 메시지 schema/fixture를 먼저 만든다. G4 전에는 protected가 조용히 평문으로 fallback하지 않고 실패해야 한다.

초기 위협 범위는 정상 프로토콜을 따르며 개별 정보에 관심을 갖는 중앙을 포함한다. 악성 서버의 조작, 판매자 공모, 호스트 관리자 침해에 대한 보장은 선택 구현의 지원 범위를 명시하고 별도로 검토한다. 한 PC에서 실행하는 시연은 호스트 관리자 격리를 증명하지 않는다.

집계된 모델 자체에서의 모든 추론을 없애는 보장은 별도다. 그런 보장이 필요하면 record/user 수준의 DP와 예산·품질 영향을 추가 설계해야 한다. 이번에는 “보안 이슈가 전혀 없다”라고 표시하지 않는다.

## 4. 주문 완료와 장애 복구

~~~mermaid
sequenceDiagram
  participant U as 구매자
  participant A as A 판매자 API
  participant DB as A 주문 DB
  participant O as A outbox 전달
  participant B as B 특징·추천
  U->>A: 주문 요청
  A->>DB: requested 저장
  A-->>U: 주문 ID·상태
  Note over A,DB: 판매자가 수락 후 완료
  A->>DB: completed + purchase_event + outbox를 한 트랜잭션으로 커밋
  O->>B: ingest_purchase_event(event)
  B->>B: 중복 키 확인·이벤트·feature_epoch 원자 반영
  B-->>O: 영속 반영 성공
  O->>DB: 전달 완료 표시
  Note over O,B: 중간 실패·재시작이면 같은 ID로 재전달
  U->>A: 추천 요청
  A->>B: predict_local
  B-->>A: 로컬 점수·모델 버전
  A-->>U: 추천 화면
~~~

A DB의 미전달 행은 성공 전까지 남는다. outbox 삭제/retention은 소비 완료 확인 이후에만 한다. 카탈로그 변경도 source_seq를 가진 로컬 outbox로 전달하고 기동 시 미전달 변경을 재생한다.

## 5. 학습과 서빙

~~~mermaid
flowchart LR
  Event[완료 구매 이벤트] --> Feature[B 특징 snapshot]
  Text[상품 텍스트] --> NLP[B frozen NLP]
  NLP --> Input[상품 표현·구매 시퀀스]
  Feature --> Input
  Base[검증된 공통 base] --> Train[전체 추천 가중치의 FL 학습 복사본]
  Feature --> Train
  Train --> Secure[C 보호 집계]
  Secure --> Release[variant별 공통 release]
  Release --> Check[B 해시·manifest 검증]
  Check -->|원자 설치| Base
  Base --> Personal[로컬 복사본의 query/scorer만 개인화]
  Feature --> Personal
  Personal --> Validate[로컬 검증·base 일치 확인]
  Validate --> Tail[현재 base의 개인화 모델]
  Base --> Live[서빙 모델 선택]
  Tail --> Live
  Input --> Live
~~~

이 흐름을 text_only/text_relation별로 적용한다. 고정 NLP를 제외한 해당 variant의 추천 가중치는 FL에서 모두 공유하며, 관계를 해석하는 MLP도 공유한다. 실제 관계 값·상품 벡터·고객 이력은 로컬이다.

진행 중 학습은 화면의 서빙 객체를 수정하지 않는다. FL는 지정된 공통 base에서 시작하며 개인화 결과를 집계에 넣지 않는다. 새 base 설치 후에는 공통 모델로 서비스하고, 새 base에서 개인화가 검증되면 전환한다. 설치 실패 시 이전 정상 서빙을 유지한다.

구매는 feature_epoch, FL는 base_model_version, 개인화는 personalization_revision을 갱신한다. 실제 추천 model_version은 선택한 공통/개인화 모델을 식별한다. 이 세 변화를 하나의 학습 완료로 표시하지 않는다.

## 6. 화면과 검증 범위

중앙은 공개 상품 목록과 판매자 홈 링크만 제공한다. 구매·내 이력·내 추천 HTML은 판매자 origin에서 제공하고 중앙 스크립트를 import하지 않는다. 고객 계정은 판매자 로컬이다. 데모의 단순 로그인은 인증 완성으로 주장하지 않는다.

상품 등록·수정은 A가 저장하고 B에 전달한다. 공개 snapshot은 비밀 필드를 제외해 중앙에 게시한다. 중앙은 같은 seller의 공개 행을 교체한다. 고객·주문 ID는 URL과 중앙 로그에 넣지 않는다.

비교 화면도 판매자 origin에서 제공한다. A가 B compare_local을 한 번 호출하면 B는 같은 이력·카탈로그·후보와 짝지은 checkpoint를 고정해 네 결과를 반환한다. 준비되지 않은 개인화는 이유를 표시한다. 이벤트 원장은 한 번만 갱신하며 비교 결과를 중앙 분석 서버로 전송하지 않는다. 세부 화면/시나리오는 [comparison.md](comparison.md)를 따른다.

호스트별 세션 쿠키를 사용한다. 쿠키는 포트로 격리되지 않으므로 localhost의 포트만 나누지 않는다. [RFC 6265 §8.5](https://httpwg.org/specs/rfc6265.html#weak-confidentiality)

G2는 주문 왕복과 중앙 DB·로그·쿠키 요청을 검사한다. catalog_snapshot의 공개 item_id_local은 허용하고 고객 ID·주문 ID·개별 추천은 허용하지 않는다. G4는 별도로 보호 집계 서버의 입력·출력·로그·임시파일 경계를 검사한다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](open-questions.md)를 확인한다.
