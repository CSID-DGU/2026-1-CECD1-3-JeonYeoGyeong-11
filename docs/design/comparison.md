# 모델 비교와 서비스 시연
갱신: 2026-09-23 · D0019 · 소유: B(비교 정의), A(화면), C(배포/통합)

## 1. 확인할 질문
주 질문: 같은 FL·구매 시퀀스·추천·개인화 조건에서, 상품 텍스트에 판매자 로컬 구매 관계를 추가하면 어떤 차이가 나는가?
정량 판단은 EVALUATION의 오프라인 평가, 시연은 동일 상태에서의 추천 비교다. 서비스에서 목록이 달라졌다는 사실만으로 성능 우위를 주장하지 않는다.

## 2. 두 모델과 네 결과

| variant | 상품 표현 | 공통 조건 |
| --- | --- | --- |
| text_only | e = Fusion(z_text, zero_l) | 같은 NLP, Fusion 입출력 차원, basket/sequence/query/scorer 설계 |
| text_relation | e = Fusion(z_text, l_relation) | 고객·basket·방향별 시간 관계를 추가 |

text_only의 zero_l은 학습 시작부터 고정된 0이다. 관계 MLP/Time MLP/pool은 실행·학습·전송하지 않으며 해당 manifest에서 제외한다. Fusion의 관계 입력 폭은 맞추지만 미사용 열을 학습된 관계라고 세지 않는다. 초기화/weight decay로 값이 존재해도 입력이 0이면 관계 정보를 쓰는 것은 아니다.
text_relation의 manifest는 MODEL_BOUNDARY의 9개 공유 그룹을 포함한다. text_only는 관계 전용 3개 그룹을 제외한 6개 그룹이다. seq_time_pos는 구매 시퀀스 입력이므로 둘 다 유지한다.
서로 다른 variant/architecture의 업데이트는 섞어 집계하지 않는다. 같은 tensor shape만으로 같은 실험이라고 판정하지 않는다.

| 결과 ID | 상품 표현 | 파인튜닝 | 표시 이름 |
| --- | --- | --- | --- |
| T-G | text_only | 없음 | 텍스트 · 공통 모델 |
| R-G | text_relation | 없음 | 관계 결합 · 공통 모델 |
| T-P | text_only | query_proj + scorer | 텍스트 · 판매자 개인화 |
| R-P | text_relation | query_proj + scorer | 관계 결합 · 판매자 개인화 |

T-G↔R-G는 개인화 전 관계 결합 효과, T-P↔R-P는 같은 개인화 조건에서의 관계 결합 효과다.
T-G↔T-P, R-G↔R-P는 후반부 파인튜닝 효과다. T-G↔R-P만으로 관계 모듈 효과를 설명하지 않는다.
이는 주 비교표의 결과 ID이며 기존 신규 판매자 A-0/A-few, 신상품 C-new/C-natural과 다른 축이다.

## 3. 텍스트 기준선의 의미
두 모델 모두 고객의 과거 구매 방문 시퀀스를 사용한다. 텍스트 기반이라는 이유로 이력을 없애지 않는다.
단순 상품명 유사도는 E-G0 sanity check일 수 있으나 주 비교의 T-G를 대신하지 않는다.
두 모델은 각각 독립 학습한다. 관계 모델 학습 후 l만 0으로 만드는 C0는 관계 제거 ablation이며 T-G가 아니다.
HAREX 원본은 상품명 토큰 생성과 매칭으로 출력이 달라 주 비교에 사용하지 않는다. 설계 참고와 별도 재현 비교를 구분한다.

## 4. 동일 조건과 비용
같은 데이터/roster/split/cutoff/후보/정답/음성 추출/FL 참여 일정/학습 budget을 사용한다.
NLP artifact와 공통 층 초기값을 맞추되, 두 모델은 별도 optimizer/checkpoint로 학습한다. relation 전용 초기화가 공통 층 초기값을 바꾸지 않도록 초기화/샘플링 seed를 관리한다.
개인화는 같은 데이터 prefix·검증 기준·대상 층·step cap을 적용한다. 공통 checkpoint 선택도 동일한 사전 정의 기준을 따른다. 서로 다른 test 결과를 보고 유리한 checkpoint를 골라 비교하지 않는다.
같은 step 수가 같은 계산량을 뜻하지 않는다. 전체/학습 가능한 파라미터 수, 전처리·학습·개인화·추론 시간, 메모리와 전송량을 따로 기록한다. 관계 추가와 모델 용량 증가가 함께 바뀐 비교라는 한계도 설명한다.
FL 자체의 이득은 동일 variant의 local_only와 비교한다. 텍스트/관계 2×2만으로 FL 효과를 입증하지 않는다.

## 5. 서비스 비교 흐름

~~~mermaid
flowchart LR
  Input[동일 판매자·고객·요청] --> Pin[B가 이력·카탈로그 snapshot과 후보 고정]
  Pin --> TG[텍스트 공통]
  Pin --> RG[관계 공통]
  Pin --> TP[텍스트 후반부 개인화]
  Pin --> RP[관계 후반부 개인화]
  TG --> View[A 판매자 origin 비교 화면]
  RG --> View
  TP --> View
  RP --> View
~~~

A는 두 번의 독립 추천 요청으로 비교하지 않고 B의 compare_local을 한 번 호출한다.
B가 as_of, feature_snapshot_id, 후보 집합, top_n을 고정하고 각 모델 핸들을 요청 종료까지 유지한다. 상품 재고/공개 catalog와 다른 개인정보가 중앙으로 가지 않는다.
두 global 모델은 사전에 짝지은 comparison 설정의 checkpoint를 사용한다. 요청 중 latest로 바꾸지 않는다. P는 같은 표의 G를 base로 사용한 개인화 결과여야 한다.
추천 순위와 상품을 나란히 보여주고 공통/개인화 상태, 기준 모델 버전, 비교 시점을 표시한다. 서로 다른 모델의 raw score를 확률이나 직접 비교 가능한 척도로 표시하지 않는다.

비교 결과는 판매자 내부 ComparisonResult로 A에 전달하여 서버에서 HTML로 렌더한다(CONTRACTS §4).
개인화 데이터 부족/검증 실패/아직 미실행이면 해당 P칸에 이유를 표시한다. G를 복사해 P라고 표시하지 않는다.
고객 이력 자체가 없는 경우의 일반 추천 fallback은 각 recommendation의 기존 표시를 유지한다. fallback 결과를 모델 품질 평가에 몰래 합치지 않는다.

## 6. 시연 시나리오

| 시나리오 | 통제 입력 | 보여줄 내용 |
| --- | --- | --- |
| 신규 판매자 | shared 학습/사전학습에 제외된 판매자, 별도 상품 목록, 고정 로컬 과거 이력 | T-G/R-G 점수화 → 같은 과거 prefix로 T-P/R-P |
| 이력 없는 판매자 | 카탈로그만, 구매 이력 없음 | 정직한 fallback. 개인화 성공 사례로 포장하지 않음 |
| 신상품 | 미구매 상품 등록, 공통 모델과 고객 이력 고정 | 관계 0 상태에서도 양쪽 후보에 포함 |
| 신상품에 거래 축적 | 사전 정한 동일 완료 구매 이벤트 replay | feature_epoch·관계/cache 변화와 두 모델의 추천 |
| FL 갱신 | 같은 실행 설정의 추가 round | base 버전 변경과 개인화의 재생성 |

시나리오/seed는 결과를 보고 유리한 사례만 고르지 않도록 사전 기록한다. 추천을 보고 사람이 서로 다른 구매를 선택하는 자유 사용은 별도 기능 시연이며 통제 비교가 아니다.
B는 원장 한 번의 반영을 두 모델 모두가 참조하게 한다. 비교 arm 수만큼 주문을 중복 생성/반영하지 않는다.
신상품의 첫 거래 이후는 warm-up 과정이다. 그 결과를 구매 이력 0 신상품 점수에 포함하지 않는다.
표현/순위가 매번 반드시 바뀐다는 성공 조건을 두지 않는다. 상태 반영과 유효 후보 점수화부터 확인한다.

## 7. 학습·특징·화면 변경의 분리
- 구매 완료: 로컬 특징 변경. shared weights가 바뀌었다는 뜻이 아니다.
- 개인화 완료: 해당 판매자의 query/scorer 복사본과 personalization_revision 변경.
- FL 완료: 집계된 base_model_version 변경.
세 동작의 실행 시각과 버전을 분리해 표시한다. 주문마다 FL나 개인화를 즉시 실행하지 않는다.

## 8. 담당과 완료
A: 동일 요청 비교 화면, unavailable/fallback 표시, 중앙 비잔류, 통제 시나리오 실행.
B: 두 variant 독립 학습, 2×2 평가, 개인화, snapshot 고정 compare_local, cache/버전 분리.
C: variant별 집계·release 분리, 개인화 가중치 미제출 확인, 통합 gate.
b2는 개인화 경계, g2는 비교 입력/화면, g3는 실제 FL 후 base/개인화 버전 전환까지 확인한다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](open-questions.md)를 확인한다.
