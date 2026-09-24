# B 작업 카드 — 데이터·NLP·추천
담당 이름: [작업 규칙](../working-agreement.md)에서 배정 · 실행 기준 D0017~D0019

## 목표와 소유
두 과거 데이터와 live 구매를 로컬 추천에 연결하고 신규 판매자/신상품을 점수화한다.
소유: commerce/packages/data_adapters/, commerce/packages/recommender/, commerce/evaluation/.
첫 읽기는 [팀 시작 안내](../start.md) §0·2를 따른다. 아래는 기능별 참고이며 전부 선독할 목록이 아니다: 입력은 [데이터](../../design/data.md), NLP·학습·개인화는 [모델 경계](../../design/model.md), 연결은 [인터페이스 계약](../../design/interfaces.md) §2~5, 학습/평가 데이터 분할과 측정은 [평가](../../design/evaluation.md).
지속 작업: [Git 협업](../git-workflow.md)에 따라 작업별 브랜치와 Draft PR을 사용한다. 시작 때 A의 입력 이벤트와 C의 모델 제출/설치 계약 변경을 확인한다.

별도 학습/서비스 인계는 [독립 모델 실험](../../design/model-lab.md)과 D0018을 따른다. 화면 없이 같은 모델 core로 학습·평가·checkpoint 재개·export를 제공한다. 시간 관계는 개별 간격의 MLP 뒤 pooling을 유지하고 이웃/시간쌍 예산을 feature fixture와 E-G0에서 확정한다.

## 첫 작업
아래는 피드백을 거쳐 나눠 수행할 초기 순서다. 먼저 팀 시작 안내의 B 첫 작업부터 확인하고 맡은 범위 안에서 이어간다.

1. 소형 두 출처 fixture와 live catalog/purchase fixture를 공통 로컬 표현으로 변환한다. 시간 종류와 미관측 수량을 보존한다.
2. 한국어·영어·규격·결측을 보존하는 text builder를 만든다. 기존 NLP 완성 모듈은 없다.
3. 정확한 인코더 ID/revision·라이선스·tokenizer·pooling·차원·길이를 선택하고 샘플로 검증한다.
4. frozen wrapper·배치 벡터화·artifact hash·z cache를 구현한다. 임의 신규 special token은 추가하지 않는다.
5. E-G0 텍스트만 기준선과 시간/메모리/절단률을 먼저 측정한다.

첫 산출물: b1, NLP 선택 기록, b2의 freeze/해시 부분, 로컬 기준선.
384차원·32토큰은 후보값이다. 실제 확정 전 manifest의 fixture 숫자를 실모델 크기로 복사하지 않는다.

## A에게 제공
seller별 open_runtime와 ingest_purchase_event / upsert_catalog_item / predict_local / compare_local.
별도 features.sqlite를 소유한다. 이벤트 중복 확인·반영·feature_epoch를 원자 처리하고 성공 후 반환한다. 카탈로그 source_seq 역순을 막는다.
입력 변경에 따른 cache 무효화와 신상품 관계 0 분기를 구현한다.
하나의 runtime/원장을 두 variant가 읽는다. compare_local에서 이력·catalog snapshot·후보·checkpoint 핸들을 한 번 고정하고 네 결과와 불가 사유를 반환한다. A가 별도 모델 파이프라인을 만들게 하지 않는다.

## C에게 제공
get_shared_manifest / export_shared_state / get_local_data_ref / train_round / install_release.
실제 key·shape의 manifest를 생성한다. frozen·상품/고객축·optimizer는 export에서 제외한다.
train 복사본과 서빙 모델을 분리하고 해시 검증 뒤 객체·버전·cache를 함께 설치한다. 학습 실패 시 기존 모델을 유지한다.
variant별 manifest와 공통 base를 제공한다. export_shared_state는 설치된 공통 base만 반환하며, 라운드 학습 결과는 TrainingResult로만 넘긴다. 개인화 서빙 가중치를 export하거나 FL 초기값에 섞지 않는다.
local_data_ref의 경로를 C coordinator에 보내지 않는다. per-client loss는 보호 경로 밖에 노출하지 않는다.

## 이후 구현과 완료
관계 특징 → basket/sequence → 공통 scorer → 실제 로컬 학습 → shared export/load → g2/g3.
b2: gradient·frozen 불변·실제 export·고정 val split·버전 교체 실패 복구.
음성 샘플에서 target basket 전체를 제외한다. 작은 카탈로그의 음성 부족·빈 데이터도 처리한다.

G4 전에는 실데이터 로컬 분석/학습까지 가능하다. 중앙 FL 평가 R1/R2는 보호 경로 뒤 진행한다.
신규 판매자 A-0와 이력조차 없는 fallback을 구분한다. 신상품 C-new와 관계만 가리는 C0도 구분한다.

## 개인화와 비교
[모델 비교](../../design/comparison.md)/[평가](../../design/evaluation.md)/[모델 경계](../../design/model.md) §8을 따른다. text_only와 text_relation을 같은 split·시퀀스·후보·학습 예산으로 각각 학습한다. 관계 모델의 입력을 나중에 0으로 가린 결과는 텍스트 기준선을 대신하지 않는다.
전체 추천 가중치 FL 후 각 base의 복사본에서 query_proj/scorer만 개인화한다. lr/최대 step/최소 데이터/검증 규칙은 E-G0와 validation으로 정해 양쪽에 적용한다. 기존 학습 작업 큐에서 FL와 개인화를 직렬 처리한다.
개인화 전후 공통 base 해시 불변, 두 그룹만 gradient 허용, 새 base에 옛 개인화 부착 거부, 데이터 부족/검증 실패 반환을 b2에서 확인한다. global과 personal 저장 경로·식별자를 분리한다.
화면 없이 T-G/R-G/T-P/R-P의 오프라인 성능·가용률·계산 비용을 평가하고, 같은 checkpoint 쌍을 서비스에 인계한다. FL 효과 주장에는 같은 variant의 local_only 비교를 별도로 둔다.

## 부하와 인계
원자료 분석, NLP, 모델, 평가를 동시에 완성하려 하지 않는다. E-G0 뒤 예산을 측정한다. A에 합성 입력 fixture, C에 결과 저장·재현 실행 도구를 인계할 수 있다.


공통 시작점: [팀 시작 안내](../start.md). 최초에 담당 카드와 해당 계약을 읽고, 이어갈 때는 관련 diff·검사·남은 작업만 확인한다.
