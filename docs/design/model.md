# 모델 경계 — NLP, 학습, 서빙

갱신: 2026-09-24 · D0017~D0021 · 소유: B

## 1. 모델의 네 종류 상태

| 종류 | 내용 | FL 처리 |
| --- | --- | --- |
| frozen_text | 고정 사전학습 텍스트 모델·tokenizer | 동일 artifact 설치. 집계 제외 |
| shared_base | 해당 variant의 전체 추천 가중치 | FL 학습 복사본의 업데이트만 보호 집계 |
| personalized | 공통 base에서 복사해 파인튜닝한 query_proj·scorer와 optimizer | 판매자에만 저장, FL 제출 제외 |
| 파생 상태 | 고객 이력, 관계, z/l/e/h, 캐시 | 판매자에만 저장 |

초기 버전의 상품별 학습 bias는 사용하지 않는다. “관계를 로컬에서 계산한다”와 “관계 MLP 가중치가 판매자 전용이다”는 다른 뜻이다.

기본안은 **전체 추천 가중치 FL → 후반부 로컬 파인튜닝**이다(D0019). 모든 forward/backward는 판매자 안에서 실행한다. 관계값과 상품 표현은 로컬에 남기고, 그 관계를 해석하는 MLP 가중치는 공통 FL로 학습한다. 별도 adapter나 판매자 전용 관계 MLP는 첫 기본안에 두지 않는다.

## 2. NLP는 B가 새로 구현할 작업이다

사전학습 가중치를 사용해도 텍스트 생성·토큰화·배치 실행·캐시 코드는 아직 없다. NLP를 완성된 외부 모듈로 가정하지 않는다.

| 순서 | 산출물 | 완료 조건 |
| --- | --- | --- |
| 1 | Dunnhumby·Instacart·live 텍스트 builder | [데이터](data.md) §3 입력 규칙, 한국어·약어·규격·결측 fixture |
| 2 | 인코더 선정 기록 | 정확한 model ID, 고정 revision, 라이선스, tokenizer, pooling, 차원·길이 |
| 3 | frozen wrapper | padding 제외 mean pooling 등 모델 고유 규칙, eval 모드·gradient 차단, 유한 벡터 |
| 4 | z 캐시와 설치본 | 모델·설정 해시, 동일 입력 재현, 원문 변경 시 재계산 |
| 5 | 텍스트 품질 보고 | 빈 입력·unknown·정규화/토큰 잘림 충돌, 규격 보존, 처리 시간·메모리 |
| 6 | 텍스트만 추천 기준선 | E-G0, 관계 0인 상품도 후보 점수 생성 |

지원 입력은 영어 데이터와 한국어 live 상품이다. 비ASCII 제거는 금지한다. 384차원·32토큰·MiniLM급은 이전 후보값이며 정확한 모델을 고른 것이 아니다. B는 한국어·영어 샘플을 실제 인코더로 확인하고 모델·길이를 고정한다. 바뀐 d_text는 fusion·relation_mlp 입력 차원과 manifest에 반영한다.

마커는 기본적으로 일반 문자열로 토큰화한다. 임의 special token을 추가한 뒤 미학습 embedding을 frozen 상태로 쓰지 않는다. 서로 다른 상품명이 원문에서 고유해도 토큰 절단 후 같은 입력이 될 수 있다.

text_artifact_hash = 파일별 SHA-256 목록과 인코더 설정(model ID/revision, tokenizer 설정, pooling, max_length, 출력 정규화)의 정규 JSON에 대한 SHA-256. 파일 목록은 상대 경로순 정렬한다. tokenizer와 가중치·config를 함께 식별한다. preprocessing_version은 텍스트 builder·관계 정의·고정 상수의 artifact 해시다. 판매자 이벤트로 fit한 사전·통계를 포함하지 않는다.

## 3. 공통 함수와 입력 경로

| 공유 그룹 | 역할 |
| --- | --- |
| shared.time_mlp | 방향별 시간 간격·관측/검열 비트 |
| shared.relation_mlp | 고객·장바구니 강도, support, 이웃 텍스트 해석 |
| shared.relation_pool | 이웃 집합을 관계 표현으로 집계 |
| shared.fusion | 텍스트 z와 관계 l 결합 |
| shared.basket_encoder | 방문 내 상품 집합 표현 |
| shared.seq_time_pos | 방문 순서·시간 간격 |
| shared.sequence | 방문 시퀀스 처리 |
| shared.query_proj | 고객 query |
| shared.scorer | 로컬 후보 상품 공통 점수 |

위 9개는 text_relation의 그룹이다. text_only는 time_mlp/relation_mlp/relation_pool을 제외한 6개를 사용하며 l=0인 Fusion을 처음부터 학습한다([모델 비교](comparison.md) §2). 이름은 그룹이며 실제 export는 shared.relation_mlp_l0_w처럼 평탄한 층별 키다. 텐서 key·shape·dtype은 variant별 B manifest로 고정하고 C는 해당 목록만 검증한다. 기본 shared dtype은 float32, 정규화는 LayerNorm. 고정 상품 ID별 출력행·공유 상품 ID embedding table은 쓰지 않는다.

    z[c] = frozen_text(build_product_text(c))
    R[c,j] = local_relations(c,j, snapshot)
    l[c] = 0 if no observed neighbors else rho(mean(phi(R[c,j], z[j])))
    e[c] = fusion(z[c], l[c])
    h[u] = sequence(basket(e[history[u]]), time, masks)
    score[u,c] = scorer(query(h[u]), e[c])

이력 0은 명시적 분기로 l=0을 만든다. MLP(0)=0이라고 가정하지 않는다. 후보는 활성 로컬 카탈로그이며 구매 0인 상품을 제외하지 않는다. 각 고객의 h는 요청 cutoff마다 계산한다.

X[u,c]와 B[b,c]는 구매 여부의 이진 행렬이다. 공동구매 강도는 R_ij/sqrt(R_ii R_jj), support는 log1p로 분리한다. 분모 0이면 관측 없음. 수량은 초기 모델 입력에서 제외한다.

시간 관계는 채택한 설계대로 q[c→j]=mean_d(time_mlp(log1p(min(d,30)), 관측/검열 비트))다. 반대 방향을 별도로 만든다. 같은 basket의 동시 구매는 이 시간쌍에서 제외하고 basket 관계로 표현한다. 간격을 먼저 평균/중앙값 하나로 줄여 MLP에 넣는 방식은 동일하지 않으며 기본안으로 묵시 대체하지 않는다. Instacart의 실제 경과일 하한이라는 [데이터](data.md)의 제한도 유지한다.

기존 sim_bask 단독 top-K는 동시구매가 적은 고객/시간 관계를 잘라낼 위험이 있어 최종 고정값에서 해제한다(D0018). B는 고객·바스켓·시간 각각의 후보를 합치는 제한 이웃 방식과 시간쌍 추출/샘플링 규칙을 세 관계가 서로 다른 이웃을 선택하는 작은 합성 예제로 확인한 뒤 관계 특징의 첫 소형 실행에서 예산을 확정한다(E-G0는 텍스트만 쓴다). 최근접 후속 방문/전 조합 등의 시간쌍 정의를 서로 다른 코드에서 임의로 선택하지 않는다. 확정 규칙·seed·K는 preprocessing_version에 반영한다.

## 4. A·B·C의 로컬 API

초기 뼈대 PR #3에서 `commerce.packages.contracts.ports`/`types`/`errors`와 `commerce.packages.recommender.runtime.open_runtime`을 추가했다. 모든 runtime API는 동기 함수다. A의 merchant lifespan이 runtime 하나와 jobs를 만들고 같은 객체를 C client에 주입한다. C start/stop은 async다. 장시간 학습·개인화는 판매자당 단일 jobs 실행기를 사용한다. `docs/development.md`에 호출 경로가 있으며 실모델·영속 반영은 아직 미구현이다.

B는 seller별 RecommenderRuntime 객체를 제공한다. A와 C는 같은 판매자 객체에 연결하고 다른 판매자 상태를 전역 singleton으로 섞지 않는다.

메서드 시그니처와 반환 타입은 `commerce/packages/contracts/ports.py`의 `RecommenderRuntime`과 `types.py`가 기준이다. 아래는 코드에 없는 의미다.

ingest 정상 반환은 영속 반영 완료를 뜻한다. 이미 같은 event ID·본문이면 성공으로 반환하고, 다른 본문이면 DUPLICATE_EVENT다. B는 이벤트 기록·반영 ID·feature_epoch를 같은 특징 DB 트랜잭션으로 갱신한다. catalog의 source_seq는 A의 로컬 outbox 순번이며 같은 상품의 오래된 갱신이 새 값을 덮지 않도록 저장한다. source_seq는 catalog JSON에 임의 추가하지 않는다.

A는 DB 커밋 후 전달을 시도하고 실패하면 pending을 유지한다([인터페이스 계약](interfaces.md) §2). B의 저장소 생성·DDL은 B만 수정하며 A DB에 캐시 테이블을 추가하지 않는다.

local_data_ref는 seller와 특징 스냅샷을 식별하는 opaque 값이다. C는 파싱하거나 경로로 사용하지 않는다.

model_variant는 text_only/text_relation, mode는 global/personalized/auto다. 이들은 판매자 내부 함수 인자이며 기존 recommendation_request JSON에 추가하지 않는다. mode=global은 공통 base만, personalized는 그 base에 대응하는 승인된 tail만 사용한다. 명시 personalized가 없으면 NOT_FOUND에 해당하는 로컬 오류로 거부한다. auto는 현재 base의 승인된 개인화가 있으면 사용하고 없으면 base를 쓴다. 이 선택 자체를 no_customer_history fallback으로 표시하지 않는다.

한 seller runtime이 두 variant를 관리하되 A 이벤트는 B 특징 저장소에 한 번만 반영한다. 두 variant가 동일한 원장/feature_epoch를 읽고 모델·파생 cache는 분리한다. compare_local은 snapshot과 각 모델 핸들을 한 번에 고정한다. 반환 객체와 unavailable 표시는 [인터페이스 계약](interfaces.md) §4, [모델 비교](comparison.md) §5를 따른다.

TrainingResult는 shared_delta(dict[str, numpy.ndarray]), metrics(dict), completed(bool)를 가진 B의 로컬 반환 객체다. C의 판매자 FL client가 completed를 delta_manifest에 옮긴다. 수행할 학습 예제가 없으면 같은 manifest shape의 0 delta와 completed=false를 반환하며 집계하지 않는다. 이는 HTTP JSON 계약을 새로 늘리는 것이 아니다.

## 5. 학습·서빙 동시성

- train_round는 round_config가 지정한 **공통 base의 학습용 복사본**과 시작 시점의 특징 스냅샷을 사용한다. 개인화된 서빙 tail을 시작값으로 사용하지 않는다. 지정 base가 없으면 다른 버전으로 대체하지 않고 학습을 거부한다.
- shared_delta는 학습 후 가중치에서 시작 가중치를 뺀 값이다. 초기 구현에서는 추가 로컬 step 재스케일을 하지 않는다.
- install_release는 key·shape·dtype·유한값·manifest/weights 해시를 확인한 새 서빙 객체를 준비하고 객체·버전·캐시 핸들을 한 번에 교체한다.
- 실패·시간 초과·집계 폐기 시 기존 서빙 버전을 유지한다. 요청 하나는 시작할 때 잡은 단일 서빙 버전으로 끝난다.
- 새 거래는 최신 feature_epoch에 반영한다. 진행 중 학습은 시작 스냅샷을 유지하고 다음 라운드부터 새 거래를 쓴다.
- 한 seller의 학습은 동시에 한 작업만 실행한다. C가 중복 라운드를 막고 B도 재진입을 거부한다.

- FL와 개인화 작업도 seller 단위 작업 큐로 직렬화한다. 두 variant의 실험은 첫 구현에서 차례로 수행해 메모리와 충돌을 줄인다.
- export_shared_state는 해당 variant의 배포된 공통 base만 반환한다. 학습 복사본 delta는 TrainingResult로만 반환하고 개인화 tail은 어느 FL export 경로에도 포함하지 않는다.
- install_release는 base를 갱신하고 새 base의 공통 추천을 원자 활성화한다. 기존 개인화는 새 base와 호환되지 않는 상태로 표시하고 적용하지 않는다. 이후 새 base에서 다시 개인화한다. 검증에 실패한 release는 이전 서빙을 유지한다.
- 이미 설치된 동일 버전·해시의 재전달은 멱등 성공이며 유효한 개인화를 초기화하지 않는다. 같은 버전에 다른 해시가 오면 거부한다.

## 6. 학습 기본값과 정확성

초기 후보: d_model=64, Transformer 2층·4 heads, FFN 256, MLP hidden 128·GELU, L=10, 방문당 최대 32개. 이 값과 dropout·pooling·정렬 규칙을 모델 config에 기록한다. 32 초과 바스켓의 절단률을 상품 단위로 측정하고 정답 집합은 자르지 않는다. 카테고리 개수 통계로 상품 절단률을 추정하지 않는다.

target 방문의 상품 집합에서 양성 1개를 고르고 배치 공유 음성 N_neg(후보 200)를 점수화한다. **예제별로 정답 집합 전체를 음성에서 제외**한다. 중복 음성은 한 번만 센다. 카탈로그가 작으면 가능한 음성 수로 줄이고, 음성이 없으면 해당 예제를 건너뛰어 기록한다. 표본 수는 중앙에 보내지 않는다.

local_steps=40, batch_size=64, max_local_epochs=6은 시작 후보다. 실제 step은 min(local_steps, ceil(n_train/batch_size) × max_local_epochs); max_local_epochs=0은 상한 없음, local_steps=0은 학습 없음이다. 부분 배치는 실제 크기로 계산하므로 소비 예제 수가 항상 2,560이라고 하지 않는다. 빈 학습셋이면 completed=false이고 제출 집계에 넣지 않는다.

optimizer는 라운드마다 새 AdamW, 초기 lr=0.001·weight_decay=0.01·clip norm=1. 실제 채택 config를 기록한다. 검증 고객 9:1 분할은 train 안에서 한 번 만들고 seller·run seed로 고정한다. **val_split seed에는 round_id를 넣지 않는다.** 너무 적어 유효 검증셋이 없으면 loss_mean=null, 조기 종료 판단에서 제외하고 사유는 로컬에 남긴다.

metrics = loss_mean(고정 로컬 검증 손실 또는 null), grad_norm_mean(유한값 또는 null). seller·round·completed는 delta_manifest가 소유한다. 합성 모드의 전송은 round_submission.v1, 최종 모드는 보호 집계 안에서 합산한다.

후보 e에는 gradient가 흐른다. CPU 예산 때문에 이력 e를 detached cache로 사용하면 그 근사와 갱신 주기를 모델 config에 기록한다. 관계 마스킹 0/10/50/100%는 학습 증강이며 신상품 평가 정의와 구분한다.

## 7. 캐시와 검증

실험 학습과 서비스는 같은 core 모델·특징 코드를 사용한다. 초기 checkpoint·불변 architecture config·release 설치와 서빙 동일성 검사는 model-lab.md를 따른다. NLP만 사전학습 가중치로 시작하며 관계/sequence/scorer의 프로젝트 가중치는 별도 실험에서 학습한다.

z key: text_artifact_hash + preprocessing_version + 정규 텍스트 SHA-256.
l/e key: model_variant + base_model_version + preprocessing_version + feature_snapshot_id + item_id. 마스킹 실행은 별도 feature_snapshot_id를 갖는다. 개인화는 query/scorer만 변경하므로 base의 l/e를 재사용할 수 있다. 최종 점수/추천 cache를 도입하면 personalization_revision과 요청 cutoff/후보도 key에 포함한다.
이는 불변 base의 추론 cache 규칙이다. 전체 FL 학습 중에는 매 optimizer step마다 임베딩 함수도 바뀌므로 base cache를 현재 학습 결과로 재사용하지 않는다. 후보 e는 현재 학습 복사본으로 계산해 gradient를 유지한다. 선택적인 이력 detached cache는 §6의 근사 설정·갱신 주기를 따르고 서빙 cache와 분리한다.
h는 요청별 계산. 모든 캐시와 반영 ID는 B 특징 저장소에 둔다.

카탈로그 변경과 완료 구매는 feature_epoch를 올린다. 첫 구현은 l/e 전체 무효화로 정확성을 확인하고, 이후 의존 이웃을 정확히 계산할 수 있을 때 부분 무효화를 도입한다. 상품 텍스트 수정은 z와 그 텍스트를 소비하는 관계 표현을 다시 만든다.

b2는 frozen 해시 불변, 공유 층 gradient, export에 local/frozen/상품축/고객축 없음, 버전 교체 실패 시 기존 추천 유지, 검증셋 불변을 확인한다. 신상품 구매 후 feature_epoch 증가와 관계를 실제로 만드는 입력에서 e 갱신을 확인한다.

## 8. 후반부 개인화와 버전

개인화는 [작업 규칙](../team/working-agreement.md) §8의 줄이는 순서 4번이다. 아래 불변식만 고정한다. 설정 항목, ID 형식, 저장 경로는 B가 구현할 때 정하고 PR에 적는다. 반환 객체는 `types.PersonalizationResult`가 기준이다.

1. 해당 variant의 공통 base 복사본에서 시작하고 query_proj·scorer만 학습한다. 나머지는 freeze/eval로 둔다. 개인화 설정은 validation에서 정해 두 variant에 똑같이 적용한다.
2. 공통 학습과 같은 음성·정답·보류 규칙을 쓰고, 검증/test label을 학습에 쓰지 않는다.
3. 결과는 판매자 로컬에만 둔다. 어떤 FL export나 중앙 release에도 넣지 않고, 공통 가중치를 덮어쓰지 않는다.
4. 개인화 결과는 만든 base에만 붙인다. base가 바뀌면 옛 결과를 쓰지 않고 새 base에서 다시 만든다.
5. 데이터가 없거나 검증을 통과하지 못하면 공통 base로 서비스하고, 비교에서 P칸을 G 결과로 채우지 않는다.

recommendation.model_version은 실제 서빙 가중치의 opaque ID다. C의 round_config.model_version은 항상 공통 base ID다. A는 서빙 ID를 해석해 모델 URL을 만들지 않는다.

b2에서 개인화 전후 공통 base 해시 불변, 두 그룹 밖 가중치 불변, export에 개인화 값 없음, 다른 base와의 결합 거부를 확인한다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](open-questions.md)를 확인한다.
