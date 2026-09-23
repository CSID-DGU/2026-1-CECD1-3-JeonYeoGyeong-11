# 인터페이스 계약
갱신: 2026-09-23 · D0017~D0019 · 소유: C, 생산자/소비자 공동 검토

## 1. 공통 기준과 계약 목록

JSON 필드·타입은 commerce/packages/contracts/schemas/의 같은 이름 파일이 기준이다. 동작은 이 문서와 [모델 경계](model.md)가 기준이다. 문서와 schema가 다르면 함께 수정한다. 아래는 구현할 계약이며 서버 구현 완료를 뜻하지 않는다.

| 계약 | 경계 | 저장·전송 범위 |
| --- | --- | --- |
| commerce_order.v1 | A 주문 상태 | 판매자 로컬 |
| purchase_event.v1 | A → B 완료 구매 | 판매자 로컬 |
| catalog_item.v1 | A → B 상품 정보 | 판매자 로컬 |
| catalog_snapshot.v1 | A → 중앙 공개 상품 | 공개 허용 필드만 |
| recommendation_request.v1 / recommendation.v1 | A ↔ B 추천 | 판매자 로컬 |
| contract_error.v1 | 공통 거부 응답 | 원문·비밀을 detail에 넣지 않음 |
| shared_model_manifest.v1 | B → C 모델 구조 | 상품·고객축 없는 공통 구조 |
| round_config.v1 | C → 판매자 학습 설정 | 기준 모델과 라운드 |
| delta_manifest.v1 | B → 판매자 FL client | 업데이트의 구조·완료 여부 |
| round_submission.v1 | 판매자 → C | 합성 평문 모드의 manifest·metrics·바이트 설명 |
| model_release.v1 | C → 판매자 | 설치할 집계 모델 식별자·무결성 |
| round_submit_ack.v1 | C → 판매자 | 합성 모드 제출 진행·처분 |

보호 집계의 전송 메시지는 C가 프로토콜을 선정한 뒤 별도 schema로 확정한다. round_submission을 암호화 없이 실거래 유래 업데이트에 사용하는 것은 금지한다.

- UTF-8 JSON, Unicode NFC, 모든 객체 additionalProperties=false. 필드 순서는 의미가 없다.
- ID는 해당 schema의 길이·패턴을 따른다. 고객·주문 ID를 중앙으로 보내지 않는다. 판매자 로컬에서도 고객 ID를 URL·로그에 넣지 않는다.
- 절대 시각은 유효한 RFC3339 UTC Z. schema의 정규식은 달력 유효성까지 확인하지 않으므로 런타임에서 파싱한다.
- 정규 JSON: json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)의 UTF-8. manifest_hash 계산은 manifest_hash 키를 뺀 객체를 사용한다.
- 숫자는 유한값. NaN·Infinity 금지. ID 공간은 출처와 판매자별로 분리한다.
- JSON 기본값은 자동 삽입되지 않는다. required인 nullable 필드도 명시한다.
- 인증으로 확인한 seller/customer와 본문을 대조한다. 본문 ID만 믿고 권한을 주지 않는다.
- 중앙 공개 상품 ID·상품명은 허용된다. 거래에 등장한 상품 목록·구매자·수량·손실은 공개 카탈로그와 다르다.

## 2. A → B 주문 이벤트와 복구

### 주문 상태

requested → accepted → completed. requested/accepted → cancelled. completed/cancelled은 종결 상태다. 완료 후 취소·환불·학습 철회는 첫 범위에 없다. 주문마다 status_version은 1부터 시작하며 전이에 성공했을 때만 1 증가한다.

A는 구매자의 입력을 서버에서 검증하고 가격·상태·완료시각을 결정한다. commerce_order는 서버가 만든 기록이며 클라이언트가 completed를 직접 지정하는 요청 형식이 아니다. 구매·판매 화면의 폼/API DTO는 A가 소유한다.

같은 seller/customer/idempotency_key로 같은 정규 주문 입력을 재시도하면 원 주문을 반환한다. 다른 입력이면 DUPLICATE_IDEMPOTENCY_KEY. 낡은 expected_status_version은 STALE_STATUS_VERSION. 동시 완료는 조건부 갱신으로 1회만 성공한다.

### 내구성 있는 전달

1. A의 한 DB transaction에서 주문 완료, purchase_event, outbox pending을 함께 기록한다.
2. transaction commit 뒤 B의 ingest_purchase_event를 호출한다.
3. B는 이벤트 본문·반영 ID·특징 변경·feature_epoch를 한 특징 DB transaction으로 저장한다.
4. B가 정상 반환하면 A가 delivered 처리한다. 중간 종료·예외 시 pending을 재시도한다.
5. 같은 ID·동일 본문 재전달은 성공. 같은 ID·다른 본문은 DUPLICATE_EVENT로 격리하고 재시도로 덮지 않는다.

구매 성공은 A commit 기준이다. B 전달 대기는 주문 실패로 바꾸지 않고 로컬 상태로 표시한다. 시작 시와 주기적으로 pending을 재생한다. A와 B의 DB를 직접 교차 쓰지 않는다.

purchase_event_id = SHA-256(정규 JSON [seller_id, source, basket_id_local])의 앞 32자리. 이벤트는 immutable이며 한 완료 주문에 하나다. basket_id_local=live order_id. 상품 중복은 주문 단계에서 수량을 합친다. 기존 이벤트를 원장으로 재생할 때도 같은 ID를 쓴다.

| 출처 | seller_partition | time / order_rank | 수량 |
| --- | --- | --- | --- |
| live | platform_seller | absolute 완료시각 / null | 실제 주문 정수 |
| dunnhumby | simulated_independent_store | absolute / null | 관측된 양의 정수 |
| instacart | synthetic_partition | relative_day 누적간격 / order_number | null, 1로 꾸미지 않음 |

학습에는 quantity_observed를 초기 입력으로 쓰지 않는다. 이벤트에 가격·주소·결제 정보는 넣지 않는다. 취소 주문에서는 구매 이벤트를 만들지 않는다.

## 3. 카탈로그와 공개 범위

A는 catalog_item을 상품 생성/수정과 함께 outbox에 기록한다. B의 upsert_catalog_item(item, source_seq)에 전달한다. source_seq는 판매자 outbox의 증가 정수이며 함수 인자다. JSON에 없는 필드로 덧붙이지 않는다. B는 상품별 최신 순번을 저장하여 역순 전달을 무시한다. 같은 순번·다른 본문은 충돌로 거부한다.

공개 catalog_snapshot은 seller_id, snapshot_at 및 items의 item_id_local/title_text/description_text/listing_status만 포함한다. 구매 횟수·재고·최초 판매 시점·고객 통계·관계·벡터를 넣지 않는다. 중앙은 인증된 판매자의 최신 스냅샷으로 교체하며 오래된 시각의 갱신을 거부한다.

상품 inactive는 추천 후보에서 제외하되 과거 이력을 삭제하지 않는다. 상세와 주문 가능 여부는 판매자에서 최종 확인한다. 첫 플랫폼은 판매자별 장바구니·주문이며 여러 판매자의 주문을 중앙에서 합치지 않는다.

## 4. 추천 호출

A가 로그인 세션의 seller/customer, 현재 UTC as_of로 runtime.predict_local을 호출한다. B의 API와 DB 위치는 [모델 경계](model.md) §4를 따른다.

- candidate_item_ids=null: as_of에 사용 가능한 활성 로컬 카탈로그. []: 빈 후보.
- 명시 후보 중 다른 판매자/미등록 ID는 NOT_FOUND, 비활성 상품은 제거한다.
- top_n=null: 기본 10. 후보가 적으면 있는 만큼만 반환한다.
- 라이브 history는 completed_at < as_of. 같은 시각의 이벤트는 다음 요청에서 포함될 수 있다.
- 오프라인 평가는 상품 동시 구매 누출을 막는 이벤트 순서 키를 사용한다([평가](evaluation.md) §2). 상대시간 Instacart에 가짜 UTC를 붙여 이 HTTP 요청으로 처리하지 않는다.
- 응답은 seller/customer/as_of를 반향하며 score 내림차순, 동점 item_id_local 사전순. 중복 상품 금지.
- 현재 score_semantics=next_purchase, horizon_days=null. 점수를 확률이라고 표시하지 않는다.
- 모델 없음 → no_shared_model, 판매자 이력 없음 → no_seller_history, 고객 이력 없음 → no_customer_history 순서로 fallback_reason을 결정한다. fallback이면 is_cold_start=true. 공통/개인화 가중치로 정상 시퀀스 추론을 하면 false/null.
- fallback은 활성 카탈로그의 로컬 인기순, 동점 ID순. 이력 없는 판매자는 ID순으로 표시하고 fallback임을 숨기지 않는다. 텍스트 기준선은 고객 이력이 있을 때 정의한다.
- 신규 판매자라는 이유만으로 항상 fallback하지 않는다. 공유 모델과 로컬 이력이 있으면 가중치 개인화 전에도 실제 시퀀스 모델로 평가한다.

B는 contract_error에 대응하는 코드·field_path를 가진 예외를 제공한다. A는 이를 HTTP 오류로 매핑한다. stack trace와 원장 내용을 응답하지 않는다.

### 4.1 모델 선택·개인화·비교의 로컬 경계

[모델 경계](model.md) §4의 model_variant/mode는 A·B·C의 판매자 내부 함수 인자다. recommendation_request.v1에 variant, mode, feature_snapshot_id를 추가하지 않는다. 첫 비교 화면은 판매자 서버에서 렌더하여 별도 브라우저 JSON 응답 계약을 만들지 않는다.

recommendation.model_version은 실제 서빙 가중치 ID다. 공통 모델은 base release ID, 개인화는 B의 로컬 ps- ID다. 이 문자열의 64자 상한은 유지한다. base/revision 해석은 B의 로컬 인덱스가 담당하며 중앙에는 개인화 ID/가중치/손실을 제출하지 않는다. C의 round_config/delta_manifest/model_release의 model_version은 계속 공통 base 버전이다.

compare_local(request)는 다음 **로컬 Python 객체** ComparisonResult를 반환한다. A는 속성으로 읽으며 이 객체를 JSON 계약으로 외부 직렬화하지 않는다.

| 항목 | 의미 |
| --- | --- |
| comparison_id | 판매자 로컬 비교 식별자 |
| as_of, feature_snapshot_id | 네 결과가 함께 참조한 시점과 immutable 특징 snapshot |
| candidate_set_hash | 중복 제거 후 상품 ID 문자열 오름차순으로 정렬한 후보 목록의 정규 JSON SHA-256 |
| arms | T-G/R-G/T-P/R-P 순서의 4개 결과 |
| 각 arm의 arm_id, model_variant, mode | 결과 ID, text_only/text_relation, global/personalized |
| 각 arm의 available, unavailable_reason | 준비 여부와 고정된 비민감 사유. 가능하면 사유는 null |
| 각 arm의 base_model_version, personalization_revision | base는 비교 설정에 지정한 ID(미설정이면 null). G와 미준비 P의 revision은 null. 준비된 P는 같은 variant G의 base와 일치 |
| 각 arm의 recommendation | available이면 기존 recommendation.v1, 아니면 null |

unavailable_reason은 model_not_ready / personalization_not_ready / insufficient_data / validation_rejected / base_mismatch 중 하나다. P칸에 G 결과를 대신 넣지 않는다. 권한·요청 오류는 비교 자체를 거부하며 unavailable로 숨기지 않는다. 추천 자체의 이력 부족 fallback은 recommendation의 기존 필드로 표시한다.

B의 seller runtime이 원장·catalog snapshot과 후보를 한 번 고정한다. A가 별도 요청 4개를 호출하거나 비교 중 구매를 arm별로 따로 반영하지 않는다. 준비된 comparison 설정은 variant별 base checkpoint를 지정하고 요청 중 latest를 따라가지 않는다. 상세 시나리오와 판정은 [모델 비교](comparison.md)을 따른다.

## 5. B ↔ C 모델과 파일

shared_model_manifest는 architecture_version, task_kind, preprocessing_version, text_artifact_hash, tensors와 manifest_hash를 고정한다. B가 생성하고 C가 전송·배포를 검증한다. 현재 프로젝트의 shared tensor는 모두 float32다. schema에 더 넓은 dtype이 있어도 이 구현 경로는 float32만 허용한다.

round_config.model_version은 **라운드 시작 기준 모델**이다. 집계 성공 후 새 model_version을 발급한다. 같은 버전의 가중치를 나중에 바꾸지 않는다. round_config/manifest의 architecture/hash가 다르면 학습을 시작하지 않는다. 스텝·배치 의미는 [모델 경계](model.md) §6을 따른다.

D0019의 variant는 불변 architecture config에 기록하고 manifest의 기존 architecture_version으로 식별한다. manifest에 model_variant 필드를 새로 추가한다는 뜻이 아니다. coordinator 한 실행은 하나의 model_variant만 집계하며 별도 registry/round/auth 설정을 사용한다. 첫 구현은 두 실험을 순차 실행하여 두 공통 release를 준비한다. 하나의 latest 포인터를 두 variant가 공유하지 않는다. 중앙 JSON에 임의 experiment 필드를 덧붙이지 않는다. C는 실행 설정의 variant를 B의 로컬 함수 인자로 전달하고 manifest 일치를 검증한다.

FL 제출은 해당 variant의 공통 base에서 시작한 전체 추천 가중치 delta다. 개인화된 query/scorer·개인화 optimizer는 제출 대상이 아니다. FL client가 현재 서빙 객체를 직접 state_dict로 읽어 제출하지 않고 B의 train_round/export 경계를 사용한다.

model_release는 model_version, manifest_hash, weights_sha256, weights_size_bytes를 제공한다. 수신자가 임의 URL을 따라가지 않도록 가중치/manifest는 같은 coordinator의 고정 경로에서 받는다. C는 descriptor를 마지막에 공개하고 부분 업로드 모델은 latest에 노출하지 않는다.

합성 round_submission은 delta_manifest, aggregate_metrics(loss_mean, grad_norm_mean; 각각 유한한 비음수 또는 null), payload_sha256, payload_nbytes를 갖는다. metrics 이름은 집계 입력이라는 뜻이며 **이 봉투 자체는 개별 지표를 숨기지 못한다**. 보호 모드에서는 쓰지 않는다.

가중치·delta는 numpy savez의 npz, media type application/octet-stream. float32 이름·shape·총 element 수를 manifest와 비교하며 pickle/object array는 허용하지 않는다.

- 전송 최대 8 MiB, payload_nbytes와 실제 바이트 길이·SHA-256 일치. 본문을 전부 읽은 뒤뿐 아니라 수신 중 누적 바이트에도 한도를 적용한다.
- npz의 헤더/ZIP 메타데이터를 포함한 전송 길이와 sum(prod(shape) × 4)의 텐서 길이는 다르다.
- 압축 해제 전 항목 수·총 비압축 크기를 제한하고, 후에는 중복/추가/누락 키·shape·dtype·NaN·Inf를 거부한다.
- 텐서 예산을 초과하면 C/B가 manifest와 제한을 함께 재검토한다. 제한을 끄고 진행하지 않는다.
- fixture의 임의 해시값은 형태 검증용이다. 실제 해시·인증·바이트 검증은 c1/g3에서 수행한다.

## 6. 합성 FL HTTP 경로와 상태

모든 모델/라운드 경로는 판매자별 Bearer 인증을 적용한다. seller는 토큰에서 확인하고 body와 대조한다. 요청 본문·Authorization·원시 텐서 로깅을 끈다. /healthz만 비민감 준비 상태를 200으로 반환한다.

| 메서드·경로 | 입력 | 성공 응답 |
| --- | --- | --- |
| GET /rounds/current | 인증 | 200 round_config 또는 204(선택된 활성 라운드 없음) |
| GET /models/latest | 인증 | 200 model_release, 미등록이면 404 |
| GET /models/{model_version}/manifest | 인증 | 200 shared_model_manifest |
| GET /models/{model_version}/weights | 인증 | 200 npz |
| POST /rounds/{round_id}/submissions | round_submission | 201 빈 본문, Location은 아래 PUT 경로 |
| PUT /rounds/{round_id}/submissions/delta | npz | 202 round_submit_ack |
| GET /rounds/{round_id}/result | 인증 | 200 round_submit_ack, 아직 제출 없으면 404 |

round_id/model_version은 C가 생성한 안전한 ID 패턴으로 제한한다. URL에 seller/customer/order ID를 넣지 않는다. C는 round·인증 seller별 제출 슬롯을 하나만 만든다. 같은 봉투·동일 바이트 재시도는 같은 결과를 반환하고, 달라지면 DUPLICATE_ROUND_SUBMIT. POST 예약 단계에는 집계하지 않는다.

첫 동기 버전은 **미리 고정한 참여자 전원 완료 또는 라운드 전체 폐기**다. 동적 부분 집계·지각 이월은 구현하지 않는다. 합성 데모 참여자는 3명 이상, 보호 모드 하한은 최소 5명이며 선택 프로토콜이 더 높은 수를 요구하면 따른다. min_clients라는 숫자만으로 보안을 주장하지 않는다.

모든 선택자의 completed=true와 유효한 delta가 deadline 안에 모이면 동일 가중치 평균(delta)을 기준 모델에 더한다. 샘플 수 가중치와 추가 step 재스케일은 없다. 미완료·탈락·deadline 초과이면 전체 폐기하고 마지막 모델을 유지한다. 실험 참여자 수와 deadline은 실행 config에 기록한다.

초기 ack 사용값은 accepted_on_time / aggregated_on_time / dropped_incomplete / round_discarded다. 대기 조회는 accepted_on_time 유지, 집계 성공 시 aggregated_in_round_id를 채운다. 그 외에는 null. staleness_rounds는 대기 null, 정상 집계 0. arrival_t_s는 C의 라운드 시작 후 단조시계 경과 초이며 가상 지연이 아니다. schema의 나머지 enum은 과거 확장용으로 보존했으며 현행 경로에서 출력하지 않는다.

coordinator가 합성 모드의 메모리 제출을 잃고 재시작하면 진행 라운드를 폐기한다. 개별 delta를 복구용 파일로 남기지 않는다. 보호 모드의 재시작·이탈·키 처리는 C가 선택 프로토콜대로 별도 명세화한다.

신규 판매자는 /models/latest → manifest/weights 검증 → B install_release 순서로 설치한다. 라운드 참가 이력이 없어도 모델을 받을 수 있다.

## 7. 보호 집계 전제

C의 첫 산출물은 프로토콜/라이브러리/버전, 서버·클라이언트 위협 가정, 참여·이탈 하한, 양자화/마스킹, 키 수명, 인증, 재시도·중복·실패 처리, 모델 일관성 확인과 테스트 계획이다. 이를 결정 기록과 schema에 반영한 뒤 G4를 구현한다.

- 중앙에는 허용된 참여 메타데이터와 집계 결과만 보인다. 개별 평문 delta·loss·로컬 sample count 금지.
- 로컬 지표가 필요하면 유효값 합계와 유효 여부를 프로토콜 안에서 함께 집계한다. 개인별 값이나 누락 사유를 중앙에 노출하지 않는다.
- 처음에는 고정 라운드 수를 쓰고, 보호된 검증 지표가 준비되기 전에는 중앙 조기 종료를 끈다.
- 최소 인원이 없거나 프로토콜 검증에 실패하면 집계를 공개하지 않는다. 같은 라운드를 작은 부분집합으로 반복 공개하지 않는다.
- FL_MODE=synthetic_plaintext는 신뢰된 합성 입력 경로로만 실행한다. source=live라는 필드만으로 합성임을 판정하지 않는다.
- G4 전 실데이터 분석·로컬 학습은 가능하지만 실데이터 유래 업데이트의 중앙 FL는 비활성이다.

## 8. 오류, 버전, fixture 게이트

오류 body는 contract_error.v1. code 우선순위는 validate.ERROR_CODE_ORDER이며 현재 다음 순서다:
SCHEMA_INVALID → UNKNOWN_FIELD → MISSING_REQUIRED_FIELD → INVALID_TYPE → INVALID_ENUM_VALUE → VERSION_MISMATCH → DUPLICATE_EVENT → DUPLICATE_IDEMPOTENCY_KEY → ILLEGAL_STATE_TRANSITION → STALE_STATUS_VERSION → FORBIDDEN → NOT_FOUND → MANIFEST_MISMATCH → TENSOR_SET_MISMATCH → DUPLICATE_ROUND_SUBMIT → ROUND_DISCARDED.

HTTP: 인증 실패 401/권한 403, 없음 404, 상태·중복·manifest/텐서 충돌·닫힌 라운드 409, payload 검증 422, 전송 크기 초과 413. 오류 detail은 고정된 비민감 설명으로 만든다. JSON 파싱 전 실패도 같은 안전한 오류 형식을 사용한다.

추가 필드를 거부하므로 optional 필드/enum 추가도 자동 호환이 아니다. 기존 v1 생산자/소비자 호환 fixture를 검증하고, 불가능하면 새 major와 명시적 전환을 만든다. 구현 전 스키마 설명 정리는 동작 변경 여부를 별도로 기록한다.

각 schema마다 valid 2개 이상, invalid 3개 이상. invalid JSON과 같은 이름의 .reason.txt에 첫 줄 기대 코드, 다음 줄 사유를 쓴다. UNKNOWN_FIELD/MISSING_REQUIRED_FIELD/INVALID_ENUM_VALUE 사례가 필수다. schema_version 외 enum/const가 없는 recommendation_request/round_config/model_release는 마지막 대신 VERSION_MISMATCH를 쓴다.

검증 명령: python -m commerce.tools.gate contracts
성공: 종료 0, 마지막 줄 CONTRACTS OK: schemas=<실제 수> fixtures=<실제 수> failures=0.
이 검사는 계약 형태·일부 의미 검증이다. 주문 복구·실제 학습·HTTP·보호 집계 성공을 뜻하지 않는다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](open-questions.md)를 확인한다.
