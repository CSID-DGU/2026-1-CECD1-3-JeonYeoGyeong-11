# 데이터 계약과 검증 상태

## 계약 목록

필드·타입은 [JSON Schema](../commerce/packages/contracts/schemas/)가 기준입니다. 현재 스키마와 검증기는 구현되어 있으며, 이를 사용하는 서비스·학습 runtime은 구현 예정입니다.

| 계약 | 용도 | 범위 |
| --- | --- | --- |
| commerce_order.v1 | 주문 상태 기록 | 판매자 로컬 |
| purchase_event.v1 | 완료 구매 이벤트 | 판매자 서비스 → 로컬 특징 모듈 |
| catalog_item.v1 | 상품 정보 | 판매자 로컬 |
| catalog_snapshot.v1 | 공개 상품 목록 | 판매자 → 중앙 공개 카탈로그 |
| recommendation_request.v1 / recommendation.v1 | 추천 요청·결과 | 판매자 로컬 |
| contract_error.v1 | 오류 응답 | 경계별 안전한 오류 표현 |
| shared_model_manifest.v1 | 공유 tensor 구조 | 공통 모델 검증 |
| round_config.v1 | 학습 기준 모델·라운드 설정 | 학습 조정 |
| delta_manifest.v1 | 업데이트 구조와 완료 여부 | 판매자 학습 모듈 → FL client |
| round_submission.v1 | 업데이트·지표·payload 설명 | 생성 합성 데이터의 평문 연결 검사 전용 |
| model_release.v1 | 배포 버전·해시·크기 | 모델 설치 검증 |
| round_submit_ack.v1 | 제출·집계 진행 결과 | 합성 FL 연결 검사 |

보호 집계의 프로토콜 메시지는 아직 구현·확정되지 않았습니다. `round_submission.v1` 자체에는 개별 업데이트를 숨기는 기능이 없습니다.

## 현재 검증하는 내용

- 필수 필드·타입·허용값·미지 필드 거부.
- 유한한 숫자, 식별자 및 tensor 집합 등 구현된 의미 규칙.
- 정상 예제의 수락과 오류 예제의 기대 오류 코드.
- 중첩 계약의 schema 사본과 원본 일치.

`purchase_event_id`는 규칙대로 계산한 값이며 검증기가 대조합니다. 그 밖의 예제 해시는 형태 확인용입니다. 실제 파일 해시·전송 크기·인증·학습 결과를 대신 검증하지 않습니다. 과거 확장을 위한 일부 허용값도 남아 있으므로 스키마에 존재한다는 것만으로 기능 지원을 주장하지 않습니다.

## 실행과 완료 상태

```text
python -m commerce.tools.gate contracts
```

`contracts`는 종료 코드 0과 `CONTRACTS OK` 요약으로 판정합니다. 정상 예제는 `fixtures/<계약>/valid`, 오류 예제는 `invalid`에 있으며 대응하는 `.reason.txt`의 첫 줄이 기대 오류 코드입니다.

각 게이트가 무엇을 확인하는지는 [작업 규칙](team/working-agreement.md) §4가 기준입니다. 이 표는 검사 위치와 현재 상태만 적습니다.

| 게이트 | 검사 위치 | 현재 상태 |
| --- | --- | --- |
| contracts | `commerce/packages/contracts/validate.py` | 실행 가능 |
| scaffold | `commerce/tests/e2e/test_scaffold.py` | 실행 가능 (업무 검증 제외) |
| docs | `commerce/tools/doccheck.py` | 실행 가능 |
| policy | `commerce/tools/policycheck.py` | 실행 가능 |
| business | 아래 업무 게이트를 모두 실행 | 실행 가능 |
| a1 | `commerce/services/merchant_api/selfcheck.py` | 미구현 |
| b1 | `commerce/packages/data_adapters/selfcheck.py` | 미구현 |
| b2 | `commerce/packages/recommender/selfcheck.py` | 미구현 |
| c1 | `commerce/services/fl_coordinator/selfcheck.py` | 미구현 |
| g2 | `commerce/tests/e2e/selfcheck_g2.py` | 미구현 |
| g3 | `commerce/tests/e2e/selfcheck_g3.py` | 미구현 |
| g4 | `commerce/tests/e2e/selfcheck_g4.py` | 미구현 |

업무 게이트는 위 위치에 미리 연결돼 있습니다. 모듈이 없으면 종료 코드 3의 `NOT_IMPLEMENTED`입니다. 담당은 그 위치에 selfcheck를 만들고 같은 PR에서 상태를 `부분 구현` 또는 `실행 가능`으로 바꿉니다. `gate.py`는 고치지 않습니다. 기능 구현 PR에는 실제 호출 방법, 오류 처리, 사용한 데이터 종류, 검사 결과를 함께 기록합니다. 계약 변경 시 생산자·소비자 코드, 예제와 설명을 함께 맞춥니다.

## 기존 설계 참조

일부 스키마 설명과 코드 주석의 `CONTRACTS.md`는 [인터페이스 계약](design/interfaces.md), `MODEL_BOUNDARY.md`는 [모델 경계](design/model.md)를 가리킨다. 이전 R/SC 규칙 번호 등은 역사적 식별자이며 현재 필드·의미 검증은 schema/validator/fixture가 수행한다. 현행 설계 선택은 [결정 요약](design/decisions.md)에 있다.

Python 로컬 함수·반환 타입·객체 수명은 [개발 안내](development.md)와 `commerce/packages/contracts/ports.py`, `types.py`를 따른다. JSON 스키마는 이번 뼈대 추가로 변경하지 않았다.
