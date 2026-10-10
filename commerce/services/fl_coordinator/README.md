# FL coordinator · C

진입점: `main:app`. `REGISTRY_DIR`·`AUTH_FILE`·`ROUND_STATE_DIR`가 없으면 health만 제공한다. 런처는 지금 이 값을 넘기지 않는다. 셋을 주면 아래 합성 라운드 경로를 연다. `FL_MODE=protected`(기본값)는 g4 전까지 기동 단계에서 `FeatureNotImplemented`로 실패하며, 평문으로 떨어지지 않는다. 검사는 `gate c1`.
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. 합성 라운드는 생성 합성 텐서만 쓴다. 보호 프로토콜 미확정 상태에서 실데이터 경로를 활성화하지 않는다.

## 구성 요소

- `round_core.py`: `SyntheticRound`(고정 cohort 전원 완료 시 균등 평균, 아니면 전체 폐기, 개별 delta는 메모리에만 두고 라운드가 끝나면 버림)와 `ModelRegistry`.
- `aggregate_uniform(deltas)`: B의 실험실 실행기가 직접 부르는 같은 규칙이다. 입력은 고정 cohort 순서의 delta 목록이며 미완료 참여자는 `None`으로 넣는다. 반환값은 float32 균등 평균이고, 하나라도 `None`이거나 목록이 비면 `None`(라운드 폐기)이다. 결과는 비보호 FL 시뮬레이션이다(D0020).
- `npz_payload.py`: npz 길이·항목·shape·dtype·유한값 검증.
- `ModelRegistry(root)`: variant 하나의 `REGISTRY_DIR`에 release를 둔다.
  - `<root>/<model_version>/`에 `weights.npz`·`manifest.json`·`release.json`·`provenance.json`을 쓴다.
  - 임시 디렉터리에서 이름을 바꿔 넣은 뒤 `<root>/latest`를 마지막에 바꾼다.
  - 기동할 때 모든 release를 다시 검증하며, 하나라도 틀리면 `RegistryCorrupt`로 기동을 거부한다.
  - `provenance.json`은 학습된 release·random init·FL 라운드를 구분하는 로컬 운영 기록이며 서빙하지 않는다.
- `service.Coordinator`: variant 하나의 상태다.
  - 운영자가 `open_round(cohort, RoundSettings(...))`로 라운드를 열며 cohort는 그때 고정된다. 열린 라운드는 한 번에 하나다.
  - 누가 언제 라운드를 여는지(트리거)와 고정 라운드 수는 OQ08에서 정한다. 그때까지 라운드는 같은 프로세스 안에서만 연다(`app.state.coordinator`).
  - `RoundLedger`는 `ROUND_STATE_DIR`에 라운드마다 상태·기준 모델·deadline·cohort 크기·결과 모델만 쓴다. delta·지표·비밀·참여자 ID는 쓰지 않는다.
  - 재시작하면 열려 있던 라운드를 `discarded`로 기록하고 그 round_id는 다시 쓰지 않는다.
- `auth.py`: 판매자 토큰 `<seller_id>.<secret>`. `AUTH_FILE`(파일)에는 scrypt 해시만 둔다.
  - 발급과 교체: `python -m commerce.services.fl_coordinator.auth issue --auth-file <AUTH_FILE> --seller <id>`. 토큰은 그때 한 번만 출력된다. 폐기는 `revoke`다.
  - variant마다 별도 파일을 쓴다.
- `import_release.py`: 초기 release를 업로드 API 없이 coordinator 호스트에서 넣는다.
  - `python -m commerce.services.fl_coordinator.import_release --registry <REGISTRY_DIR> --manifest <json> --weights <npz> --model-version <id> --kind trained|random_init`.
  - B의 불변 architecture config 등록값과의 대조는 그 등록부가 생긴 뒤 추가한다.

## 합성 라운드 경로

불변식은 [인터페이스](../../../docs/design/interfaces.md) §6·8에 있다. `/healthz`를 뺀 모든 경로는 판매자 Bearer 토큰이 필요하다. seller는 토큰에서 읽고 본문의 seller와 대조한다.

| 메서드·경로 | 입력 | 성공 응답 |
| --- | --- | --- |
| GET /rounds/current | 인증 | 200 round_config, 이 판매자가 든 열린 라운드가 없으면 204 |
| GET /models/latest | 인증 | 200 model_release, 미등록이면 404 |
| GET /models/{model_version}/manifest | 인증 | 200 shared_model_manifest |
| GET /models/{model_version}/weights | 인증 | 200 npz (`application/octet-stream`) |
| POST /rounds/{round_id}/submissions | round_submission (1 MiB 이하) | 201 빈 본문, Location은 아래 PUT 경로 |
| PUT /rounds/{round_id}/submissions/delta | npz (8 MiB 이하) | 202 round_submit_ack |
| GET /rounds/{round_id}/result | 인증 | 200 round_submit_ack |

- ack는 accepted_on_time / aggregated_on_time / dropped_incomplete / round_discarded만 쓴다. POST 단계에서는 집계하지 않는다.
- 같은 바이트의 재시도는 같은 ack를 돌려주고, 다른 바이트는 409 `DUPLICATE_ROUND_SUBMIT`이다.
- 오류 본문은 `contract_error.v1`이며 코드 외의 입력·식별자를 담지 않는다.
  - 403 `FORBIDDEN`(cohort 밖, 다른 seller의 본문)
  - 404 `NOT_FOUND`(없는 모델·라운드·경로)
  - 422 검증 오류
  - 409 상태·중복·manifest 충돌·폐기된 라운드
- 재시작 전 라운드의 결과 조회는 409 `ROUND_DISCARDED`다.
- PUT 전에 POST가 없으면 409 `ILLEGAL_STATE_TRANSITION`이다.
- 전송 한도는 수신 중 누적 바이트에 적용한다.
- **OQ13 전 임시 처리:** 토큰이 없거나 틀리면 401과 `WWW-Authenticate: Bearer`만 보내고 본문은 비운다. v1 enum에 인증 오류 코드가 없기 때문이다.
- **임시 처리:** 크기 초과는 413에 `SCHEMA_INVALID` 본문을 쓴다. v1 enum에 크기 초과 코드가 없어 가장 가까운 코드를 골랐다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: `REGISTRY_DIR`, `AUTH_FILE`, `ROUND_STATE_DIR`, `FL_MODE`
- `REGISTRY_DIR`·`ROUND_STATE_DIR`는 디렉터리, `AUTH_FILE`은 파일이다. 셋은 함께 주며, variant마다 따로 둔다([작업 규칙](../../../docs/team/working-agreement.md) §3).
- 구현 시 추가: `FL_MODEL_VARIANT`(지금은 registry의 manifest 고정이 variant 혼합을 막는다)
