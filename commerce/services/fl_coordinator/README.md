# FL coordinator · C

진입점: `main:app`. 현재 HTTP는 health만 있고 제출·집계·모델 배포 API는 없다.
HTTP와 분리한 합성 집계 core는 있다: `round_core.py`(`ModelRegistry`, `SyntheticRound`: 고정 cohort 전원 완료 시 균등 평균, 아니면 전체 폐기, 메모리만 사용)와 `npz_payload.py`(npz 길이·항목·shape·dtype·유한값 검증). 검사는 `gate c1`.
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. 첫 라운드 경로는 생성 합성 텐서만 쓴다.
집계 core는 웹 앱과 분리해 B의 `commerce/evaluation/`에서도 재사용한다. B의 실험실 실행기는 `round_core.aggregate_uniform(deltas)`를 직접 부른다. 입력은 고정 cohort 순서의 delta 목록이며, 미완료 참여자는 `None`으로 넣는다. 반환값은 float32 균등 평균이고, 하나라도 `None`이거나 목록이 비면 `None`(라운드 폐기)이다. 결과는 비보호 FL 시뮬레이션이다(D0020). 보호 프로토콜 미확정 상태에서 실데이터 경로를 활성화하지 않는다.

`ModelRegistry(root)`는 variant 하나의 `REGISTRY_DIR`에 release를 둔다. `<root>/<model_version>/`에 `weights.npz`·`manifest.json`·`release.json`·`provenance.json`을 쓰고, 임시 디렉터리에서 이름을 바꿔 넣은 뒤 `<root>/latest`를 마지막에 바꾼다. 기동할 때 모든 release의 해시·크기·manifest를 다시 검증하며, 하나라도 틀리면 `RegistryCorrupt`로 기동을 거부한다. `provenance.json`은 학습된 release·random init·FL 라운드를 구분하는 로컬 운영 기록이며 서빙하지 않는다.

`service.Coordinator`는 variant 하나의 상태다. 운영자가 `open_round(cohort, RoundSettings(...))`로 라운드를 열며 cohort는 그때 고정된다. 열린 라운드는 한 번에 하나다. 누가 언제 라운드를 여는지(트리거)는 OQ08에서 정한다. `RoundLedger`는 `ROUND_STATE_DIR`에 라운드마다 상태·기준 모델·deadline·cohort 크기·결과 모델만 쓴다. delta·지표·비밀·참여자 ID는 쓰지 않는다. 재시작하면 열려 있던 라운드를 `discarded`로 기록하고 그 round_id는 다시 쓰지 않는다. `protected` 모드는 g4 전까지 생성 단계에서 `FeatureNotImplemented`로 실패한다.

초기 release는 업로드 API 없이 coordinator 호스트에서 넣는다: `python -m commerce.services.fl_coordinator.import_release --registry <REGISTRY_DIR> --manifest <json> --weights <npz> --model-version <id> --kind trained|random_init`. B의 불변 architecture config 등록값과의 대조는 그 등록부가 생긴 뒤 추가한다.

## 합성 라운드 경로 초안

C가 구현하면서 바꿔도 되는 초안이다. 지켜야 할 불변식은 [인터페이스](../../../docs/design/interfaces.md) §6에 있다.

| 메서드·경로 | 입력 | 성공 응답 |
| --- | --- | --- |
| GET /rounds/current | 인증 | 200 round_config 또는 204(선택된 활성 라운드 없음) |
| GET /models/latest | 인증 | 200 model_release, 미등록이면 404 |
| GET /models/{model_version}/manifest | 인증 | 200 shared_model_manifest |
| GET /models/{model_version}/weights | 인증 | 200 npz |
| POST /rounds/{round_id}/submissions | round_submission | 201 빈 본문, Location은 아래 PUT 경로 |
| PUT /rounds/{round_id}/submissions/delta | npz | 202 round_submit_ack |
| GET /rounds/{round_id}/result | 인증 | 200 round_submit_ack, 아직 제출 없으면 404 |

ack는 처음에 accepted_on_time / aggregated_on_time / dropped_incomplete / round_discarded만 쓴다. POST 단계에서는 집계하지 않는다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: 없음
- 구현 시 추가: `REGISTRY_DIR`, `AUTH_FILE`, `ROUND_STATE_DIR`, `FL_MODE`, `FL_MODEL_VARIANT`
