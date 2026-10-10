# 판매자 FL client · C

진입점: `lifecycle.create_client(runtime, jobs, config)`.
A가 제공한 B runtime과 동일한 jobs 객체를 사용한다. `await start()` / `await stop()`은 앱 lifespan에서 호출한다.
현재 기본 enabled=false로 네트워크/학습을 하지 않는다. protected는 g4 전까지 FeatureNotImplemented로 시작을 거부한다. synthetic_plaintext는 coordinator URL·토큰·합성 입력 확인 파일이 모두 있어야 시작하며([결정](../../../docs/design/decisions.md) D0025), 판매자 앱의 환경변수 경로에는 확인 파일이 없어 계속 거부된다. 이 설정은 `commerce/deploy/fl_demo.py`만 만든다(policy 게이트).
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. B 호출은 주입된 jobs에서 실행하고, 실데이터 유래 평문 제출을 허용하지 않는다. 보호 프로토콜 선정과 메시지 계약은 별도 작업이다.

합성 라운드의 판매자 쪽 구성 요소다. `start()`가 synthetic_plaintext에서 폴링 루프(`tick`)로 쓴다: 더 새 release를 먼저 설치하고, 이 판매자의 열린 라운드에 한 번만 참여한다.

- `attestation`: 합성 입력 확인 파일(`c.synthetic_attestation.v1`). 학습 직전 B runtime의 `snapshot_digest(local_data_ref)`가 파일의 내용 해시와 다르거나 runtime에 그 메서드가 없으면 `SyntheticInputRefused`로 아무것도 제출하지 않는다. `source` 필드는 보지 않는다.

- `submission.build_submission`: B의 `TrainingResult.shared_delta`만으로 round_submission과 npz를 만든다.
  - tensor 집합·float32·유한값·지표(유한한 비음수 또는 null)를 확인한다.
  - 서빙 모델·개인화 tail·`export_shared_state`는 읽지 않는다.
- `submission.verify_release`: model_release·manifest 해시·weights 크기와 SHA-256·tensor 집합을 확인한다. 확인을 통과해야 `install_release`에 넘긴다.
- `transport.CoordinatorTransport`: coordinator 경로를 Bearer 토큰으로 부른다.
  - weights 수신에 8 MiB 한도를 적용한다.
  - 제출의 Location이 정해진 경로가 아니면 따라가지 않는다.
- `rounds.participate`: 열린 라운드가 이 판매자의 것일 때만 학습한다.
  - round_config의 manifest·architecture가 B의 manifest와 다르면 학습하지 않는다.
  - 설정한 variant를 B 호출에 그대로 넘긴다.
- `rounds.install_latest`: latest → manifest·weights 검증 → B `install_release`. 라운드 참가 이력이 없는 신규 판매자도 같다.
