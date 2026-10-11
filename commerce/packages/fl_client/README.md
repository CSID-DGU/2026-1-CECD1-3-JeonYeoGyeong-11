# 판매자 FL client · C

진입점: `lifecycle.create_client(runtime, jobs, config)`.
A가 제공한 B runtime과 동일한 jobs 객체를 사용한다. `await start()` / `await stop()`은 앱 lifespan에서 호출한다.
현재 기본 enabled=false로 네트워크/학습을 하지 않는다. enabled=true는 두 mode 모두 FeatureNotImplemented로 시작을 거부한다. synthetic_plaintext는 OQ17(신뢰된 합성 입력 확인), protected는 g4 뒤에 연다.
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. B 호출은 주입된 jobs에서 실행하고, 실데이터 유래 평문 제출을 허용하지 않는다. 보호 프로토콜 선정과 메시지 계약은 별도 작업이다.

합성 라운드의 판매자 쪽 구성 요소는 있다. 아직 `start()`에 연결하지 않았고 c1이 생성 tensor runtime으로만 실행한다.

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
