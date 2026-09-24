# FL coordinator · C

진입점: `main:app`. 현재 health만 있고 제출·집계·모델 배포 API는 없다.
작업 순서는 [C 카드](../../../docs/team/tasks/C.md)를 따른다. 첫 라운드 경로는 생성 합성 텐서만 쓴다.
집계 core는 웹 앱과 분리해 B의 `commerce/evaluation/`에서도 재사용하도록 만든다. 보호 프로토콜 미확정 상태에서 실데이터 경로를 활성화하지 않는다.

## 환경변수

`gate docs`가 첫 줄을 이 서비스 디렉터리의 코드 전체와 대조한다. 코드에서 새 값을 읽으면 같은 PR에서 이 줄을 고친다.

- 현재 코드가 읽는 값: 없음
- 구현 시 추가: `REGISTRY_DIR`, `AUTH_FILE`, `ROUND_STATE_DIR`, `FL_MODE`, `FL_MODEL_VARIANT`
