# 로컬 실행 · C

`python -m commerce.deploy.run_local --check`로 프로세스 계획을 확인한다.
`python -m commerce.deploy.run_local --smoke --merchants 2`는 중앙·coordinator·판매자 둘의 health를 확인한 뒤 종료한다.
`--smoke`를 빼면 Ctrl+C까지 유지한다. 포트는 8000, 8200, 8101부터이며 loopback에만 바인딩한다. 실행은 프로젝트 루트·개발 가상환경 기준이다.

FL는 항상 비활성이고, 지금은 DB/모델 파일을 만들지 않는다. smoke는 각 `/healthz`가 200이고 `service`가 맞으며 `ready`가 bool인지만 본다. 지금은 모두 `ready=false`지만, 서비스가 구현돼 true가 돼도 smoke가 깨지지 않게 값은 고정하지 않는다.
coordinator는 `REGISTRY_DIR`·`AUTH_FILE`·`ROUND_STATE_DIR` 없이 기동하므로 health만 제공한다. 런처로 합성 FL을 켜는 방법은 OQ16에서 정한다.

`python -m commerce.deploy.run_local --check-hosts --merchants N`은 `central.localhost`·`coordinator.localhost`·`merchant-i.localhost`가 OS 해석에서 loopback만 가리키는지 확인한다.
- 실패하면 hosts 파일에 넣을 `127.0.0.1 <이름>` 줄을 출력하고 종료 코드 1로 끝난다.
- 브라우저는 `*.localhost`를 스스로 loopback으로 처리하지만, 판매자 서버가 coordinator를 부르는 것 같은 서버 쪽 호출은 OS 해석을 쓴다.
- 2026-10-02 Windows 11 개발 PC에서는 다섯 이름 모두 해석되지 않았다. 이 이름을 쓰는 연결 전에 각자 이 검사를 실행한다.

세션·화면을 추가하기 전 위 loopback 해석, host-only 쿠키·CSRF·토큰/DB 설정을 구현해야 한다. 포트만 다른 localhost를 고객 세션 분리로 사용하지 않는다. 현재 강제 종료 처리는 실학습 체크포인트 복구를 제공하지 않는다.
