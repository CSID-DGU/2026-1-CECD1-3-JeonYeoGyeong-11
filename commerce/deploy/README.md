# 로컬 실행 · C

`python -m commerce.deploy.run_local --check`로 프로세스 계획을 확인한다.
`python -m commerce.deploy.run_local --smoke --merchants 2`는 중앙·coordinator·판매자 둘의 health를 확인한 뒤 종료한다.
`--smoke`를 빼면 Ctrl+C까지 유지한다. 포트는 8000, 8200, 8101부터이며 loopback에만 바인딩한다. 실행은 프로젝트 루트·개발 가상환경 기준이다.

FL는 항상 비활성이고, 지금은 DB/모델 파일을 만들지 않는다. smoke는 각 `/healthz`가 200이고 `service`가 맞으며 `ready`가 bool인지만 본다. 지금은 모두 `ready=false`지만, 서비스가 구현돼 true가 돼도 smoke가 깨지지 않게 값은 고정하지 않는다.
세션·화면을 추가하기 전 central.localhost / merchant-i.localhost의 loopback 해석, host-only 쿠키·CSRF·토큰/DB 설정을 구현해야 한다. 포트만 다른 localhost를 고객 세션 분리로 사용하지 않는다. 현재 강제 종료 처리는 실학습 체크포인트 복구를 제공하지 않는다.
