# 로컬 실행 · C

`python -m commerce.deploy.run_local --check`로 프로세스 계획을 확인한다.
`python -m commerce.deploy.run_local --smoke --merchants 2`는 중앙·coordinator·판매자 둘의 health를 확인한 뒤 종료한다.
`--smoke`를 빼면 Ctrl+C까지 유지한다. 포트는 8000, 8200, 8101부터이며 loopback에만 바인딩한다. 실행은 프로젝트 루트·개발 가상환경 기준이다.

이 런처는 health 뼈대용이며 FL는 항상 비활성, DB/모델 파일은 만들지 않는다. HTTP 200은 프로세스 기동 확인이며 업무 준비 상태는 ready=false다.
세션·화면을 추가하기 전 central.localhost / merchant-i.localhost의 loopback 해석, host-only 쿠키·CSRF·토큰/DB 설정을 구현해야 한다. 포트만 다른 localhost를 고객 세션 분리로 사용하지 않는다. 현재 강제 종료 처리는 실학습 체크포인트 복구를 제공하지 않는다.
