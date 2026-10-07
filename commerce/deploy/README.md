# 로컬 실행 · C

`python -m commerce.deploy.run_local --check`로 프로세스 계획을 확인한다.
`python -m commerce.deploy.run_local --smoke --merchants 2`는 중앙·coordinator·판매자 둘의 health를 확인한 뒤 종료한다.
`--smoke`를 빼면 Ctrl+C까지 유지한다. 포트는 8000, 8200, 8101부터이며 loopback에만 바인딩한다. 실행은 프로젝트 루트·개발 가상환경 기준이다.

FL는 항상 비활성이고, 지금은 DB/모델 파일을 만들지 않는다. smoke는 각 `/healthz`가 200이고 `service`가 맞으며 `ready`가 bool인지만 본다. 지금은 모두 `ready=false`지만, 서비스가 구현돼 true가 돼도 smoke가 깨지지 않게 값은 고정하지 않는다.
coordinator는 `REGISTRY_DIR`·`AUTH_FILE`·`ROUND_STATE_DIR` 없이 기동하므로 health만 제공한다. run_local은 FL을 켜지 않는다.

합성 FL 시연은 `python -m commerce.deploy.fl_demo`만 켠다([결정](../../docs/design/decisions.md) D0025).
- 새 시연 폴더(`commerce/deploy/var/fl_demo/<시각>/`)에 B의 g3 시나리오로 판매자 4곳을 만들고, 판매자마다 생성한 입력의 내용 해시를 확인 파일로 남긴다.
- coordinator(8250)는 이 프로세스에서, 판매자 앱(8151~)은 별도 프로세스에서 띄우고 `--rounds`만큼 라운드를 연다. 네 번째 판매자는 cohort 밖이라 새 base만 설치한다.
- 기본은 CI 크기 모델이라 인코더가 필요 없다. 실제 모델은 `--encoder-dir`·`--release-dir`을 준다. `--keep`이면 끝난 뒤에도 판매자 앱을 유지한다.
- `--tamper <판매자>`는 확인 파일을 쓴 뒤 생성기 밖 구매를 한 건 넣는다. 그 판매자는 제출하지 않고 라운드는 폐기된다(OQ17 보호의 시연).
- B runtime의 `snapshot_digest`가 있어야 돈다. 없으면 준비 단계에서 멈춘다.
- 판매자 앱의 출력은 `<시연 폴더>/<판매자>/merchant.log`에 남는다.

### 전체 기능 리허설

`python -m commerce.deploy.fl_demo --rounds 2 --rehearse`

- **판매자 구성:**
  - cohort 3곳은 위와 같이 B 시나리오로 채운다.
  - cohort 밖 네 번째 판매자("가게")는 A의 데모 seed(`seed_demo_data --bulk 20`)로 A 경로를 통해 채운다. 판매자 계정 `owner-1`, 고객 계정, 상품, 지난 주문이 생기고, A 앱이 기동하면서 그 outbox를 B로 보낸다.
  - 가게는 설치 전용(`install_only`)이라 새 base만 받는다.
- **라운드 뒤 HTTP로 확인하는 것** (모두 PASS면 `REHEARSAL OK`):

  | # | 확인 |
  | --- | --- |
  | R1 | 가게가 최신 FL base로 서빙한다 |
  | R2 | 고객(`cust-eunji`)이 로그인한다 |
  | R3 | 주문 전 구매자 홈이 응답한다. 이 고객은 B에 구매 이력이 없어 "첫 방문 고객" 대체 추천이 나온다 |
  | R4 | A의 JSON 경로로 주문 → 수락 → 완료되고, B가 반영한다 |
  | R5 | 주문 뒤 구매자 홈에 모델 추천(FL base)이 나온다 |
  | R6 | 판매자 비교 화면에 T-G·R-G·T-P·R-P가 나온다 |
  | R7 | cohort 판매자에 A 경로로 상품을 하나 등록하면 확인 파일과 내용이 달라져 그 판매자가 제출하지 않고, 다음 라운드는 폐기된다 |

- 사람이 직접 보려면 `--keep`을 붙이고 `http://127.0.0.1:8154/buyer/g3-seller-new/`(고객), `http://127.0.0.1:8154/seller/g3-seller-new/compare`(판매자)를 연다. 비밀번호는 A의 seed 출력에 있다.
- **실제 모델:** 인코더를 B가 고정한 revision으로 받은 뒤 `--encoder-dir <…/paraphrase-multilingual-MiniLM-L12-v2/e8f8c21…>`를 준다. 받는 방법은 `python -m commerce.evaluation.encoder_probe --download --only minilm-l12`이다. 학습된 release가 있으면 `--release-dir`, 없으면 base-0은 random init이다.
- **측정** (2026-10-07, Windows 11·16 GB·Python 3.11.4, base-0 random init, B의 `snapshot_digest`는 아래 제안 구현을 임시로 넣어 실행):
  - CI 크기 모델: 2라운드와 리허설 7/7 통과, 약 54초.
  - 실제 모델(MiniLM, R_lm): 1라운드 63초, 2라운드 71초, 리허설 7/7 통과.
  - 실제 모델 10라운드: 10라운드 모두 집계, 리허설 7/7, 152초, 판매자 로그 오류 0건.
    - 5초마다 잰 메모리에서 라운드를 거듭해도 cohort 판매자는 약 360~370 MB로 일정해, 누수 징후는 없었다.
    - 판매자 a(R7에서 상품 등록 뒤 재계산)와 가게(화면·비교 계산)는 그 단계에서 약 980·790 MB까지 늘었다.
    - 시스템 여유 메모리는 최저 0.25 GB까지 내려갔다. 메모리가 모자라면 Windows가 작업 메모리를 줄이므로 위 수치는 실제보다 작을 수 있다.
  - 판매자 4곳이 각자 인코더를 올리므로 여유 메모리가 약 5 GB 필요하다. 여유가 1.1 GB이던 첫 실행에서는 R4의 주문 완료가 한 번 500으로 실패했고(187초), 원인은 확인하지 못했다. 이후 merchant.log를 남기며 두 번 다시 돌렸을 때는 재현되지 않았다.

`python -m commerce.deploy.run_local --check-hosts --merchants N`은 `central.localhost`·`coordinator.localhost`·`merchant-i.localhost`가 OS 해석에서 loopback만 가리키는지 확인한다.
- 실패하면 hosts 파일에 넣을 `127.0.0.1 <이름>` 줄을 출력하고 종료 코드 1로 끝난다.
- 브라우저는 `*.localhost`를 스스로 loopback으로 처리하지만, 판매자 서버가 coordinator를 부르는 것 같은 서버 쪽 호출은 OS 해석을 쓴다.
- 2026-10-02 Windows 11 개발 PC에서는 다섯 이름 모두 해석되지 않았다. 이 이름을 쓰는 연결 전에 각자 이 검사를 실행한다.

세션·화면을 추가하기 전 위 loopback 해석, host-only 쿠키·CSRF·토큰/DB 설정을 구현해야 한다. 포트만 다른 localhost를 고객 세션 분리로 사용하지 않는다. 현재 강제 종료 처리는 실학습 체크포인트 복구를 제공하지 않는다.
