# 작업 규칙 — 3인 분업과 통합

갱신: 2026-09-23 · D0017~D0019 · 소유: C, 패키지 소유권 변경은 공동 확인

## 1. 담당과 파일 소유권

| 패키지 | 담당자 | 소유 경로 |
| --- | --- | --- |
| A 플랫폼 | 미배정 | commerce/apps/buyer/, commerce/apps/seller/, commerce/services/central_api/, commerce/services/merchant_api/ |
| B 데이터·모델 | 미배정 | commerce/packages/data_adapters/, commerce/packages/recommender/, commerce/evaluation/ |
| C FL·통합 | 미배정 | commerce/packages/contracts/, commerce/packages/fl_client/, commerce/services/fl_coordinator/, commerce/deploy/, commerce/tools/, commerce/tests/e2e/ |
| 공동 결정 | 팀 | docs/design/decisions.md |
| 문서 | A: architecture / B: model·data·evaluation·model-lab·comparison / C: interfaces·working-agreement·README·AGENTS·CLAUDE / 공동: docs/README | 변경 소비자가 리뷰. COMPARISON 화면 절은 A 리뷰 |

하위 이름은 밑줄을 사용한다. 다른 패키지 파일은 읽을 수 있으나 임의 수정하지 않는다. 공통 계약 변경은 생산자·소비자 영향과 fixture를 함께 제안하고 C가 통합을 조정한다. 작성자 본인의 변경은 해당 소비자의 동료 검토를 거친다.

팀원용 [작업 시작 지시서](start.md)는 C가 유지하고 A/B가 자기 역할 절을 검토한다. 상세 소유권·계약·완료 기준은 위 기준 문서를 따르며 지시서에서 별도로 바꾸지 않는다.
Git 운영 문서 git-workflow.md와 `.github/`의 템플릿·향후 CI·CODEOWNERS는 C가 유지하며, 기존 저장소 설정과 소비자 영향을 확인한다.

## 2. 최초 작업과 의존성

**기본 방식은 각자 진행하며 결과를 남기고 비동기로 피드백을 반영하는 것이다.** 팀 시작 안내의 첫 작업 예제는 먼저 공유하기 좋은 결과이며 전원 대기 지점이 아니다. 맡은 범위 안의 독립 작업은 연속 수행할 수 있고, 상대 구현은 현행 계약의 stub으로 대체한다. 미확정 변경에 의존하는 부분만 보류하며 실제 연결 검사는 상대 결과가 준비되면 수행한다. 날짜별 진도나 매 작업 전원 회의를 요구하지 않는다.

| 작업 | 주 담당 | 제공할 것 | 기다릴 필요 없는 것 |
| --- | --- | --- | --- |
| 합성 상품·계정·주문과 이벤트 재전달 | A | 주문 API, catalog_item, purchase_event | B 실모델은 fixture로 대체 |
| 데이터 어댑터·NLP | B | 두 출처와 live 입력 샘플, 텍스트 wrapper, 텍스트만 기준선 | C 집계는 단일 판매자 로컬 학습으로 대체 |
| 계약·합성 FL·보호 집계 조사 | C | validator, 더미 round, 모델 release, 보호 방식 선정 기록 | B 학습은 stub trainer로 대체 |

연결 순서: A→B 이벤트·카탈로그 → B→A 실제 추천 → B↔C 합성 FL → 보호 집계.
NLP는 B의 별도 구현 작업이다. 기존 완성 모듈을 가정하지 않는다. 상세 시작 지시는 docs/team/tasks/에 있다.

D0018에 따라 B의 모델 실험은 A 화면과 병행한다. B는 화면 없는 학습/평가 실행기를 소유하고 C의 공통 집계 모듈을 호출한다. 모델 core와 실험 실행기를 웹 앱에 묶지 않는다. checkpoint → 초기 release 인계는 ../design/model-lab.md를 따른다.

D0019에 따라 전체 추천 가중치를 공유한 뒤 query_proj/scorer만 로컬 개인화한다. B는 학습·개인화·비교 API, A는 비교 화면, C는 variant별 공통 모델 집계·배포를 맡는다. 개인화 설정이나 모델 선택을 A/C가 별도로 구현하지 않는다.

## 3. 런타임과 배치 계약

Python 3.11, FastAPI/Jinja2, SQLite, PyTorch. 첫 CPU 실행을 기준으로 의존성을 검증한다. CPU 전용 여부와 속도 수치를 추정해 확정하지 않는다.

| 프로세스 | 모듈 | 기본 주소 | 필수 설정 |
| --- | --- | --- | --- |
| 중앙 공개 앱 | commerce.services.central_api.main:app | central.localhost:8000 | CENTRAL_DB_PATH |
| 판매자 i | commerce.services.merchant_api.main:app | merchant-i.localhost:8100+i | MERCHANT_ID, MERCHANT_DB_PATH, FEATURE_DB_PATH, MERCHANT_SECRET, COORDINATOR_URL, FL_CLIENT_TOKEN |
| coordinator | commerce.services.fl_coordinator.main:app | coordinator.localhost:8200 | REGISTRY_DIR, AUTH_FILE, ROUND_STATE_DIR, FL_MODE, FL_MODEL_VARIANT |

모든 호스트명은 로컬 loopback으로 해석되도록 C가 실행 환경에서 확인한다. 안 되면 hosts 또는 로컬 DNS 설정 절차를 제공한다. 같은 localhost의 포트만으로 쿠키를 분리하지 않는다. 쿠키는 Domain을 지정하지 않는 host-only, HttpOnly, SameSite 설정과 CSRF 검사를 사용한다. HTTPS 배포에서는 Secure를 켠다.

중앙 → coordinator → 판매자 순으로 /healthz의 200을 확인한다. C 소유 run_local.py가 기동·종료를 담당한다. A는 앱 lifespan에서 C의 FL client 시작/종료 훅만 호출한다. 학습 작업은 요청 이벤트 루프를 막지 않는다.

FL_MODEL_VARIANT는 text_only 또는 text_relation이며 실행 중 바꾸지 않는다. C는 같은 값을 판매자 FL client의 시작 설정에 전달하고 B 호출의 model_variant로 사용한다. 불변 architecture config/manifest와 다르면 실패한다. 두 실험은 우선 순차 실행하며 variant마다 registry·인증 설정·라운드 디렉터리를 분리한다. 두 release를 미리 설치한 판매자 runtime은 coordinator가 한 variant로 실행 중이어도 두 모델의 추론을 제공할 수 있다.

| 저장소 | 경로 | 단독 쓰기 소유 |
| --- | --- | --- |
| 주문·카탈로그·이벤트·전달 진행 | commerce/deploy/var/merchant_i/orders.sqlite | A |
| 특징·반영 이벤트 ID·feature_epoch·캐시 | commerce/deploy/var/merchant_i/features.sqlite | B |
| 공통 base | commerce/deploy/var/merchant_i/models/base/{variant}/{base_version}/ | B, C는 B API 호출 |
| 개인화 결과 | commerce/deploy/var/merchant_i/models/personal/{variant}/{base_version}/{revision}/ | B, 중앙 배포/집계 제외 |
| 집계 모델 | commerce/deploy/var/fl/{variant}/registry/{model_version}/ | C |
| 작업 비용 해시 설정 | commerce/deploy/var/fl/{variant}/auth.json | C |
| 라운드 상태 | commerce/deploy/var/fl/{variant}/rounds/ | C, 비밀·개별 평문 제외 |

REGISTRY_DIR는 디렉터리, AUTH_FILE은 파일이다. 중앙 프로세스에 판매자 DB 경로나 비밀키를 주입하지 않는다. 동일 PC 프로세스 분리는 host 관리자에 대한 암호적 격리가 아니다.

각 패키지는 의존성 선언을 소유한다. C가 호환 버전을 검증한 개발 환경 제약 파일을 관리한다. 각자 requirements에 서로 다른 torch 버전을 임의 고정하지 않는다.

## 4. 완료 게이트

명령: python -m commerce.tools.gate <이름>. 구현 전 게이트는 NOT_IMPLEMENTED와 비영 종료 코드를 유지한다. 출력 문자열만 만들어 통과시키지 않는다.

| 이름 | 확인 | 주 담당 |
| --- | --- | --- |
| scaffold | 초기 import·runtime 연결·health 뼈대 (업무 완료 제외) | C, A/B 각자 실행 |
| contracts (G1) | 모든 스키마·fixture·의미 검증. 현재 개수는 러너가 출력 | C, A/B 각자 실행 |
| a1 | 주문·상태 전이·타 판매자 권한 거부·중앙 비잔류·쿠키 범위 | A |
| b1 | 두 어댑터·합성 live 이벤트·텍스트 생성·식별자·결측 보존 | B |
| b2 | NLP artifact/freeze, variant별 공유 export, gradient/고정 검증 분할, 개인화 두 그룹 제한·base 불변·옛 tail 거부 | B |
| c1 | 합성 텐서 검증·균등 집계·배포·미달 폐기·신규 판매자 설치, variant 혼합 거부 | C |
| g2 (G2) | 주문→특징→실제 추천·장애 후 1회 반영, 비교 snapshot/후보 동일성·개인화 불가 표시 | A 실행, B 연결, C 통합 |
| g3 (G3) | 합성 3판매자 FL 후 base 변경·개인화 재생성, 별도 상품 목록의 4번째 판매자 점수화 | C 실행, B 실모델 |
| g4 (G4) | 보호 집계, 개별 평문 업데이트·손실 비노출, 이탈/미달/잘못된 라운드 실패 | C 실행, A/B 검토 |

G3의 신상품 검사는 관계를 만드는 통제된 구매 예제를 사용한다. 구매 한 건이면 반드시 점수가 바뀐다고 가정하지 않는다. 캐시·feature_epoch 갱신과 관계 변화가 있을 때 표현 갱신을 확인한다. 보호 집계는 G3의 부가 표기가 아니라 별도 G4다.

각 패키지가 selfcheck.py에 검증 함수를 제공하고 C가 gate에 연결한다. 합성/원자료 실행 모드를 결과에 명시한다. 원자료 없는 합성 통과를 데이터 전수 검증으로 보고하지 않는다.
g2는 먼저 기본 주문/추천 경로를 연결하고 이후 네 결과 비교 검사를 추가한다. 중간 단계 성공은 부분 완료로 보고하며 비교가 없는 상태를 D0019의 최종 g2 완료로 표시하지 않는다. 두 variant가 실제 학습되기 전에는 A가 계약 stub으로 화면을 개발할 수 있다.

## 5. 첫 통합에 필요한 기능

A: 공개 목록 → 판매자 화면 → 상품·장바구니 → 주문 요청 → 판매자 수락/완료 → 구매자 이력·추천. 완료 상태는 실제 결제 완료라고 표시하지 않는다.
B: NLP와 텍스트만 추천 → 관계 특징·시퀀스 → variant별 shared export/load/train → 후반부 개인화·compare_local → 두 콜드스타트의 2×2 평가.
C: 모델 manifest·release → 합성 동기 FL → variant 분리·개인화 미제출 확인 → 보호 집계. 최초 동기 모드에는 지각 이월 큐가 없다.

성능 실험 R1/R2는 G4 뒤 중앙 FL로 실행한다. 지각 이월 X4·작업량 실험 X5·속도 주입은 필수 서비스 구현 일정과 분리한다.

여기서 G4 보호 모듈 검증은 화면 없이 먼저 수행할 수 있다. 모델 실험이 G2 화면 통합을 기다리는 의존성을 만들지 않는다. 서비스 전체 중앙 비잔류 검사는 A 연결 뒤 완료한다. 플랫폼 startup은 준비된 모델 load이며 장시간 학습 실행이 아니다.

## 6. 작업량 재배분

A의 g2 연결 작업이 끝난 시점과 C의 c1 시점에 B의 NLP·어댑터 진행을 확인한다. B가 막히면 합성 시드/입력 품질 fixture는 A, 재현 실행·결과 저장 도구는 C가 받을 수 있다. 데이터 어댑터 이관은 원자료 접근·라이선스를 먼저 확인하고 소유 경로를 변경한다.
보호 집계가 막히면 C의 게이트 시나리오 구현은 A가 받을 수 있다. 모델 의미·평가 해석은 B, 보호 가정·프로토콜은 C가 계속 책임진다.

## 7. Git과 변경 절차

지속적인 커밋·동기화·상호 검토는 [git-workflow.md](git-workflow.md)를 기준으로 한다. 저장소는 사용자 지정 CSID-DGU/2026-1-CECD1-3-JeonYeoGyeong-11이며 기존 연구 코드를 보존한다.

- 공동 통합은 `commerce/integration`, 실제 작업은 `commerce/a|b|c/<작업명>`의 짧은 브랜치와 독립 작업 디렉터리를 사용한다. 현재 로컬 폴더의 Git 연결부터 확인한다.
- 작업 시작 때 fetch하고 최근 integration 변경·관련 PR·계약 변경을 읽는다. 새 작업과 이어가기 모두 적용한다.
- 작은 단위로 commit하고 자기 작업 브랜치에 push해 Draft PR로 공유한다. 진행·의존성·검사 SHA는 PR에서 관리한다.
- 작성자 외 검토와 관련 소비자 확인 후 PR로 병합한다. 기본 병합 방식은 merge commit이며, 공개 이력 재작성·통합 브랜치 직접 push를 하지 않는다.
- 계약 변경은 schema/fixture/동작 문서/소비자 영향을 함께 검토한다. 정상 실행 경로를 유지할 호환 전환 또는 공동 변경을 준비한다.
- 실제 데이터·산출물·비밀을 제외하고 자기 변경 경로만 stage한다. 승인된 작업의 Git 수행 범위는 GIT_WORKFLOW §9 요청문을 따른다.

초기 PR #3에 CI와 작업 브랜치를 준비했다. 실제 담당자 배정·브랜치 보호·required check 활성화는 별도 준비 사항이다.

## 8. 과제와 일정

약 2주 완성을 목표로 하며 날짜별 진도·고정 회의를 강제하지 않는다. [팀 시작 안내](start.md)의 소형 평가 범위·비용 제한·비동기 진행을 적용한다. 각자 가능한 때에 몰아서 진행할 수 있으며 절대 제출일·담당 이름·발표 형식은 팀이 채운다.

전체 roster 실험과 확장은 후순위로 두고, 두 출처의 작은 고정 표본으로 필수 경로를 완성한다. 검토는 결과가 준비되면 진행하며 합의된 내부 작업은 자율 수행한다. 마감이 가까워지면 새 기능보다 오류 수정·재현을 우선한다. 모델 비용·보호 구현 가능성은 먼저 확인하고 실패를 최종 완료로 포장하지 않는다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](../design/open-questions.md)를 확인한다.
