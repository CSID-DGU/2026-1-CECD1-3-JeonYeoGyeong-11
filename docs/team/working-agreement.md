# 작업 규칙 — 3인 분업과 통합

갱신: 2026-09-24 · D0017~D0021 · 소유: C, 패키지 소유권 변경은 공동 확인

## 1. 담당과 파일 소유권

| 패키지 | 담당 계정 | 소유 경로 |
| --- | --- | --- |
| A 플랫폼 | [A 카드](tasks/A.md) 2행 | commerce/apps/buyer/, commerce/apps/seller/, commerce/services/central_api/, commerce/services/merchant_api/, commerce/evaluation/metrics/ |
| B 데이터·모델 | [B 카드](tasks/B.md) 2행 | commerce/packages/data_adapters/, commerce/packages/recommender/, commerce/evaluation/ (metrics/ 제외) |
| C FL·통합 | [C 카드](tasks/C.md) 2행 | commerce/packages/contracts/, commerce/packages/fl_client/, commerce/services/fl_coordinator/, commerce/deploy/, commerce/tools/, commerce/tests/e2e/ |
| 공동 결정 | 팀 | docs/design/decisions.md |
| 문서 | 각 역할: 자기 카드 tasks/A·B·C.md, 자기가 새로 만든 `docs/design/` 문서 / A: architecture / B: model·data·evaluation·model-lab·comparison / C: interfaces·working-agreement·start·git-workflow·development·contracts·README·AGENTS와 도구별 포인터 파일 / 공동: docs/README·open-questions | 검토는 [Git 협업](git-workflow.md) 첫 표 |

**역할과 계정:** 역할은 사람이 정한다. 회의에서 정해지면 세 역할 카드 2행의 `담당 계정:`을 한 PR로 채운다. 공개 저장소이므로 실명 대신 GitHub 계정을 쓴다. CODEOWNERS·PR 작성자·리뷰 요청이 모두 같은 식별자를 쓰기 때문이다. 에이전트는 일을 시작하기 전에 `gh api user --jq .login`으로 로그인 계정을 확인하고, 그 계정이 사용자 본인 것이며 카드 2행과 같은지 사용자에게 확인받는다. 한 계정은 한 역할만 맡는다.

Python 패키지·모듈 이름은 밑줄(`data_adapters`), 브랜치 작업명은 하이픈(`commerce/a/first-order`)을 쓴다. 다른 패키지 파일은 읽을 수 있으나 임의 수정하지 않는다. 다른 파트 문제는 PR/이슈에 담당·재현·기대/실제 동작·검사 SHA를 남기고 해당 담당 에이전트가 수정한다. 소비자는 연결 동작을 재확인하며 내부 구현 전체를 대신 이해하거나 작성하지 않는다. 공통 계약 변경은 생산자·소비자 영향과 fixture를 함께 제안하고 C가 통합을 조정한다. 동료 검토는 다른 파트가 소비하는 동작과 공동 경계의 변경에 적용한다.

팀원용 [작업 시작 지시서](start.md)는 C가 유지한다. 상세 소유권·계약·완료 기준은 위 기준 문서를 따르며 지시서에서 별도로 바꾸지 않는다.
Git 운영 문서 git-workflow.md와 `.github/`의 템플릿·CI·CODEOWNERS는 C가 유지하며, 기존 저장소 설정과 소비자 영향을 확인한다.

## 2. 최초 작업과 의존성

**기본 방식은 각자 진행하며 결과를 남기고 비동기로 피드백을 반영하는 것이다.** 팀 시작 안내의 첫 작업 예제는 먼저 공유하기 좋은 결과이며 전원 대기 지점이 아니다. 맡은 범위 안의 독립 작업은 연속 수행할 수 있고, 상대 구현은 현행 계약의 stub으로 대체한다. 미확정 변경에 의존하는 부분만 보류하며 실제 연결 검사는 상대 결과가 준비되면 수행한다. 날짜별 진도나 매 작업 전원 회의를 요구하지 않는다.

| 작업 | 주 담당 | 제공할 것 | 기다릴 필요 없는 것 |
| --- | --- | --- | --- |
| 합성 상품·계정·주문과 이벤트 재전달 | A | 주문 API, catalog_item, purchase_event | B 실모델은 fixture로 대체 |
| 데이터 어댑터·NLP | B | 두 출처와 live 입력 샘플, 텍스트 wrapper, 텍스트만 기준선 | C 집계는 단일 판매자 로컬 학습으로 대체 |
| 계약·합성 FL·보호 집계 조사 | C | validator, 더미 round, 모델 release, 보호 방식 선정 기록 | B 학습은 stub trainer로 대체 |

연결 순서: A→B 이벤트·카탈로그 → B→A 실제 추천 → B↔C 합성 FL → 보호 집계.
NLP는 B의 별도 구현 작업이다. 기존 완성 모듈을 가정하지 않는다. 상세 시작 지시는 docs/team/tasks/에 있다.

D0018에 따라 B의 모델 실험은 A 화면과 병행한다. B는 화면 없는 학습/평가 실행기를 소유하고 C의 공통 집계 모듈을 호출한다. 모델 core와 실험 실행기를 웹 앱에 묶지 않는다. checkpoint → 초기 release 인계는 [독립 모델 실험](../design/model-lab.md)을 따른다.

D0019에 따라 전체 추천 가중치를 공유한 뒤 query_proj/scorer만 로컬 개인화한다. B는 학습·개인화·비교 API, A는 비교 화면, C는 variant별 공통 모델 집계·배포를 맡는다. 개인화 설정이나 모델 선택을 A/C가 별도로 구현하지 않는다.

## 3. 런타임과 배치 계약

목표 스택은 Python 3.11, FastAPI/Jinja2, SQLite, PyTorch다. 첫 CPU 실행을 기준으로 의존성을 검증한다. CPU 전용 여부와 속도 수치를 추정해 확정하지 않는다.
현재 `commerce/requirements-lock.txt`에는 뼈대 실행에 필요한 것만 있다(jsonschema, numpy, fastapi, uvicorn, httpx). **Jinja2는 A가 화면을, PyTorch는 B가 모델을 구현하는 PR에서 공용 `commerce/requirements-dev.txt`와 lock에 추가하고 C가 호환을 검증한다.** 목표 스택에 적혀 있다는 이유로 이미 설치돼 있다고 가정하지 않는다.

각 서비스가 읽는 환경변수는 그 서비스 README의 '환경변수' 절에 적고, `docs` 게이트가 서비스 코드 전체와 대조한다. 서비스 담당이 자기 README를 고치므로 환경변수를 추가해도 이 문서를 바꿀 필요가 없다.

| 프로세스 | 모듈 | 목표 주소 (현재 런처는 127.0.0.1:포트) | 환경변수 |
| --- | --- | --- | --- |
| 중앙 공개 앱 | commerce.services.central_api.main:app | central.localhost:8000 | [central README](../../commerce/services/central_api/README.md) |
| 판매자 i | commerce.services.merchant_api.main:app | merchant-i.localhost:8100+i | [merchant README](../../commerce/services/merchant_api/README.md) |
| coordinator | commerce.services.fl_coordinator.main:app | coordinator.localhost:8200 | [coordinator README](../../commerce/services/fl_coordinator/README.md) |

**필수 환경변수.** 런처가 넘기지 않는 필수 값은 docs 게이트가 막는다. C에게 런처(`commerce/deploy/run_local.py`) 변경을 요청해 먼저 병합하거나, `os.environ.get`과 기본값을 쓴다.

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
| 판매자 토큰 해시(평문 토큰 없이 작업 비용이 있는 해시만) | commerce/deploy/var/fl/{variant}/auth.json | C |
| 라운드 상태 | commerce/deploy/var/fl/{variant}/rounds/ | C, 비밀·개별 평문 제외 |

REGISTRY_DIR는 디렉터리, AUTH_FILE은 파일이다. 중앙 프로세스에 판매자 DB 경로나 비밀키를 주입하지 않는다. 동일 PC 프로세스 분리는 host 관리자에 대한 암호적 격리가 아니다.

의존성은 공용 `commerce/requirements-dev.txt` 한 곳에 선언하고, 추가한 역할이 PR에 용도를 적는다. C가 호환 버전을 검증한 lock을 관리한다. 역할별 requirements 파일을 따로 만들거나 서로 다른 torch 버전을 고정하지 않는다.

### 머신이 세 대라는 전제

A/B/C는 서로 다른 로컬에서 동작하며 공유 수단은 저장소뿐이다. 자기 머신에서만 되는 상태를 만들지 않는다.

- lock이 바뀌었으면 **자기 머신에서 다시 설치하고**, 게이트는 자기 머신에서 직접 실행한다. 상대가 통과시켰다는 기록만 보고 자기 환경이 같다고 보지 않는다.
- **의존성을 추가하면 같은 PR에서 `commerce/requirements-dev.txt`와 `commerce/requirements-lock.txt`를 함께 갱신한다.** 자기 머신에만 설치하고 lock을 두지 않으면 다른 두 머신은 재현할 수 없고 아무도 알아채지 못한다. lock을 바꾼 PR의 검토는 [Git 협업](git-workflow.md) 첫 표를 따른다.
- PR에 **실행한 OS와 Python 버전**을 남긴다. `run_local.py`에는 Windows 분기가 있고 CI는 Linux다. "내 머신에서 통과"는 OS를 밝히지 않으면 재현 근거가 아니다.
- 로컬에만 있는 파일(`commerce/deploy/var/`, `.team/`, 내려받은 원자료)은 다른 머신에 없다. 그 존재를 전제로 한 지시나 검사를 만들지 않는다.

## 4. 완료 게이트

명령: python -m commerce.tools.gate <이름>. 구현 전 게이트는 NOT_IMPLEMENTED와 비영 종료 코드를 유지한다. 출력 문자열만 만들어 통과시키지 않는다.

| 이름 | 확인 | 주 담당 |
| --- | --- | --- |
| scaffold | 초기 import·runtime 연결·health 뼈대 (업무 완료 제외) | C, A/B 각자 실행 |
| docs | 문서 대 코드 대조. 문서를 고치거나 환경변수·게이트·공개 import를 바꾸면 실행 | 변경한 사람 |
| policy | 일부 위험 신호 검사. 실패는 담당자가 수정하고 정책 변경이 필요한 경우만 사람이 결정한다 | 변경한 사람 |
| contracts (G1) | 모든 스키마·fixture·의미 검증. 현재 개수는 러너가 출력 | C, A/B 각자 실행 |
| business | 아래 업무 게이트를 모두 실행. 실패(1·2)가 없으면 통과, 미구현·부분 구현(3)은 허용. CI가 실행 | C |
| a1 | 주문·상태 전이·타 판매자 권한 거부·중앙 비잔류·쿠키 범위, 평가 지표·기준선의 손계산 예제 | A |
| b1 | 두 어댑터·합성 live 이벤트·텍스트 생성·식별자·결측 보존 | B |
| b2 | NLP artifact/freeze, variant별 공유 export, gradient/고정 검증 분할, 개인화 두 그룹 제한·base 불변·옛 tail 거부 | B |
| c1 | 합성 텐서 검증·균등 집계·배포·미달 폐기·신규 판매자 설치, variant 혼합 거부 | C |
| g2 (G2) | 주문→특징→실제 추천·장애 후 1회 반영, 비교 snapshot/후보 동일성·개인화 불가 표시 | A 실행, B 연결, C 통합 |
| g3 (G3) | 합성 3판매자 FL 후 base 변경·개인화 재생성, 별도 상품 목록의 4번째 판매자 점수화 | C 실행, B 실모델 |
| g4 (G4) | 보호 집계, 개별 평문 업데이트·손실 비노출, 이탈/미달/잘못된 라운드 실패 | C 실행, A/B 검토 |

g2·g3·g4의 selfcheck는 `commerce/tests/e2e/`(C 소유)에 C가 작성한다. A·B는 자기 API와 시나리오 입력을 제공한다. 표의 "실행"은 그 시나리오를 돌려 결과를 PR에 올리는 역할이다.

G3의 신상품 검사는 관계를 만드는 통제된 구매 예제를 사용한다. 구매 한 건이면 반드시 점수가 바뀐다고 가정하지 않는다. 캐시·feature_epoch 갱신과 관계 변화가 있을 때 표현 갱신을 확인한다. 보호 집계는 G3의 부가 표기가 아니라 별도 G4다.

업무 게이트는 각 담당 소유 경로의 selfcheck 모듈에 **미리 연결돼 있다.** 위치는 [계약 문서](../contracts.md)의 표, 종료 코드 규약(0 통과·3 부분 구현·1 실패·2 검사 깨짐)은 `commerce/tools/gate.py` 머리말에 있다. 담당은 그 위치에 selfcheck를 만들고 `gate.py`는 고치지 않는다. 자기 selfcheck를 더하거나 고치는 것은 내부 변경이다. CI의 `business` 게이트가 모든 PR에서 이를 돌리므로, selfcheck는 lock과 저장소만으로 Linux에서 빠르게 끝나야 하고 임시 디렉터리만 쓴다. 게이트 항목은 CI에서 돌 수 있는 것(작은 합성 입력, 작은 테스트 모델)만 둔다. 원자료·실제 가중치로만 확인되는 것(예: b1의 원자료 재현, [데이터](../design/data.md) §6)은 PR에 따로 보고한다. 자기 테스트는 자기 소유 경로 아래 두고 selfcheck가 호출한다. 합성 통과를 원자료 전수 검증으로 보고하지 않는다.
g2는 먼저 기본 주문/추천 경로를 연결하고 이후 네 결과 비교 검사를 추가한다. 중간 단계 성공은 부분 완료로 보고하며 비교가 없는 상태를 D0019의 최종 g2 완료로 표시하지 않는다. 두 variant가 실제 학습되기 전에는 A가 계약 stub으로 화면을 개발할 수 있다.

## 5. 첫 통합에 필요한 기능

A: 공개 목록 → 판매자 화면 → 상품·장바구니 → 주문 요청 → 판매자 수락/완료 → 구매자 이력·추천. 완료 상태는 실제 결제 완료라고 표시하지 않는다.
B: NLP와 텍스트만 추천 → 관계 특징·시퀀스 → variant별 shared export/load/train → 후반부 개인화·compare_local → 두 콜드스타트의 2×2 평가.
C: 모델 manifest·release → 합성 동기 FL → variant 분리·개인화 미제출 확인 → 보호 집계. 최초 동기 모드에는 지각 이월 큐가 없다.

성능 실험 R1/R2는 실험실 FL 시뮬레이션으로 G4와 별도로 실행한다(D0020). G4 뒤 보호 모듈로 다시 돌린 결과는 추가로 보고한다. 지각 이월·작업량 실험·속도 주입은 필수 서비스 구현 일정과 분리한다.

여기서 G4 보호 모듈 검증은 화면 없이 먼저 수행할 수 있다. 모델 실험이 G2 화면 통합을 기다리는 의존성을 만들지 않는다. 서비스 전체 중앙 비잔류 검사는 A 연결 뒤 완료한다. 플랫폼 startup은 준비된 모델 load이며 장시간 학습 실행이 아니다.

## 6. 작업량 재배분

B의 부담이 가장 크므로 평가 지표와 모델이 아닌 기준선은 처음부터 A가 맡는다(§1, [A 카드](tasks/A.md)). §8의 D3·D7 점검에서 B의 어댑터·NLP 진행을 확인한다. B가 늦으면 두 출처 어댑터(원자료 → purchase_event·catalog_item 변환)를 A가 받을 수 있다. A는 이미 live 쪽에서 같은 계약을 만든다. 합성 시드·입력 품질 fixture는 A, 재현 실행·결과 저장 도구는 C가 받을 수 있다. 데이터 어댑터 이관은 원자료 접근·라이선스를 먼저 확인하고 소유 경로를 변경한다.
보호 집계가 막히면 C의 게이트 시나리오 구현은 A가 받을 수 있다. 모델 의미·평가 해석은 B, 보호 가정·프로토콜은 C가 계속 책임진다.

## 7. Git과 변경 절차

커밋·검토·병합은 [Git 협업](git-workflow.md)이 기준이다. 저장소는 CSID-DGU/2026-1-CECD1-3-JeonYeoGyeong-11이며 기존 연구 코드를 보존한다.

## 8. 과제와 일정

약 2주 완성을 목표로 한다. 매일 진도를 맞추거나 고정 회의를 하지는 않는다. 다만 아래 세 점검 시점에는 각자 결과를 PR에 올리고 범위를 판단한다. D1은 세 역할의 계정을 카드에 기입한 다음 날이다. 제출일(D-day)과 발표 형식은 팀이 채운다.

| 시점 | 확인할 결과 | 미달이면 |
| --- | --- | --- |
| D3 | C: HTTP와 분리한 집계 core(전원 완료 또는 폐기, 균등 평균)를 B가 호출할 수 있게 병합. 보호 방식 선정 기록과 작은 합성 예제(D0021 범위). B: Instacart 소형 표본의 어댑터와 text builder, torch를 넣은 lock PR(OQ18, C 검토). A: 주문 상태·transaction·outbox. 사람: OQ13·14·15·18·19 결정 | 집계 core가 없으면 B는 실행기 안에 같은 규칙의 임시 함수를 두고 진행한다. core가 병합되면 바꿔 끼우고 결과가 같은지 확인한다. 보호 방식이 보이지 않으면 C는 c1과 집계 core를 먼저 끝내고, G4는 조사 기록과 fail-closed로 마무리할 준비를 한다 |
| D7 | B: Instacart E-G0(text_only)와 기준선 수치, 텍스트 모델 크기와 계산 예산 확정. 관계 예산은 관계 특징의 첫 소형 실행에서 정한다. A: 평가 지표·기준선 모듈(B의 E-G0 채점에 쓴다), a1 일부와 이벤트 전달. C: c1 합성 라운드. 사람: OQ16·17 결정 | 아래 줄이는 순서대로 범위를 줄인다 |
| D10 | 1차 대비와 기준선 결과가 나와 있다. B의 text_only runtime(ingest·upsert·predict_local)이 A에 연결돼 주문→추천 왕복이 돈다. 기능 동결 | 이후에는 새 기능 없이 결과·보고서·재현만 한다. G4가 통과하지 못했으면 조사 기록과 protected의 fail-closed 상태를 한계로 보고한다 |

사람 결정은 담당 역할이 선택지·추천안·영향을 이슈로 올리고 팀이 정한다. 기한까지 정하지 못하면 그 결정에 걸린 항목은 부분 완료로 두고 줄이는 순서를 적용한다.

**줄이는 순서.** 늦어지면 1번부터 차례로 뺀다.

1. Dunnhumby 재현(보조)
2. C-natural
3. A-few
4. 개인화 두 결과(T-P·R-P). 비교 화면의 개인화 칸은 "준비되지 않음"과 사유로 표시한다
5. 서비스 합성 FL 연결과 base 교체 시연(g3)
6. 관계 셔플 대조. 빼면 1차 결론의 문구를 [평가](../design/evaluation.md) §4에 미리 정한 대로 약하게 쓴다
7. 콜드스타트 A-0·C-new
8. FL vs local_only

플랫폼 쪽은 1번부터: 중앙 공개 목록의 검색·정렬 같은 편의 기능 → 판매자 운영 화면을 최소 폼으로 → 관계 모델의 서비스 연결(text_only로 서비스).

1차 대비와 기준선([평가](../design/evaluation.md) §4), 서비스의 주문→추천 왕복(text_only 모델로 충분)은 빼지 않는다. G4는 별도 트랙이다(D0020).

**최종 산출물.**
- 시연: [모델 비교](../design/comparison.md) §6 가운데 주문→추천 왕복, 같은 snapshot의 비교(준비되지 않은 결과는 사유 표시), 그리고 줄이지 않았다면 신규 판매자와 합성 FL 라운드 뒤 base 교체.
- 결과표: 1차 대비와 기준선을 맨 앞에 두고, 남은 비교는 줄이는 순서의 역순으로 둔다. 각 표에 seed 수, 판매자 수, "비보호 FL 시뮬레이션" 여부를 적는다.
- 보고서: 문제와 가정(데이터 한계 포함) → 설계 → 실험 설정 → 결과(1차 대비 먼저) → 보호 범위와 한계 → 재현 방법.
- 재현: 게이트 명령, 실험 설정 파일 경로, 사용한 원자료 파일의 해시(파일 자체는 올리지 않음).

전체 roster 실험과 확장은 후순위다. 검토는 결과가 준비되면 진행하며 합의된 내부 작업은 자율 수행한다. 모델 비용과 보호 구현 가능성은 먼저 확인하고, 실패를 최종 완료로 포장하지 않는다.

관련 미확정 사항과 완료 증거는 [열린 구현 항목](../design/open-questions.md)를 확인한다.
