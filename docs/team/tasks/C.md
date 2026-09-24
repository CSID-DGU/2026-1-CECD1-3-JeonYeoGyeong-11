# C 작업 카드 — 계약·FL·통합
담당 계정: 미선점 · 선점은 [작업 규칙](../working-agreement.md) §1 · 실행 기준 [D0017~D0019](../../design/decisions.md)

## 목표와 소유
A/B가 독립 구현해도 연결되는 계약과 실행 환경을 제공하고 최종 개별 업데이트 보호 집계를 완성한다.
소유: commerce/packages/contracts/, commerce/packages/fl_client/, commerce/services/fl_coordinator/, commerce/deploy/, commerce/tools/, commerce/tests/e2e/.
첫 읽기는 [팀 시작 안내](../start.md) §0·2를 따른다. 아래는 기능별 참고이며 전부 선독할 목록이 아니다: 환경·게이트는 [작업 규칙](../working-agreement.md), 집계·배포는 [인터페이스 계약](../../design/interfaces.md), 실제 모델 연결은 [모델 경계](../../design/model.md) §4~8, 보호 전제는 [아키텍처](../../design/architecture.md).
지속 작업: [Git 협업](../git-workflow.md)에 따라 작업별 브랜치와 Draft PR을 사용한다. 시작 때 B의 manifest/학습 API와 A의 실행 훅 변경을 확인한다. 저장소 준비를 맡으면 통합 브랜치·보호 설정·계약 CI를 별도 산출물로 기록한다.

## 첫 작업 두 갈래
아래는 피드백을 거쳐 나눠 수행할 초기 순서다. 먼저 [팀 시작 안내](../start.md)의 C 첫 작업을 확인하고, 전체 coordinator와 보호 집계를 한 번에 구현하지 않는다.

1. contracts 게이트와 fixture를 확인하고 합성 stub trainer로 round → 제출 → 균등 평균 → release를 연결한다. stub trainer는 자기 소유 경로에 두고 주입한다([개발 안내 §상대 모듈을 대체하는 방법](../../development.md#상대-모듈을-대체하는-방법)). 더미 tensor는 `shared_model_manifest.v1/valid/dummy_tensors_for_round_bringup.json` fixture를 기준으로 만든다.
2. 동시에 최종 보호 집계 구현의 타당성을 먼저 확인한다. 기존 검증된 구현·정확한 버전·환경 호환·위협 가정·참여/이탈 하한·메시지·양자화·실패 조건을 작은 예제로 검토한다.

보호 방식 선정 기록은 C 첫 산출물이다. G3까지 다 만든 뒤 처음 검토하지 않는다. 선정 전에도 합성 평문 c1과 A/B 계약 작업은 진행 가능하다.
라이브러리 미지원/예산 초과면 근거와 대안을 팀에 보고한다. “메모리만 썼으므로 안전”으로 목표를 낮추지 않는다.

## c1 산출물
- Bearer seller 인증, round config, round_submission/npz 검증, 재시도 멱등성.
- 사전 고정 cohort 전원 완료 또는 전체 폐기. 지각 이월 없음.
- 실제 npz 전송 길이/해시와 개별 tensor 크기를 구분한 제한.
- immutable model release, latest/manifest/weights, 신규 판매자 설치.
- coordinator 재시작 시 진행 합성 라운드 폐기, 개별 delta 미저장.

더미 평균 검증 c1과 B 실모델 g3를 구분한다. G3에는 미참여·비중복 카탈로그의 네 번째 판매자가 필요하다.

## 통합 실행
run_local.py와 호환 의존성·환경 예제를 소유한다. 중앙/coordinator/판매자 health를 순서대로 확인하고 종료를 처리한다.
판매자별 origin을 분리하고 루프백 해석·쿠키 범위를 확인한다. REGISTRY_DIR는 디렉터리, AUTH_FILE은 파일.
A는 lifespan hook, B는 runtime API를 제공한다. C가 다른 소유자의 DB를 직접 변경하지 않는다.

[독립 모델 실험](../../design/model-lab.md)과 D0018에 따라 집계 모듈을 HTTP 앱과 분리하고 B의 실험 실행기에서도 재사용한다. 초기 학습 release의 로컬 import/검증/registry 등록을 제공한다. G4 보호 모듈 테스트는 A 화면 없이 합성 입력으로 실행할 수 있으며, 서비스 전체 비잔류는 A 연결 후 추가 검사한다.

## 공통 모델과 개인화 분리
D0019/[모델 비교](../../design/comparison.md)을 따른다. FL_MODEL_VARIANT는 한 실행 동안 고정하고 text_only/text_relation의 registry·라운드·인증 설정을 분리한다. 먼저 순차 실행하고 두 공통 release를 판매자에 미리 설치한다. shape가 같더라도 architecture/variant가 다른 제출은 c1에서 거부한다.
B train_round는 round_config의 공통 base에서 시작해야 한다. C는 B의 서빙 state_dict나 개인화 경로를 읽어 제출물을 만들지 않는다. 개인화 결과·개별 검증 결과·로컬 식별자는 중앙 등록/집계 대상이 아니다.
새 공통 버전 설치 후 옛 개인화 사용 중단과 새 base에서의 재개를 g3에서 B와 확인한다. A 비교 화면에 필요한 metadata는 판매자 내부 반환 객체이며 중앙 수집 API를 추가하지 않는다.

## G4 완료 조건
선정 기록 → 보호 프로토콜 schema/fixture → 합성 보호 집계 → 실패/재시도 검사 → 실제 데이터 경로 활성화 순서다.
중앙이 개별 평문 delta/loss를 읽지 않음, min 참여와 이탈 조건, round/model 일관성, replay/중복 거부, 비밀 폐기, 같은 cohort 결과의 반복 부분 공개 방지, 양자화 오차를 확인한다.
고객 거래가 중앙 HTTP/log/temp/DB에 없는지도 A/B와 함께 확인한다.
프로토콜이 보장하지 않는 악성 서버/공모/최종 모델 추론은 결과에 한계를 명시한다.


공통 시작점: [팀 시작 안내](../start.md). 최초에 담당 카드와 해당 계약을 읽고, 이어갈 때는 관련 diff·검사·남은 작업만 확인한다.
