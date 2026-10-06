# 판매자 앱 시연 순서 · A

중간발표에서 판매자 한 곳(`merchant-1`)의 구매자·판매자 화면과 추천 연결을 보여주는 순서다. 모든 명령은 저장소 루트에서 PowerShell로 실행한다. 데이터는 전부 합성이다([데이터](../../../docs/design/data.md) §6).

## 1. 준비 (한 번)

```powershell
.venv\Scripts\python.exe -m pip install -r commerce\requirements-lock.txt
```

- lock에 torch가 들어 있다(#17). Windows에서 경로 길이 260자 제한에 걸리면 짧은 경로의 가상환경을 쓴다.
- 실제 모델 추천에는 B의 설치물이 `MODEL_DIR` 아래 있어야 한다(작업 규칙 §3, #33에서 B가 안내): frozen 텍스트 인코더는 `{MODEL_DIR}/frozen_text/`, 공통 base는 `{MODEL_DIR}/base/{variant}/`. 시연용 첫 release는 B가 `release_bundle.py`로 만들어 두었고, C의 #29 `install_latest` 또는 `import_release`로 설치한다. 없으면 B는 자기 fallback(`popularity.local`)으로 답하고 화면은 "공유 모델 준비 전"을 표시한다.
- 모델이 있으면 상품을 처음 넣은 직후 B가 상품 텍스트 벡터를 백그라운드에서 계산한다(상품 수천 개면 약 26초, 34개면 금방). 그동안 온 추천 요청은 계산이 끝날 때까지 기다린다.

## 2. 데이터 만들기

서버를 끈 상태에서 실행한다. 이미 있으면 지우고 새로 만든다.

```powershell
Remove-Item commerce\deploy\var\merchant_1\orders.sqlite* -ErrorAction SilentlyContinue
$env:MERCHANT_ID="merchant-1"; $env:FEATURE_DB_PATH="$PWD\commerce\deploy\var\merchant_1\features.sqlite"
.venv\Scripts\python.exe -m commerce.services.merchant_api.seed_demo_data --bulk 40
```

상품 34종, 고객 48명(취향 패턴 5종), 60일치 주문 약 280건, 쪽지·소식·공동구매·시세가 생긴다. 계정 비밀번호는 모두 `demo-pass-1234`, 판매자는 `owner-1`이다.

## 3. 서버 켜기

```powershell
$env:MERCHANT_ID="merchant-1"; $env:FEATURE_DB_PATH="$PWD\commerce\deploy\var\merchant_1\features.sqlite"
$env:MODEL_DIR="$PWD\commerce\deploy\var\merchant_1\models"; $env:MERCHANT_SECRET="아무-긴-문자열"
.venv\Scripts\python.exe -m uvicorn commerce.services.merchant_api.main:app --host 127.0.0.1 --port 8101
```

- 켜지자마자 밀린 상품·구매 기록이 뒤에서 B로 전달된다. 판매자 개요의 "추천 모델로 보낸 데이터"가 "모두 전달됨"이 되면 준비 끝이다(수 초).
- 여러 판매자·중앙·coordinator를 함께 띄우는 것은 C의 `commerce/deploy/run_local.py`가 한다. 이 문서는 판매자 하나만 띄운다.

## 4. 발표 흐름 (약 7분)

| 순서 | 화면 | 보여줄 것 |
| --- | --- | --- |
| 1 | 판매자 `http://127.0.0.1:8101/seller/merchant-1/signup` | 판매자 가입은 국세청 사업자등록 진위확인을 거친다(키 없으면 형식만 보는 mock임을 말한다) |
| 2 | 판매자 개요 (`owner-1`로 로그인) | **추천 모델 연결 상태**: "실제 모델로 추천 중"과 모델 버전, B로 전달된 구매 기록 수 |
| 3 | 구매자 `http://127.0.0.1:8101/buyer/merchant-1/` | "OO님을 위한 추천 상품". 실제 모델이면 고객마다 다르다. 패턴이 다른 고객으로 번갈아 로그인하면 차이가 잘 보인다: `bulk-cust-001`(아침 장보기) · `bulk-cust-002`(수산물) · `bulk-cust-003`(커피) · `bulk-cust-004`(과일·채소) · `bulk-cust-005`(고기 집밥) |
| 4 | 상품 상세 → 장바구니 → 주문 | 시세 그래프, 장바구니 여러 상품을 주문 1건으로, 폰 폭에서는 하단 고정 주문 바 |
| 5 | 판매자 주문 관리 | 수락 → 완료 → "모델에 반영됨" (완료된 주문만 B로 간다) |
| 6 | 터미널에서 시뮬레이터 | 아래 명령. 주문이 실시간으로 쌓이고 개요의 숫자가 늘어난다 |
| 7 | 판매자 모델 비교 | 한 고객에 대해 T-G·R-G·T-P·R-P 네 칸. 개인화가 아직이면 칸마다 이유가 보인다(P칸을 G로 채우지 않는다) |
| 8 | (여유 있으면) 공동구매·쪽지·소식 | 부가 기능 |

```powershell
.venv\Scripts\python.exe -m commerce.services.merchant_api.simulate_activity --base http://127.0.0.1:8101 --seller merchant-1 --interval 3
```

## 5. B의 #31이 머지된 날 확인할 것 (g2)

1. 1~3 그대로 실행 → 판매자 개요가 "실제 모델로 추천 중"인지. "연결됨 · 모델 준비 중"이면 B 설치물(release)이 `MODEL_DIR`에 없는 것이다.
2. 구매자 홈 배지가 사라졌는지(실제 모델 순위면 배지가 없다), 고객마다 추천이 다른지.
3. 주문 하나를 완료 → 개요의 전달 수가 늘고 주문 표에 "모델에 반영됨".
4. 서버를 껐다 켜기 → 로그인이 유지되고(`MERCHANT_SECRET`), 밀린 전달이 다시 나간다.
5. 모델 비교 화면이 네 칸을 채우거나 이유를 보여주는지.

## 6. 막힐 때

| 증상 | 원인 / 조치 |
| --- | --- |
| 추천 배지 "임시 · 추천 모델 미연결" | B runtime이 stub이다(#31 머지 전). 정상 |
| 추천 배지 "공유 모델 준비 전" | B는 연결됐지만 release가 설치되지 않았다 |
| 재시작하면 로그아웃됨 | `MERCHANT_SECRET`을 넣지 않았다 |
| 판매자 여럿을 띄웠는데 서로 로그인이 풀림 | 이제는 판매자별 쿠키라 풀리지 않는다. 풀리면 서버를 최신 코드로 다시 켠다 |
| 포트 사용 중 | 다른 서버가 8101을 쓰고 있다. 끄거나 `--port`를 바꾼다(URL도 함께) |
| 시뮬레이터 "seller login failed" | seed를 먼저 실행한다 |
| 화면은 뜨는데 데이터가 비어 있음 | seed의 `MERCHANT_ID`와 서버의 `MERCHANT_ID`가 다르다 |
