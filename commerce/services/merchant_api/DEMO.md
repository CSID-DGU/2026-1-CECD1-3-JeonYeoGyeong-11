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

서버를 끈 상태에서 실행한다. 이미 있으면 지우고 새로 만든다. **A의 `orders.sqlite`와 B의 `features.sqlite`는 한 판매자의 짝이라 함께 지운다**(interfaces.md §2 '재시작과 초기화'). 하나만 지우면 B에 같은 고객 이력이 두 벌 쌓이고 상품 전달이 거부된다(seed가 이 경우를 감지하고 멈춘다). 옛 원장으로 만든 개인화(`models/personal/`)도 지운다. 공통 base release(`models/base/`)와 인코더(`models/frozen_text/`)는 남긴다.

```powershell
Remove-Item commerce\deploy\var\merchant_1\orders.sqlite*, commerce\deploy\var\merchant_1\features.sqlite* -ErrorAction SilentlyContinue
Remove-Item commerce\deploy\var\merchant_1\models\personal -Recurse -ErrorAction SilentlyContinue
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

## 7. 통합 리허설: 판매자 여러 곳

FL은 판매자가 여럿이어야 한다. C의 런처(`run_local.py`)는 판매자 i를 `merchant-i`, `commerce/deploy/var/merchant_i/`, 포트 `8100+i`로 띄운다. 같은 규칙으로 데이터를 한 번에 만든다(서버를 끈 상태, 초기화는 2단계처럼 판매자 폴더마다 `orders.sqlite*`·`features.sqlite*`를 함께 지운 뒤).

```powershell
.venv\Scripts\python.exe -m commerce.services.merchant_api.seed_demo_data --all 4 --bulk 40
.venv\Scripts\python.exe -m commerce.deploy.run_local --merchants 4
```

| 판매자 | 가게 | 상품 | 주소 |
| --- | --- | --- | --- |
| merchant-1 | 제주 유기농 농장 (기본 + bulk) | 34종 | http://127.0.0.1:8101/buyer/merchant-1/ |
| merchant-2 | 제주 바다 수산 | 8종 | http://127.0.0.1:8102/buyer/merchant-2/ |
| merchant-3 | 한라 베이커리 & 커피 | 12종 | http://127.0.0.1:8103/buyer/merchant-3/ |
| merchant-4 | 오이네 청과·정육 | 13종 | http://127.0.0.1:8104/buyer/merchant-4/ |

- 가게마다 고객 30명(`cust-001`~`cust-030`, 비밀번호 `demo-pass-1234`, 가게마다 따로인 계정)과 60일치 주문이 있다. 판매자는 모두 `owner-1`. 우유·계란·쌀은 여러 가게가 함께 판다.
- 한 브라우저로 여러 가게에 동시에 로그인해도 된다(쿠키가 판매자별이다).
- 판매자 "모델 비교" 화면의 **개인화 실행** 버튼은 그 가게 기록으로 두 모델의 마지막 층을 학습한다(OQ08). FL 라운드가 도는 중이면 "이미 학습 중"으로 거절된다.
- `--merchants 5` 이상이면 5번부터 수산·베이커리·청과가 다시 돌아온다.

## 8. 플랫폼 데이터셋: 매장 6곳 + SNS·공동구매·리뷰 (발표용 권장)

7단계의 `seed_demo_data --all` 대신, 발표 화면이 "실제 SNS·장터처럼" 보이도록 매장 6곳의 90일 활동을 만든다(서버를 끈 상태, 판매자 폴더의 `orders.sqlite*`·`features.sqlite*`·`media/`를 함께 지운 뒤).

```powershell
.venv\Scripts\python.exe -m commerce.services.merchant_api.platform_dataset --stores 6 --export commerce\deploy\var\platform_export
.venv\Scripts\python.exe -m commerce.deploy.run_local --merchants 6
```

| 판매자 | 가게 | 상품 | 고객 | 기간 | 역할 |
| --- | --- | --- | --- | --- | --- |
| merchant-1 | 제주 유기농 농장 | 40 | 80 | 90일 | FL 참여 |
| merchant-2 | 제주 바다 수산 | 40 | 70 | 90일 | FL 참여 |
| merchant-3 | 한라 베이커리 & 커피 | 38 | 85 | 90일 | FL 참여 |
| merchant-4 | 오이네 청과 | 40 | 75 | 90일 | FL 참여 |
| merchant-5 | 돌담 정육 | 40 | 70 | 90일 | FL 참여 |
| merchant-6 | 할망 반찬가게 | 20 | 40 | 30일 | 신규 판매자(라운드 불참, 공유 모델만 설치) |

- 로그인: 판매자 `owner-1`, 고객 `cust-001`~ (가게마다 따로), 비밀번호 모두 `demo-pass-1234`.
- 같은 명령은 어느 PC에서든 같은 데이터를 만든다: 끝 시각이 `2026-10-10T09:00Z`로 고정되고(`--now`로 바꿀 수 있음), 주문 ID를 주문 키에서 만든다. 그래서 export JSON과 `ids.snapshot_digest`가 PC마다 같다(#49 B 확인 반영).
- 고객마다 방문 주기(매주·격주·매달·한 번), 좋아하는 분류 1~2개, 단골 상품 3개가 있고, 가게별로 함께 사는 상품 쌍과 "다음 방문에 사는" 순서가 있다. 각 가게에는 기간 후반에 올라온 신상품 1개가 있다(B의 신상품 경로 확인용). 우유·계란·쌀은 여러 가게가 함께 판다.
- 소식 탭: 가게마다 게시물·릴스 8~18개(자동 생성 그래픽), 고객 취향에 따른 조회·좋아요·댓글, 판매자 답글. 같은 가게라도 로그인한 고객마다 탐색 순서가 다르다.
- 공동구매: 가게마다 구매자 제안 4건(성사 1·실패 1·모집 중 2).
- 직거래: 가게별 픽업 장소·시간과 택배비·무료배송 기준, 주문마다 픽업 또는 택배(지어낸 이름·연락처·주소, 완료된 택배는 송장번호), 일부 상품 재고(하나는 품절), 단골, 최근 3일 주문의 알림.
- `--export` 폴더의 `merchant-N.json`은 B `SellerInput` 모양이라, FL 실행기가 같은 이력을 그대로 읽을 수 있다(`summary.json`에 매장별 통계).
- 주의(D0025): 이 데이터는 A 화면 경로로 만든 것이라 C의 `fl_demo` 증명(attestation) 규칙상 그대로는 FL 코호트 입력이 아니다. `fl_demo`는 B의 g3 시나리오를 쓴다. 이 데이터로 라운드를 돌리려면 C가 `--dataset` 같은 입력 경로를 정해야 한다(팀 결정).

## 9. 학습된 모델 연결 (드라이브의 임시 모델, FL 없이)

B가 공유 드라이브에 올린 `models.zip`(Instacart 판매자 100곳, FL 500라운드: `ic100-T_lm-r500` text_only, `ic100-R_lm-r500` text_relation)을 판매자마다 설치하면 홈 추천이 인기순 대신 실제 모델로 나온다. 설치 명령은 B의 #53(`commerce.packages.recommender.install`)이다. torch가 있는 환경에서 실행한다.

```powershell
# 1) 받은 파일 확인 (models/ 폴더가 있는 곳에서)
sha256sum -c SHA256SUMS.txt
# 2) 인코더(MiniLM-L12, 고정 리비전) 내려받기
python -m commerce.evaluation.encoder_probe --download --only minilm-l12 --cache-dir commerce\evaluation\cache\encoders
# 3) 판매자마다 설치 (i = 1..6)
python -m commerce.packages.recommender.install --model-dir commerce\deploy\var\merchant_1\models --encoder-dir <인코더 폴더>
python -m commerce.packages.recommender.install --model-dir commerce\deploy\var\merchant_1\models --release-dir models\ic100-R_lm-r500 --variant text_relation
python -m commerce.packages.recommender.install --model-dir commerce\deploy\var\merchant_1\models --release-dir models\ic100-T_lm-r500 --variant text_only
```

- 확인: 판매자 개요의 "추천 모델" 칸이 "실제 모델로 추천 중"이 되고 설치된 두 버전이 보인다. 홈의 추천 칸에서 "임시" 배지가 사라진다.
- 모델 파일은 `models/` 아래에 있다. 주문 DB를 다시 만들어도(8단계) `models/`는 지우지 않아도 된다.
- 실제 runtime을 쓰면 판매자 하나가 뜨는 데 10초쯤 걸린다(torch·인코더 로딩). `run_local`이 판매자당 20초 안에 응답을 기다리므로 6곳을 한 번에 띄우면 "merchant health timeout"이 날 수 있다. 그때는 판매자 수를 줄여 띄우거나, C에게 대기 시간 조정을 요청한다(C 파일).
- 이 모델은 Instacart로 학습한 공통 모델이다. 데모 가게의 숫자(적중률 등)는 성능 근거로 쓰지 않는다(#49 B 의견).
