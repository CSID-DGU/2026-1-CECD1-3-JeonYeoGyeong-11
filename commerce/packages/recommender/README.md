# 로컬 추천 runtime · B

시작: `runtime.py`의 `open_runtime`. 공통 명세는 `commerce.packages.contracts.ports.RecommenderRuntime`이다.
`open_runtime`은 실제 runtime `seller_runtime.SellerRuntime`을 돌려준다.

## 판매자 runtime
- `feature_store.py`: 판매자 특징 원장(features.sqlite). 구매 이벤트는 ID·본문으로 한 번만 반영하고 다른 본문은 DUPLICATE_EVENT, 카탈로그는 source_seq 역순을 무시하고 같은 순번·다른 본문을 거부한다. 반영과 feature_epoch 증가가 한 트랜잭션이며, 지난 epoch의 snapshot을 다시 읽을 수 있다.
- `serving.py`: 서비스 모델 등록(architecture_version 1 = text_only = harex.T_lm.v1, 2 = text_relation = harex.R_lm.v1), manifest 생성, `shared.<키>` 텐서 이름, 정규 npz(같은 텐서 → 같은 바이트), release 폴더 읽기·쓰기, 설치된 frozen 인코더(`FrozenText`).
- `seller_runtime.py`: A의 ingest·upsert·predict_local(fallback 순서는 interfaces.md §4), C의 get_shared_manifest·export_shared_state·get_local_data_ref·train_round·install_release.
  - 모델 파일: `{model_dir}/base/{variant}/{model_version}/`(release.json·manifest.json·weights.npz)와 `CURRENT`. 검증을 모두 통과한 release만 임시 폴더→이름 바꾸기→CURRENT 순으로 반영하므로 실패하면 이전 base가 계속 서빙한다.
  - frozen 인코더는 B 설치물로 `{model_dir}/frozen_text/`에 둔다(model-lab.md §6.6). 없으면 모델을 쓰지 않고 fallback만 한다.
  - Instacart처럼 상대시간인 과거 원장은 달력이 없으므로 어떤 live as_of보다도 앞선 이력으로 본다.
  - train_round의 검증 고객(10명 중 1명)은 seller와 round_config.seed로 정하고 round_id는 쓰지 않는다. 그래서 seed를 라운드마다 바꾸면 검증 고객도 바뀐다. C coordinator는 한 실행 동안 seed를 고정한다.
  - install_release는 받은 manifest의 hash를 이 패키지가 기대하는 manifest(설치된 인코더의 text_artifact_hash와 preprocessing_version 포함)와 비교한다. 다른 인코더나 전처리로 만든 release는 MANIFEST_MISMATCH다.
  - personalize_local: 설치된 base 복사본에서 query_proj·scorer만 학습하고, 고정 검증 고객의 손실이 base보다 낮을 때만 `{model_dir}/personal/{variant}/{base_version}/{revision}/`에 두고 쓴다. base가 바뀌면 옛 결과는 붙이지 않는다. 설정 기본값은 `DEFAULT_PERSONAL`(두 variant 공통)이다.
  - 미리 계산(`open_runtime`은 `warm=True`로 연다): 백그라운드 스레드가 현재 epoch의 상품 텍스트 벡터 z, 원장 전체 관계, 설치된 base의 e를 계산해 둔다. 여는 때와 새 구매·새 카탈로그 버전·release 설치 뒤에 다시 돌고, 연달아 온 변경은 0.3초 모아 한 번에 처리한다. 그 사이 온 추천 요청은 같은 계산을 반복하지 않고 끝나기를 기다린다. now보다 뒤 시각의 이벤트가 있으면 z만 계산하고 나머지는 요청의 as_of로 계산한다. `wait_warm()`은 따라잡았는지 기다리고, `close()`는 스레드를 멈춘다(앱 종료 때 부르지 않아도 daemon이라 남지 않는다).
  - compare_local: snapshot·후보·두 variant의 handle을 한 번 고정하고 T-G·R-G·T-P·R-P를 채우거나 불가 사유(model_not_ready, personalization_not_ready, insufficient_data, validation_rejected, base_mismatch)를 준다.

## 모델 core
학습·평가·서비스가 같은 코드를 쓴다.
- `text_encoder.py`: frozen 텍스트 인코더(평균 pooling·L2·text_artifact_hash). `z_cache.py`: 판매자 로컬 z cache(SQLite, key = artifact·preprocessing·텍스트 해시)
- `examples.py`: "이전 방문 → 다음 방문 상품 집합" 예제와 서빙 질의(`query_example`). `replay.py`: Instacart 진행률 replay(판매자가 target 시점에 볼 수 있는 구매)
- `relations.py`: 판매자 로컬 상품 관계 snapshot(고객·장바구니·방향별 시간)
- `model.py`: 초기 설계의 두 variant(text_only 6개·text_relation 9개 공유 그룹)와 OQ01 손실. `training.py`: 판매자·snapshot별 배치 학습·고정 검증 손실·전체 후보 점수
- `harex.py`: D0022 비교용 HAREX(GCI)식 공통 뼈대(1층 Transformer)와 네 상품 표현 T_hx·R_hx·T_lm·R_lm. 서비스는 이 중 lm 두 가지를 쓴다. hx 단어 토큰 표는 판매자 로컬이라 공유·집계하지 않는다

b2 selfcheck는 작은 무작위 BERT와 가짜 인코더로 CI에서 돈다.
`UnimplementedRuntime`은 scaffold가 검사하는 기준 stub이므로 고치거나 지우지 않는다([개발 안내](../../../docs/development.md)의 scaffold 절).
작업 순서는 [B 카드](../../../docs/team/tasks/B.md)를 따른다.
A/C는 같은 runtime을 사용한다. API 시그니처/반환 객체 변경은 공통 ports/types와 호출자·검사를 함께 수정한다. 실험 core는 웹 앱을 import하지 않는다.
