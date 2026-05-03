# 실험 결과

Smoke experiment 결과와 현재 해석을 기록한다. raw output file은 로컬 `experiments_current/` 아래에만 두고 Git에는 포함하지 않는다.

## 현재 상태

- 실험 질문은 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)에 정리
- projection, update graph, spectral diagnostic, aggregation, strategy module 분리 완료
- 실행 구조를 Flower `ClientApp`/`ServerApp` 기반 App runner로 정리
- 반복 실험 조건을 `configs/*.json`로 실행 가능
- graph source와 aggregation target을 CLI/config에서 분리 가능
- `configs/general_extension_smoke.json`로 확장 옵션 smoke suite 실행 가능
- dependency 범위는 `requirements.txt`와 `pyproject.toml`에 기록
- `tests/`와 GitHub Actions smoke CI로 graph/spectrum/aggregation/config 최소 회귀 확인
- 새 실험 output은 로컬 `experiments_current/` 아래에 저장
- 현재 해석: client update dynamics를 graph-spectral 관점에서 분석하는 방향은 추가 실험 가치가 있으나, performance claim은 아직 확립되지 않음

## 확장 옵션 Smoke

목적:

- `--graph-source`와 `--aggregation-target` 분리 경로 확인
- 신규 graph mode (`mutual_knn`, `magnitude`, `global_alignment`) 실행 경로 확인
- suite summary와 decomposition export에 graph/aggregation 설정이 남는지 확인

명령어:

```powershell
python run_general_suite.py --config configs/general_extension_smoke.json
```

상태: 2026-05-04 완료.

출력:

```text
experiments_current/extension_option_smoke
```

확인:

- `ours_weight_graph`, `ours_magnitude`, `ours_mutual_knn_k1`, `ours_weight_agg` 실행 성공
- `general_suite_summary.csv`에 `graph_mode`, `graph_source`, `graph_source_used`, `aggregation_target`, `aggregation_target_used` 컬럼 저장
- `scripts/spectral_decomposition_report.py` export 성공
- Cora/FGL direct smoke도 `--graph-source weight --graph-mode global_alignment --aggregation-target weight` 조합으로 통과

## 실행 구조 점검

목적:

- 기본 실행 경로에서 deprecated `start_simulation()` 제거
- Flower App component가 기존 result JSON 형식을 그대로 저장하는지 확인
- 기존 `scripts/spectral_decomposition_report.py`와 호환성 확인

명령어:

```powershell
python run_general_experiment.py `
  --method ours `
  --dataset fashionmnist `
  --model mlp `
  --num-clients 2 --rounds 1 --local-epochs 1 `
  --batch-size 32 `
  --partition dirichlet --dirichlet-alpha 0.1 `
  --train-subset-size 200 --test-subset-size 100 `
  --graph-mode knn --knn-k 1 `
  --warmup-rounds 0 `
  --diagnostic-only true `
  --out-dir experiments_current/app_runner_smoke_v2 `
  --run-tag app_runner_smoke_v2
```

상태: 2026-05-03 완료.

출력:

```text
experiments_current/app_runner_smoke_v2/result_general_ours_seed42_app_runner_smoke_v2.json
```

확인:

- result JSON 저장 성공
- frequency decomposition export 성공
- `start_simulation()` deprecation warning 없음
- Ray/torch-geometric 외부 dependency warning만 남음
- 같은 App runner로 Cora/FGL 2-client 1-round smoke 통과

## Phase A. Diagnostic Smoke

목적:

- graph/spectral trace 기록 여부 확인
- FedAvg, dense, kNN, random-matched, uniform graph variant 비교
- `--diagnostic-only true`로 aggregation은 FedAvg와 동일하게 유지

명령어:

```powershell
python run_general_suite.py `
  --dataset fashionmnist `
  --model mlp `
  --num-clients 5 --rounds 3 --local-epochs 1 `
  --seeds 42 `
  --partition dirichlet --dirichlet-alpha 0.1 `
  --train-subset-size 1000 --test-subset-size 300 `
  --variants fedavg ours_dense ours_knn_k2 ours_random_matched_k2 ours_uniform `
  --diagnostic-only true `
  --warmup-rounds 0 `
  --out-dir experiments_current/phaseA_diagnostic_smoke
```

상태: 2026-05-03 완료.

출력:

```text
experiments_current/phaseA_diagnostic_smoke
```

요약:

| variant | final acc | mean delta vs FedAvg | mean H_spec | mean H_spec raw current | mean graph density | mean raw current density |
|---|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.2533 | 0.0000 | n/a | n/a | n/a | n/a |
| dense | 0.2533 | 0.0000 | 0.1965 | 0.1305 | 0.4667 | 0.3000 |
| kNN k=2 | 0.2533 | 0.0000 | 0.1835 | 0.1305 | 0.5000 | 0.3000 |
| random matched k=2 | 0.2533 | 0.0000 | 1.4420 | 1.3230 | 0.5333 | 0.3000 |
| uniform | 0.2533 | 0.0000 | 4.6078 | 4.6078 | 1.0000 | 1.0000 |

해석:

- accuracy delta 0은 의도된 결과. diagnostic-only에서는 aggregation weight가 FedAvg와 동일
- graph choice에 따라 `H_spec`가 크게 달라짐
- kNN/dense는 density-matched random과 uniform보다 훨씬 smooth
- kNN/random의 raw current density는 0.3으로 맞춰져 있으나, random-matched는 훨씬 큰 `H_spec`를 보임
- graph construction이 spectral structure를 바꾼다는 1차 신호. performance claim은 아님

## Phase B. Frequency Decomposition Smoke

목적:

- spectral residual 기반 aggregation 적용
- graph Fourier low/mid/high band energy 기록
- client-level low/mid/high component norm ratio export
- high-frequency client가 label skew 또는 boundary client와 맞물리는지 확인

구현 변경:

- `mid_frequency_energy_ratio` 추가
- `high_to_low_energy_ratio` 추가
- dominant spectral mode diagnostic 추가
- client-level component norm ratio 추가:
  - `low_frequency_component_norm_ratio_list`
  - `mid_frequency_component_norm_ratio_list`
  - `high_frequency_component_norm_ratio_list`
- `scripts/spectral_decomposition_report.py` 추가
- client-level decomposition을 `client_class_distribution`과 맞출 수 있도록 client id logging 수정

명령어:

```powershell
python run_general_suite.py `
  --dataset fashionmnist `
  --model mlp `
  --num-clients 5 --rounds 3 --local-epochs 1 `
  --seeds 42 `
  --partition dirichlet --dirichlet-alpha 0.1 `
  --train-subset-size 1000 --test-subset-size 300 `
  --variants fedavg ours_dense ours_knn_k2 ours_random_matched_k2 ours_uniform `
  --warmup-rounds 0 `
  --conflict-mix 0.2 `
  --min-client-weight 0.05 `
  --out-dir experiments_current/phaseB_frequency_decomp_smoke_v2
```

상태: 2026-05-03 완료.

출력:

```text
experiments_current/phaseB_frequency_decomp_smoke_v2
```

요약:

| variant | final acc | delta vs FedAvg | mean H_spec | low | mid | high | high/low | effective clients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.2533 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a |
| kNN k=2 | 0.2567 | +0.0033 | 0.2112 | 0.1181 | 0.6582 | 0.2237 | 2.6007 | 4.6904 |
| dense | 0.2533 | 0.0000 | 0.1412 | 0.1693 | 0.7033 | 0.1275 | 1.2203 | 4.6757 |
| random matched k=2 | 0.2467 | -0.0067 | 1.3670 | 0.1025 | 0.6293 | 0.2682 | 2.9212 | 4.6583 |
| uniform | 0.2433 | -0.0100 | 4.6089 | 0.0782 | 0.6574 | 0.2644 | 3.3989 | 4.5310 |

client-level kNN k=2 frequency 요약:

| client | high component | residual e | mean alpha | label histogram |
|---:|---:|---:|---:|---|
| 1 | 0.4368 | 0.5311 | 0.1428 | [0, 27, 0, 1, 0, 0, 24, 82, 8, 0] |
| 4 | 0.4026 | 0.4420 | 0.2661 | [85, 81, 2, 1, 1, 1, 71, 23, 1, 1] |
| 3 | 0.3160 | 0.4086 | 0.1379 | [0, 0, 52, 81, 0, 0, 1, 0, 0, 4] |
| 0 | 0.2899 | 0.2977 | 0.2154 | [0, 0, 13, 5, 0, 95, 0, 0, 96, 8] |
| 2 | 0.1942 | 0.2535 | 0.2377 | [0, 0, 43, 0, 100, 0, 1, 0, 0, 92] |

아이디어 관점 해석:

- 추가 실험할 가치가 있는 방향
- 현재 가장 강한 신호는 accuracy improvement가 아니라, client update가 graph-dependent spectral structure를 보인다는 점
- dense는 가장 smooth한 graph지만, 이 작은 smoke run에서 가장 좋은 variant는 아님
- kNN은 dense보다 high-frequency disagreement를 더 보존하면서도 이 smoke run에서는 가장 좋은 결과
- random-matched는 sparsity 조건은 비슷하지만 spectral structure가 훨씬 거칠고 accuracy도 낮음
- uniform은 모든 client를 같은 relation graph 안에 묶으며 가장 큰 `H_spec`와 high/low 값을 만듦
- 유용한 주장: client update graph construction은 부수적인 구현 디테일이 아니라, shared/disagreement component를 해석하는 frequency basis를 바꾸는 선택

현재 confidence:

```text
강한 1차 신호:
  graph choice가 spectral structure를 바꾼다.

중간 정도의 1차 신호:
  similarity-aware sparse graph가 random sparse graph보다 의미 있을 수 있다.

아직 약한 주장:
  spectral aggregation이 accuracy를 개선한다.

아직 답하지 못한 질문:
  high-frequency client가 harmful conflict인지 useful heterogeneity인지.
```

## 다음 실험

Phase B를 더 많은 seed와 k 값으로 반복한다.

```text
seeds: 42, 43, 44
k: 1, 2, 3
variants:
  fedavg
  ours_dense
  ours_knn_k1 / k2 / k3
  ours_random_matched_k1 / k2 / k3
  ours_uniform
```

질문:

```text
1. kNN은 random-matched와 일관되게 다른가?
2. high-frequency component는 label skew와 상관이 있는가?
3. high-frequency component는 alpha suppression으로 이어지는가?
4. high-frequency는 harmful conflict인가, useful heterogeneity인가?
5. 가장 좋은 graph가 반드시 가장 smooth한 graph인가?
```
