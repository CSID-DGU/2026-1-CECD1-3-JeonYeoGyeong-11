# Phase 1 Diagnostics Report

## 목적

Graph-FL diagnostic metric이 Non-IID setting에서 의미 있는 신호를 보이는지 초기 점검했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| partition | Dirichlet |
| clients | 5 |
| seeds | 42, 43, 44 |
| metrics | update norm, `DI`, `N_eff`, alignment, `LOO` |

## 결과

| 관찰 | 의미 |
|---|---|
| client contribution imbalance 확인 | dominance diagnostic 필요 |
| seed별 variation 확인 | multi-seed summary 필요 |
| interaction pathology 신호 확인 | Phase 2 graph informativeness 평가로 연결 |

## 현재 연결

현재 diagnostic 기준은 `docs/framework/metrics.md`를 따른다.
