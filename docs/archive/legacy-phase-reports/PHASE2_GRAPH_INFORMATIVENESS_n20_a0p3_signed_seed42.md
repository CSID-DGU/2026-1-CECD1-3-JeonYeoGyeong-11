# Phase 2 Graph Informativeness: n20 alpha 0.3 signed seed42

## 목적

20-client, Dirichlet alpha 0.3 setting에서 signed conflict graph의 informativeness를 확인했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| clients | 20 |
| Dirichlet alpha | 0.3 |
| graph preset | `signed_conflict_knn` |
| variants | real/update, random, shuffled, uniform, identity |

## 결과

| 관찰 | 의미 |
|---|---|
| graph-control separation 약함 | graph quality diagnostic 필요 |
| alpha 변화에 따른 sensitivity 확인 | Non-IID stress calibration 필요 |
| control family 유지 필요 | current design의 `correction_family`로 연결 |
