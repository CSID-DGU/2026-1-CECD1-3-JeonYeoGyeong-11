# Phase 2 Graph Informativeness: n20 alpha 0.1 signed seed42

## 목적

20-client, Dirichlet alpha 0.1 setting에서 signed conflict graph의 informativeness를 확인했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| clients | 20 |
| Dirichlet alpha | 0.1 |
| graph preset | `signed_conflict_knn` |
| variants | real/update, random, shuffled, uniform, identity |

## 결과

| 관찰 | 의미 |
|---|---|
| control 대비 real graph advantage 약함 | stronger graph validation 필요 |
| seed별 변동 확인 | multi-seed summary 필요 |
| graph/control 분리 필요 | current evidence pack 설계로 연결 |
