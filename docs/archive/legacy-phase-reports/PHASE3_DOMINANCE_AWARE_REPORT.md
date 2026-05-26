# Phase 3 Dominance-Aware Report

## 목적

Graph gain으로 보이는 변화가 dominant client correction으로 설명되는지 확인했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| partition | Dirichlet |
| method | dominance-aware baseline |
| metrics | accuracy, loss, update norm, `DI`, `N_eff` |

## 결과

| 관찰 | 의미 |
|---|---|
| dominance correction signal 확인 | graph-free control 필요 |
| graph smoothing과 dominance correction 비교 필요 | attribution design으로 연결 |

## 현재 연결

현재 설계는 `correction_family`에 graph-free dominance correction을 포함한다.
