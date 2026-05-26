# Phase 2.5 Smoothing Failure Analysis

## 목적

Graph smoothing path가 real relation signal 없이도 성능 변화를 만들 수 있는지 분석했다.

## 결과

| 관찰 | 의미 |
|---|---|
| smoothing-only effect 존재 | real graph와 generic smoothing control 분리 필요 |
| random/uniform control 경쟁력 확인 | matched control 설계 필요 |
| graph quality 민감도 확인 | graph construction validation 필요 |

## 현재 연결

현재 evidence pack은 `real`, `random`, `shuffled`, `uniform`, `identity`, `graph-free` control을 분리해 기록한다.
