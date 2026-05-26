# Phase 2 Graph Informativeness: signed conflict kNN

## 목적

`signed_conflict_knn` graph preset이 control graph 대비 informative한지 평가했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| graph preset | `signed_conflict_knn` |
| graph mode | dense |
| variants | real/update, random, shuffled, uniform, identity |

## 결과

| 관찰 | 의미 |
|---|---|
| update graph effect 제한적 | source/mode attribution 필요 |
| control graph와의 비교 중요 | matched control suite 필요 |
| graph construction 검증 필요 | current graph evidence pack으로 연결 |
