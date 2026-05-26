# Phase 6 Aggregation Delivery State Hooks

## 목적

Aggregation target, delivery policy, state store, local objective hook의 extension point를 정리했다.

## 범위

| 영역 | 내용 |
|---|---|
| aggregation target | update, EMA update, weight, graph-filtered target |
| delivery policy | global/personalized delivery hook |
| state store | EMA, previous graph, accumulated signal |
| local objective | proximal/local hook surface |

## 결과

| Result | 의미 |
|---|---|
| aggregation target 확장 | graph 적용 위치 분리 |
| personalized method slot 정의 | paper mechanism mapping 가능 |
| Phase 7 준비 | validation and docs로 연결 |
