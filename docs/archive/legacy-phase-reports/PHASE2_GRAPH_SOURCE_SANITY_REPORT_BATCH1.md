# Phase 2 Graph Source Sanity Report Batch 1

## 목적

`update`, `ema_update`, `classifier_head_update` graph source가 smoke setting에서 artifact와 metric을 생성하는지 확인했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| graph sources | `update`, `ema_update`, `classifier_head_update` |
| variants | update, random, identity |

## 결과

| 관찰 | 의미 |
|---|---|
| source별 실행 가능성 확인 | source contract 필요 |
| source별 metric 차이 확인 | source attribution 필요 |
| artifact 생성 경로 확인 | evidence artifact contract로 연결 |
