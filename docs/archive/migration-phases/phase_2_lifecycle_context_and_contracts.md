# Phase 2 Lifecycle Context And Contracts

## 목적

Lifecycle module이 공유할 context와 return contract를 정의했다.

## 범위

| 영역 | 내용 |
|---|---|
| context | round, client, state, config 정보 |
| contract | module input/output boundary |
| support status | unsupported/interface-target 상태 표현 |
| compatibility | 기존 `graph_source`, `graph_mode`, `aggregation_target` 연결 준비 |

## 결과

| Result | 의미 |
|---|---|
| module boundary 확정 | relation, topology, aggregation 분리 |
| unsupported status 표준화 | silent fallback 방지 |
| Phase 3 준비 | design composer와 registry로 연결 |
