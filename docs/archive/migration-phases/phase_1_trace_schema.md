# Phase 1 Trace Schema

## 목적

Lifecycle component가 공통으로 남길 trace schema를 정의했다.

## 범위

| 영역 | 내용 |
|---|---|
| trace record | stable top-level field |
| design identity | `TraceRecord.values`에 component metadata 기록 |
| support level | core, proxy, interface status 기록 |
| compatibility | 기존 diagnostics writer와 연결 |

## 결과

| Result | 의미 |
|---|---|
| trace vocabulary 표준화 | component claim과 artifact 연결 |
| design-space key 도입 | source, relation, topology, target 기록 |
| Phase 2 준비 | lifecycle context contract로 연결 |
