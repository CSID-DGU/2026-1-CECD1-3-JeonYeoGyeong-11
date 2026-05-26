# Semantic Client Graph Roadmap Archive

## 목적

Semantic client graph 방향의 초기 roadmap을 보관한다.

## Roadmap 요약

| Idea | Current Mapping |
|---|---|
| richer client representation | `graph_source` extension |
| semantic relation score | `graph_builder` / `graph_mode` extension |
| graph smoothing | `graph_filtered_*` aggregation target |
| control comparison | `correction_family` |
| metric interpretation | diagnostics and evidence pack |

## 전환

| 이전 표현 | 현재 표현 |
|---|---|
| semantic graph | graph source + relation mechanism |
| smoothing gain | control-specific diagnostic gap |
| raw update limitation | source/mode/target attribution |
| roadmap item | framework extension contract |

## 현재 연결

현재 graph design 범위는 `docs/framework/components.md`와 `docs/framework/evidence.md`에 정리한다.
