# Graph Algorithm Extension Guide

이 문서는 `docs/framework/components.md`로 통합되었다.

핵심 내용:

- 새 graph algorithm은 `graph_source`, `graph_builder`, `GraphFLDesign`, `aggregation_target` 조합으로 추가한다.
- custom source, builder, preset은 trace, metadata, diagnostics, artifact contract를 통과해야 한다.
- extension test는 graph, design, lifecycle, diagnostics, validation 영역에 둔다.

Canonical:

- `docs/framework/components.md`
- `docs/framework/evidence.md`
