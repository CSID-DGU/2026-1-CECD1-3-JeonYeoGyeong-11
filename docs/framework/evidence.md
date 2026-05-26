# Graph-FL Evidence

## Framework-Quality Claim

Graph-FL 프레임워크는 기존 graph builder를 수치적으로 재현하고, built-in graph-design 조합 8,640개를 전수 확인했으며, 논문식 또는 논문 유도 mechanism과의 대응 범위를 row 단위로 기록하고, 새로운 조합과 custom extension이 metadata, diagnostics, trace, artifact 계약을 유지하는지 검증된 조립식 graph-construction framework다.

## Evidence Summary

| 축 | 주장 | 핵심 수치 | Artifact |
|---|---|---:|---|
| Construction drift | 조립식 lifecycle graph가 기존 내부 reference graph를 재현한다 | 18 graph modes pass, max abs diff `2.21e-12`, edge F1 `1.0` | `graph_parity_summary.csv` |
| Paper-mechanism alignment | pFedGraph, FedAMP, SFL, FedAGA component가 paper mechanism과 대응된다 | 5 / 5 rows pass | `external_mechanism_alignment.csv` |
| Diagnostic sensitivity | diagnostics가 synthetic pathology에서 expected direction으로 반응한다 | framework diagnostic 60 pass | `scenario_manifest.json`, `metric_validity_summary.csv` |
| Composability | 대표 component 조합이 pass, explicit unsupported, needs-review로 분류된다 | 6 supported-pass, 2 unsupported-explicit | `composability_matrix.csv` |
| Design-space coverage | built-in graph-design 조합 전체가 계산 계약을 통과한다 | 8,640 / 8,640 calculation checks pass | `design_space_matrix.csv`, `design_space_summary.csv` |
| Extensibility | custom source, builder, preset이 trace와 artifact까지 이어진다 | 4 / 4 extension rows pass | `extension_contract_summary.csv` |
| Measurement integrity | real diagnostic 값의 측정 상태를 분류한다 | real/random/uniform measured nonzero, identity expected-zero control | `real_diagnostic_consistency.csv` |

## Validation Logic

| 검증 단위 | Pass 조건 | Failure signal |
|---|---|---|
| Graph 재현성 | 같은 입력에서 `max_abs_diff <= 1e-9`, `edge_f1 = 1.0` | weight drift 초과, edge support mismatch |
| Mechanism 대응 | row마다 reference type, source, provenance, matched component, claim scope 기록 | provenance 누락, claim scope 누락 |
| Synthetic sensitivity | expected direction을 `scenario_manifest.json`에서 읽고 row 단위로 판정 | expected direction mismatch, weak response row |
| Composability | supported 조합은 graph, trace, diagnostics, artifact row 생성 | silent fallback, artifact 누락 |
| Design-space coverage | Cartesian product 전체 row가 calculation checks 통과 | source, graph, target, control, diagnostic, metadata check 실패 |
| Extensibility | custom component가 metadata와 artifact 계약 유지 | trace, metadata, diagnostics, artifact 단절 |
| Measurement integrity | row를 measured nonzero, expected-zero control, needs-review로 분류 | NaN, missing metric, unexpected zero movement |

## Construction Drift

| Metric | Result |
|---|---:|
| compared graph modes | 18 |
| non-pass rows | 0 |
| max absolute drift | `2.21e-12` |
| edge F1 | `1.0` |
| pass gate | `max_abs_diff <= 1e-9`, `edge_f1 = 1.0` |

오차 해석:

- 최대 오차 `2.21e-12`는 gate `1e-9` 대비 약 450배 작다.
- 모든 row에서 edge F1 `1.0`이므로 graph topology는 동일하다.
- 남은 weight 차이는 normalization, sparsification, floating-point 연산 순서에서 생기는 numeric round-off 범위다.

## Paper-Mechanism Alignment

| Method | Matched Component | Reference Type | Directionality | Claim Scope |
|---|---|---|---|---|
| pFedGraph | directed collaboration kernel | paper-kernel | directed | collaboration matrix kernel |
| pFedGraph | symmetric diagnostic projection | paper-kernel | symmetric projection | diagnostic projection of collaboration graph |
| FedAMP | model-distance attentive weighting kernel | paper-kernel | undirected/symmetric | attentive relation kernel |
| SFL | learned smoothness graph proxy | proxy-reference | undirected | smoothness-driven graph proxy |
| FedAGA | EMA/update magnitude-aware relation proxy | proxy-reference | undirected | update-magnitude relation proxy |

## Diagnostic Sensitivity

| Metric Family | Verdict | Count |
|---|---|---:|
| framework_diagnostic | pass | 60 |
| framework_diagnostic | not-applicable | 15 |
| validation_metric | pass | 235 |
| validation_metric | needs-review | 15 |

기록된 row:

| Scenario | Metric | Family | Min Pass Rate | Min Rho | Verdict |
|---|---|---|---:|---:|---|
| sample_prior_collaboration | directed_row_similarity_to_ground_truth | validation_metric | 1.00 | 0.45 | needs-review |

## Design-Space Coverage

```text
16 graph sources
x 18 graph modes
x 5 aggregation targets
x 6 correction profiles
= 8,640 combinations
```

| Status | Count |
|---|---:|
| supported-pass | 8,640 |
| calculation_checks_passed | 8,640 |
| needs-review | 0 |
| unsupported-explicit | 0 |

Row-level checks:

- source vector count, dimension, finite values, client variation
- adjacency shape, finite values, symmetry, zero diagonal, non-negative weights
- graph diagnostics numeric validity
- aggregation target output shape and finite values
- graph-filtered target diagnostics
- identity, uniform, random, shuffled control semantics
- graph density, edge count, target metadata artifact fields

## Extensibility

| Extension Point | Example | Verdict |
|---|---|---|
| `graph_source` | `evidence_unit_source` | pass |
| `graph_builder` | `evidence_unit_builder` | pass |
| `design_preset` | `evidence_unit_design` | pass |
| `aggregation_target` | `v1_core_code_extension_point` | pass |

## Provenance

| Artifact | 역할 |
|---|---|
| `poster_tables.md` | poster용 compact summary |
| `claim_boundaries.md` | claim scope table |
| `scenario_manifest.json` | scenario, seed, expected direction, pass rule |
| `graph_parity_summary.csv` | construction drift rows |
| `external_mechanism_alignment.csv` | paper-mechanism alignment rows |
| `metric_validity_summary.csv` | synthetic diagnostic validity rows |
| `composability_matrix.csv` | representative composition rows |
| `design_space_matrix.csv` | built-in design-space rows |
| `design_space_summary.csv` | design-space aggregate |
| `design_space_boundaries.csv` | design-space axis boundary |
| `extension_contract_summary.csv` | custom extension checks |
| `real_diagnostic_consistency.csv` | real/control measurement status |
| `validation_verdict.json` | final verdict |

Scenario manifest hash:

```text
b277f3885562d9074301a0e7f2f4d7afc01814f0afda5f75b5bf878fed345215
```

## Reproduction Entry Point

```text
python scripts/validation/graph_evidence_report.py --profile poster --include-external --out-dir <out-dir> --real-suite-dir <real-suite-dir>
```

Primary implementation:

- `graphfl_lab/validation/graph_evidence.py`
- `scripts/validation/graph_evidence_report.py`
- `tests/validation/test_graph_evidence.py`
