# Framework Validity Evidence

이 문서는 Graph-FL Design Lab가 단순 실행 코드가 아니라 Graph-FL gain attribution framework로 성립한다는 근거를 정리한다. 핵심은 Graph-FL claim을 같은 measurement vocabulary, matched controls, graph-free controls, diagnostic metrics, artifact contract로 검증 가능하게 만드는 것이다.

## Validity Claim

| 검증 축 | 주장 | 근거 artifact |
|---|---|---|
| graph construction | lifecycle assembly가 기존 reference graph semantics를 보존한다 | `graph_parity_summary.csv` |
| prior-work mechanism | pFedGraph, FedAMP, SFL, FedAGA의 핵심 mechanism을 component slot으로 표현한다 | `external_mechanism_alignment.csv` |
| diagnostics | synthetic pathology에서 metric이 expected direction으로 움직인다 | `scenario_manifest.json`, `metric_validity_summary.csv` |
| composability | source, mode, target, control 조합을 pass/unsupported/needs-review로 명시한다 | `composability_matrix.csv` |
| design space | built-in Cartesian product 전체가 calculation contract를 통과한다 | `design_space_matrix.csv`, `design_space_summary.csv` |
| extensibility | custom source, builder, preset, target이 trace와 artifact contract를 보존한다 | `extension_contract_summary.csv` |
| measurement integrity | real/control diagnostic row가 measured nonzero와 expected-zero control로 구분된다 | `real_diagnostic_consistency.csv` |

이 Evidence는 특정 Graph-FL method가 항상 우월하다는 주장이 아니다. 같은 client artifact를 real graph, matched controls, graph-free corrections, source/mode/target ablations, prior-work proxy mechanisms에 통과시켜 Graph-FL gain의 원인을 분해할 수 있다는 주장이다.

## Experiment Questions

| 질문 | 실험 handle | 필요한 Evidence |
|---|---|---|
| graph implementation이 faithful한가 | lifecycle-built graph와 reference builder 비교 | construction drift, edge support, edge F1 |
| paper mechanism이 정직하게 표현되는가 | source, relation, topology, aggregation, directionality, claim scope 매핑 | paper-kernel/proxy-reference rows |
| diagnostics가 의미 있게 반응하는가 | expected direction이 알려진 synthetic case 실행 | metric validity, pass/fail direction check |
| framework가 조합 가능한가 | source, mode, target, control, diagnostics 조합 | explicit supported, unsupported, needs-review verdict |
| design space가 조용히 깨지지 않는가 | built-in Cartesian product enumeration | row-level calculation checks |
| 사용자가 확장할 수 있는가 | custom source, builder, preset, target 등록 | trace, metadata, diagnostics, artifact preservation |
| real measurement가 해석 가능한가 | real/control diagnostic movement 분류 | measured nonzero, expected-zero control, needs-review |

## Evidence Summary

| Evidence Axis | 증명 내용 | Quantitative Verdict | Primary Artifact |
|---|---|---:|---|
| Construction drift | assembled lifecycle graph가 reference graph를 재현한다 | 18 graph modes pass, max abs diff `2.21e-12`, edge F1 `1.0` | `graph_parity_summary.csv` |
| Paper-mechanism alignment | pFedGraph, FedAMP, SFL, FedAGA mechanism이 framework component로 매핑된다 | 5 / 5 rows pass | `external_mechanism_alignment.csv` |
| Diagnostic sensitivity | diagnostics가 synthetic pathology에서 expected direction으로 움직인다 | 60 framework diagnostic rows pass | `scenario_manifest.json`, `metric_validity_summary.csv` |
| Composability | component 조합이 pass, explicit unsupported, needs-review로 분류된다 | 6 supported-pass, 2 unsupported-explicit | `composability_matrix.csv` |
| Design-space coverage | built-in 조합 전체가 calculation contract를 만족한다 | 8,640 / 8,640 checks pass | `design_space_matrix.csv`, `design_space_summary.csv` |
| Extensibility | custom source, builder, preset, target이 trace와 artifact까지 도달한다 | 4 / 4 extension rows pass | `extension_contract_summary.csv` |
| Measurement integrity | real diagnostic과 expected-zero control을 구분한다 | real/random/uniform measured nonzero, identity expected-zero control | `real_diagnostic_consistency.csv` |

## Pass Criteria

| Validation Unit | Pass Condition | Failure Signal |
|---|---|---|
| Graph reproduction | 같은 input에서 `max_abs_diff <= 1e-9`, `edge_f1 = 1.0` | weight drift 초과 또는 edge support mismatch |
| Mechanism alignment | reference type, source, provenance, matched component, claim scope 기록 | provenance 또는 claim scope 누락 |
| Synthetic sensitivity | scenario manifest의 expected direction과 row verdict 일치 | expected direction mismatch 또는 weak response |
| Composability | supported 조합이 graph, trace, diagnostics, artifact row를 생성 | silent fallback 또는 artifact 누락 |
| Design-space coverage | 모든 Cartesian product row가 calculation checks 통과 | source, graph, target, control, diagnostic, metadata check 실패 |
| Extensibility | custom component가 metadata와 artifact contract를 보존 | trace, metadata, diagnostics, artifact field 누락 |
| Measurement integrity | row가 measured nonzero, expected-zero control, needs-review 중 하나로 분류 | NaN, missing metric, unexpected zero movement |

## Construction Drift

| Metric | Result |
|---|---:|
| compared graph modes | 18 |
| non-pass rows | 0 |
| max absolute drift | `2.21e-12` |
| edge F1 | `1.0` |
| pass gate | `max_abs_diff <= 1e-9`, `edge_f1 = 1.0` |

해석:

| Result | Meaning |
|---|---|
| `2.21e-12` max drift | `1e-9` gate보다 충분히 작아 numeric round-off 수준이다 |
| `edge_f1 = 1.0` | edge topology가 reference와 동일하다 |
| 18 graph modes pass | lifecycle assembly가 graph semantics를 바꾸지 않는다 |

Construction drift는 framework 정당성의 첫 조건이다. componentized path가 기존 builder output을 numeric tolerance 안에서 재현하고 edge support를 보존해야 이후 attribution 결과를 graph mechanism의 결과로 해석할 수 있다.

## Paper-Mechanism Alignment

| Method | Matched Component | Reference Type | Directionality | Claim Scope |
|---|---|---|---|---|
| pFedGraph | directed collaboration kernel | paper-kernel | directed | collaboration matrix kernel |
| pFedGraph | symmetric diagnostic projection | paper-kernel | symmetric projection | diagnostic projection of collaboration graph |
| FedAMP | model-distance attentive weighting kernel | paper-kernel | undirected/symmetric | attentive relation kernel |
| SFL | learned smoothness graph proxy | proxy-reference | undirected | smoothness-driven graph proxy |
| FedAGA | EMA/update magnitude-aware relation proxy | proxy-reference | undirected | update-magnitude relation proxy |

Reference type 의미:

| Type | Meaning |
|---|---|
| `paper-kernel` | 논문 수식 또는 mechanism 설명을 독립 kernel로 구현한 row |
| `proxy-reference` | 논문 mechanism을 repository component로 대리 표현한 row |
| `interface-target` | slot과 hook이 정의되어 method family를 받아들일 수 있는 확장 surface |

## Diagnostic Sensitivity

| Metric Family | Verdict | Count |
|---|---|---:|
| framework_diagnostic | pass | 60 |
| framework_diagnostic | not-applicable | 15 |
| validation_metric | pass | 235 |
| validation_metric | needs-review | 15 |

Needs-review row는 숨기지 않고 별도 verdict로 남긴다.

| Scenario | Metric | Family | Min Pass Rate | Min Rho | Verdict |
|---|---|---|---:|---:|---|
| sample_prior_collaboration | directed_row_similarity_to_ground_truth | validation_metric | 1.00 | 0.45 | needs-review |

Diagnostic sensitivity는 framework-level diagnostics와 validation metrics로 나뉜다. Framework diagnostics는 measurement vocabulary가 expected direction으로 움직이는지를 확인한다. Validation metrics는 더 넓은 synthetic check를 제공하고, 약한 row를 `needs-review`로 분리해 claim boundary를 명확하게 만든다.

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

| Check Group | Required Fields |
|---|---|
| source output | vector count, dimension, finite values, client variation |
| adjacency | shape, finite values, symmetry, zero diagonal, non-negative weights |
| graph diagnostics | numeric validity, density, edge count |
| target output | output shape, finite values, graph-filtered target diagnostics |
| controls | identity, uniform, random, shuffled semantics |
| metadata | target metadata, artifact fields, trace identity |

8,640-row check는 framework가 한 preset에만 맞춰진 구조가 아니라는 근거다. 모든 row가 source output quality, graph construction validity, target output validity, control semantics, metadata를 같은 기준으로 통과한다.

## Extensibility

| Extension Point | Example | Verdict |
|---|---|---|
| `graph_source` | `evidence_unit_source` | pass |
| `graph_builder` | `evidence_unit_builder` | pass |
| `design_preset` | `evidence_unit_design` | pass |
| `aggregation_target` | `v1_core_code_extension_point` | pass |

Extensibility Evidence는 custom component가 독립 실행에 그치지 않고 built-in component와 같은 trace, metadata, diagnostic, artifact pipeline을 통과하는지 확인한다.

## Artifact Contract

| Artifact | Required Information |
|---|---|
| `round_metrics.csv` | pre/post aggregate, `DI`, `N_eff`, alignment, `LOO` |
| `client_metrics.csv` | client contribution, update norm, alignment |
| `graph_stats.csv` | density, degree, entropy, spectral metrics |
| `counterfactual_metrics.csv` | real graph and control graph gap |
| `metric_validity_summary.csv` | synthetic expected-direction result |
| `design_space_matrix.csv` | source/mode/target/control/diagnostic row validity |
| `extension_contract_summary.csv` | custom component trace and artifact preservation |

CSV/JSON artifact는 reproducible output이다. Repository 문서는 row meaning, pass criteria, quantitative verdict, provenance, reproduction command를 canonical contract로 남긴다.

## Provenance

| Artifact | Role |
|---|---|
| `poster_tables.md` | compact poster summary |
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

## Reproduction

Repository root에서 실행한다.

```text
python scripts/validation/graph_evidence_report.py --profile poster --include-external --out-dir <out-dir> --real-suite-dir <real-suite-dir>
```

Primary implementation:

| Surface | Path |
|---|---|
| validation logic | `graphfl_lab/validation/graph_evidence.py` |
| report entry point | `scripts/validation/graph_evidence_report.py` |
| tests | `tests/validation/test_graph_evidence.py` |
