# 검증 근거

이 문서는 코드 구조와 검증 결과를 정리한다. 여기의 검증은 특정 Graph-FL 방법이 항상 더 좋다는 뜻이 아니다. 이 레포가 graph, control, diagnostics를 같은 기준으로 비교할 수 있는지 확인하는 절차다.

## 검증이 보는 것

| 항목 | 확인하는 내용 | 주요 산출물 |
|---|---|---|
| graph construction | component로 조립한 graph가 reference builder와 같은 결과를 내는가 | `graph_parity_summary.csv` |
| prior work mapping | pFedGraph, FedAMP, SFL, FedAGA의 핵심 구조를 component로 표현할 수 있는가 | `external_mechanism_alignment.csv` |
| diagnostics | synthetic case에서 metric이 예상한 방향으로 움직이는가 | `scenario_manifest.json`, `metric_validity_summary.csv` |
| composability | source, builder, target, control 조합이 명시적으로 통과하거나 제외되는가 | `composability_matrix.csv` |
| design space | built-in 조합이 같은 계산 조건을 통과하는가 | `design_space_matrix.csv`, `design_space_summary.csv` |
| extensibility | custom component가 trace와 artifact까지 이어지는가 | `extension_contract_summary.csv` |

## 요약 결과

| 검증 항목 | 결과 |
|---|---:|
| graph construction parity | 18 modes pass, max diff `2.21e-12`, edge F1 `1.0` |
| prior-work mapping | 5 / 5 rows pass |
| framework diagnostic rows | 60 rows pass |
| representative composability | 6 supported-pass, 2 unsupported-explicit |
| full design-space checks | 8,640 / 8,640 pass |
| extension contract | 4 / 4 rows pass |

## 해석 기준

| 결과 | 의미 |
|---|---|
| pass | 현재 contract와 계산 조건을 만족한다 |
| unsupported-explicit | 지원하지 않는 조합을 조용히 fallback하지 않고 명시적으로 제외한다 |
| needs-review | metric 반응이 약하거나 해석을 보류해야 한다 |

`needs-review` row는 실패를 숨기지 않기 위해 남긴다. 예를 들어 sample-prior collaboration의 일부 validation metric은 pass gate가 약해 별도로 표시한다.

## Graph Construction

| Metric | Result |
|---|---:|
| compared graph modes | 18 |
| non-pass rows | 0 |
| max absolute drift | `2.21e-12` |
| edge F1 | `1.0` |
| pass gate | `max_abs_diff <= 1e-9`, `edge_f1 = 1.0` |

이 check는 lifecycle로 조립한 graph가 기존 graph semantics를 바꾸지 않는지 확인한다. 이후 결과를 graph mechanism의 결과로 해석하려면 먼저 이 조건이 맞아야 한다.

## Prior Work Mapping

| Method | 현재 표현 | 범위 |
|---|---|---|
| pFedGraph | directed collaboration kernel, symmetric diagnostic projection | paper-kernel |
| FedAMP | model-distance attentive weighting kernel | paper-kernel |
| SFL | learned smoothness graph proxy | proxy-reference |
| FedAGA | EMA/update magnitude-aware relation proxy | proxy-reference |
| FED-PUB / GPFL | functional embedding, personalized target hook | interface-target |

이 매핑은 논문 전체를 그대로 재현했다는 뜻이 아니다. 이 레포 안에서 비교 가능한 component slot으로 어느 부분을 표현했는지 남기는 용도다.

## Design-Space Check

Full validation은 built-in 조합 전체를 계산한다.

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
| calculation checks passed | 8,640 |
| needs-review | 0 |
| unsupported-explicit | 0 |

각 row는 source output, adjacency, target output, control semantics, metadata, artifact row를 확인한다.

## 한계

| 항목 | 의미 |
|---|---|
| 성능 증명 아님 | 이 문서는 Graph-FL이 항상 baseline보다 좋다는 결론을 내리지 않는다 |
| proxy 범위 | 일부 prior work는 paper mechanism의 일부를 proxy로 표현한다 |
| real experiment 별도 | 실제 dataset에서의 성능 해석은 run output과 diagnostic artifact를 함께 봐야 한다 |
| 내부 profile 이름 | full validation CLI 옵션 이름은 현재 `--profile poster`다. 이름은 남아 있지만 의미는 전체 조합 검증이다 |

## 실행

빠른 확인:

```powershell
python scripts/validation/graph_evidence_report.py `
  --profile smoke `
  --include-external `
  --out-dir tmp/evidence_smoke
```

전체 조합 확인:

```powershell
python scripts/validation/graph_evidence_report.py `
  --profile poster `
  --include-external `
  --out-dir tmp/evidence_full
```

Primary implementation:

| Surface | Path |
|---|---|
| validation logic | `graphfl_lab/validation/graph_evidence.py` |
| report entry point | `scripts/validation/graph_evidence_report.py` |
| tests | `tests/validation/test_graph_evidence.py` |
