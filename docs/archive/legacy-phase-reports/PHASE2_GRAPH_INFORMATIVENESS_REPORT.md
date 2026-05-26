# Phase 2 Graph Informativeness Report

## 목적

Update 기반 graph가 random, shuffled, uniform, identity control 대비 informative한지 초기 평가했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| partition | Dirichlet |
| clients | 5 |
| graph source | classifier head update |
| variants | update, random, shuffled, uniform, identity |

## 결과

| 관찰 | 의미 |
|---|---|
| control graph 경쟁력 확인 | graph-specific explanation 분리 필요 |
| seed variation 확인 | multi-seed evidence 필요 |
| update graph 민감도 확인 | graph source와 topology 검증 필요 |

## 현재 연결

현재 framework-quality evidence는 `graph_parity_summary.csv`, `metric_validity_summary.csv`, `design_space_matrix.csv`로 graph construction과 diagnostics를 검증한다.
