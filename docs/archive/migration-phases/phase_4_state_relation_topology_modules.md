# Phase 4 State Relation Topology Modules

## 목적

Client state extraction, relation estimation, topology construction을 module로 분리했다.

## 범위

| 영역 | 내용 |
|---|---|
| state envelope | update, weight, embedding, history signal 표현 |
| relation module | cosine, RBF, QP 등 relation score |
| topology module | dense, kNN, threshold, control graph |
| metadata | graph kind, density, edge count 기록 |

## 결과

| Result | 의미 |
|---|---|
| graph construction 분해 | source, relation, topology 구분 |
| proxy mechanism 수용 | pFedGraph-QP, learned smooth 계열 연결 |
| Phase 5 준비 | counterfactual diagnostics로 연결 |
