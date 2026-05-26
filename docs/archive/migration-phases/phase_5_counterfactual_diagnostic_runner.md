# Phase 5 Counterfactual Diagnostic Runner

## 목적

같은 client artifacts로 real graph와 control graph diagnostic path를 비교하는 runner를 설계했다.

## 범위

| 영역 | 내용 |
|---|---|
| actual path | real graph aggregation |
| shadow path | random, shuffled, uniform, identity, clustering control |
| metrics | `DI`, `N_eff`, alignment, `LOO`, graph stats |
| artifacts | per-round counterfactual rows |

## 결과

| Result | 의미 |
|---|---|
| graph-control comparison 표준화 | attribution evidence 생성 |
| diagnostic artifact 추가 | mechanism metric 추적 |
| Phase 6 준비 | aggregation/delivery/state hook으로 연결 |
