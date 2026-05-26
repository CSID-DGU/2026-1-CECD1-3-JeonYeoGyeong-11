# Phase 2 Graph Informativeness: pFedSim-like

## 목적

pFedSim-like graph setting에서 update graph와 control graph의 차이를 확인했다.

## 설정

| Field | Value |
|---|---|
| dataset | Fashion-MNIST |
| graph family | pFedSim-like |
| variants | real/update, random, shuffled, uniform, identity |

## 결과

| 관찰 | 의미 |
|---|---|
| proxy graph 품질 민감도 확인 | paper-mechanism alignment 필요 |
| control 대비 차이 제한적 | matched controls 유지 필요 |
| diagnostic 중심 해석 필요 | current evidence pack으로 연결 |
