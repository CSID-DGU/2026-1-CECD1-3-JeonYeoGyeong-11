# Graph-FL Metrics

## 표기

| Symbol | 의미 |
|---|---|
| $N$ | client 수 |
| $g_i^t$ | round $t$의 raw local update |
| $\tilde g_i^t$ | corrected 또는 graph-filtered update |
| $p_i^t$ | FedAvg sample-size weight |
| $q_i^t$ | contribution share |
| $A^t$ | client graph adjacency |
| $L^t$ | graph Laplacian |
| $z_i^t$ | graph construction용 client representation |
| $\epsilon$ | numerical stability constant |

기본 update:

$$
g_i^t = w_i^t - w^t
$$

FedAvg aggregation:

$$
g_{\mathrm{avg}}^t = \sum_i p_i^t g_i^t
$$

Pre/Post update:

$$
\Delta_{\mathrm{pre}} = \sum_i p_i^{pre} g_i,
\qquad
\Delta_{\mathrm{post}} = \sum_i p_i^{post} \tilde g_i
$$

## Core Diagnostic

| Metric | 정의 | 해석 |
|---|---|---|
| `DI` | $\max_i q_i$ | dominant client 집중도 |
| `N_eff` | $1 / \sum_i q_i^2$ | effective contributing clients |
| alignment | $\cos(g_i, \Delta)$ | aggregate direction 대표성 |
| `LOO` | client 제거 전후 aggregate direction 변화 | single-client sensitivity |
| real-control gap | $Score_{real} - Score_{control}$ | relation-specific signal |

Contribution share:

$$
q_i = \frac{p_i \lVert g_i \rVert}{\sum_j p_j \lVert g_j \rVert + \epsilon}
$$

Alignment:

$$
\cos(x,y)=\frac{x^\top y}{\lVert x\rVert_2 \lVert y\rVert_2+\epsilon}
$$

Leave-one-out:

$$
\mathrm{LOO}_i = 1 - \cos(\Delta, \Delta_{-i})
$$

## Graph Metric

| Metric | 의미 |
|---|---|
| density | active edge 비율 |
| degree | client별 연결량 |
| entropy | edge weight 분산성 |
| smoothness | graph 위 update 변화량 |
| spectral energy | Laplacian 기준 high-frequency 성분 |
| temporal stability | round 간 graph 변화 |

## 해석 Rule

| Pattern | 해석 |
|---|---|
| `DI` down, `N_eff` up | dominance 완화 |
| alignment up | aggregate direction 대표성 증가 |
| `LOO` down | 특정 client 영향 완화 |
| real-control gap up | relation-specific signal 증가 |
| graph density up | smoothing 양 증가 |

## Artifact Field

| Artifact | 주요 field |
|---|---|
| `round_metrics.csv` | pre/post aggregate, `DI`, `N_eff`, alignment, `LOO` |
| `client_metrics.csv` | client contribution, norm, alignment |
| `graph_stats.csv` | density, degree, entropy, spectral metric |
| `counterfactual_metrics.csv` | real graph와 control graph gap |
| `metric_validity_summary.csv` | synthetic expected-direction 결과 |
