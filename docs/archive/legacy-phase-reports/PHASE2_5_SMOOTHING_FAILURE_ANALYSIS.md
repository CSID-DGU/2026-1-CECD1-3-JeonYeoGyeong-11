# PHASE2_5_SMOOTHING_FAILURE_ANALYSIS

## Batch Status
- completed: A
- pending: B, C, D, E

## Effect Summary
| experiment | setting | method | mean_acc | std_acc | direction_change |
| --- | --- | --- | --- | --- | --- |
| A_laplacian | fashion_alpha0p03_n5 | A_fedavgm_baseline | 0.8135 | 0.0148 | nan |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p2_identity | 0.8108 | 0.0143 | 0.0000 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p2_shuffled | 0.8037 | 0.0198 | 0.0002 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p2_uniform | 0.8003 | 0.0298 | 0.0001 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p2_update | 0.8124 | 0.0141 | 0.0001 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p5_identity | 0.8108 | 0.0143 | 0.0000 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p5_shuffled | 0.8056 | 0.0212 | 0.0012 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p5_uniform | 0.8033 | 0.0202 | 0.0009 |
| A_laplacian | fashion_alpha0p03_n5 | A_lap_l0p5_update | 0.8058 | 0.0224 | 0.0010 |
| A_laplacian_lambda0p05 | fashion_alpha0p03_n5 | graph_identity | 0.8108 | 0.0143 | 0.0000 |
| A_laplacian_lambda0p05 | fashion_alpha0p03_n5 | graph_random | 0.8095 | 0.0163 | 0.0000 |
| A_laplacian_lambda0p05 | fashion_alpha0p03_n5 | graph_shuffled | 0.8103 | 0.0152 | 0.0000 |
| A_laplacian_lambda0p05 | fashion_alpha0p03_n5 | graph_uniform | 0.8087 | 0.0156 | 0.0000 |
| A_laplacian_lambda0p05 | fashion_alpha0p03_n5 | graph_update | 0.8008 | 0.0282 | 0.0000 |
| B_residual | fashion_alpha0p03_n5 | B_res_l0p5_shuffled | 0.7592 | 0.0262 | 0.0808 |
| B_residual | fashion_alpha0p03_n5 | B_res_l0p5_update | 0.7665 | 0.0506 | 0.0451 |
| D_dominance_only | fashion_alpha0p03_n5 | D_fedavgm | 0.8058 | 0.0180 | nan |
| D_dominance_only | fashion_alpha0p03_n5 | D_norm_clip_p75 | 0.8009 | 0.0175 | 0.0057 |
| D_dominance_only | fashion_alpha0p03_n5 | D_uniform | 0.8086 | 0.0105 | 0.0401 |

## Notes
- Report is incremental and resumable.
- Each batch merges into summary files without rerunning existing result files.

_Generated at: 2026-05-11T00:18:58.451954_