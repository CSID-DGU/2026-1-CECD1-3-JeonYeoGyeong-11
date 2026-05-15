# EXPERIMENT_REPORT

## Short goal
Evaluate whether update-derived/signed client graphs explain or correct single-global label-skew FL aggregation pathology better than graph-free dominance correction or generic smoothing controls.

## Experiment settings
- Core setting: FashionMNIST, Dirichlet label-skew, 20 clients, 20 rounds, local_epochs=1, seeds=42..46.
- Stages: main separation, alpha sweep, client-count sweep, operator sanity, dataset expansion (conditional).
- This report is generated from runs with status `RUN`; missing specs are marked `NOT_RUN` in CSV tables.

## Results tables
| method | runs | final_acc_mean | last5_acc_mean | last5_acc_std | mean_CR | mean_CA | mean_DI | mean_N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contribution_cap | 1 | 0.8179 | 0.8130 | 0.0000 | 0.4063 | 0.5955 | 0.0939 | 13.1375 |
| ema_update_graph_correction | 1 | 0.8417 | 0.8337 | 0.0000 | 0.4184 | 0.6130 | 0.1496 | 11.3349 |
| fedavg | 1 | 0.7891 | 0.7855 | 0.0000 | 0.4484 | 0.6328 | 0.1587 | 11.0134 |
| fedavgm | 1 | 0.8467 | 0.8354 | 0.0000 | 0.4216 | 0.6181 | 0.1529 | 11.2774 |
| identity_or_no_graph_control | 1 | 0.8423 | 0.8335 | 0.0000 | 0.4232 | 0.6188 | 0.1532 | 11.2878 |
| n_eff_aware_weighting | 1 | 0.8474 | 0.8316 | 0.0000 | 0.4226 | 0.6072 | 0.1020 | 15.2803 |
| norm_clipping | 1 | 0.8359 | 0.8321 | 0.0000 | 0.4079 | 0.6038 | 0.1375 | 11.6992 |
| random_graph_control | 1 | 0.8466 | 0.8350 | 0.0000 | 0.4245 | 0.6146 | 0.1523 | 11.2912 |
| shuffled_graph_control | 1 | 0.8468 | 0.8329 | 0.0000 | 0.4192 | 0.6140 | 0.1503 | 11.3466 |
| signed_conflict_graph_correction | 1 | 0.8403 | 0.8366 | 0.0000 | 0.4247 | 0.6156 | 0.1535 | 11.2789 |
| soft_dominance_reweighting | 1 | 0.8535 | 0.8341 | 0.0000 | 0.4168 | 0.6131 | 0.1462 | 11.6393 |
| uniform_graph_control | 1 | 0.8400 | 0.8337 | 0.0000 | 0.4197 | 0.6136 | 0.1514 | 11.3228 |
| update_graph_correction | 1 | 0.8415 | 0.8385 | 0.0000 | 0.4142 | 0.6116 | 0.1515 | 11.3242 |

## Method ranking by final accuracy
1. `soft_dominance_reweighting`: mean=0.8535, std=0.0000
2. `n_eff_aware_weighting`: mean=0.8474, std=0.0000
3. `shuffled_graph_control`: mean=0.8468, std=0.0000
4. `fedavgm`: mean=0.8467, std=0.0000
5. `random_graph_control`: mean=0.8466, std=0.0000
6. `identity_or_no_graph_control`: mean=0.8423, std=0.0000
7. `ema_update_graph_correction`: mean=0.8417, std=0.0000
8. `update_graph_correction`: mean=0.8415, std=0.0000
9. `signed_conflict_graph_correction`: mean=0.8403, std=0.0000
10. `uniform_graph_control`: mean=0.8400, std=0.0000
11. `norm_clipping`: mean=0.8359, std=0.0000
12. `contribution_cap`: mean=0.8179, std=0.0000
13. `fedavg`: mean=0.7891, std=0.0000

## Method ranking by last5 accuracy
1. `update_graph_correction`: mean=0.8385, std=0.0000
2. `signed_conflict_graph_correction`: mean=0.8366, std=0.0000
3. `fedavgm`: mean=0.8354, std=0.0000
4. `random_graph_control`: mean=0.8350, std=0.0000
5. `soft_dominance_reweighting`: mean=0.8341, std=0.0000
6. `ema_update_graph_correction`: mean=0.8337, std=0.0000
7. `uniform_graph_control`: mean=0.8337, std=0.0000
8. `identity_or_no_graph_control`: mean=0.8335, std=0.0000
9. `shuffled_graph_control`: mean=0.8329, std=0.0000
10. `norm_clipping`: mean=0.8321, std=0.0000
11. `n_eff_aware_weighting`: mean=0.8316, std=0.0000
12. `contribution_cap`: mean=0.8130, std=0.0000
13. `fedavg`: mean=0.7855, std=0.0000

## Method ranking by stability across seeds
1. `fedavg`: last5 std=0.0000
2. `fedavgm`: last5 std=0.0000
3. `update_graph_correction`: last5 std=0.0000
4. `signed_conflict_graph_correction`: last5 std=0.0000
5. `ema_update_graph_correction`: last5 std=0.0000
6. `random_graph_control`: last5 std=0.0000
7. `shuffled_graph_control`: last5 std=0.0000
8. `uniform_graph_control`: last5 std=0.0000
9. `identity_or_no_graph_control`: last5 std=0.0000
10. `norm_clipping`: last5 std=0.0000
11. `contribution_cap`: last5 std=0.0000
12. `soft_dominance_reweighting`: last5 std=0.0000
13. `n_eff_aware_weighting`: last5 std=0.0000

## Pathology metric analysis
- CR/CA/DI/N_eff are reported per-round in `round_logs.csv` and summarized per-run in `raw_results.csv`.
- Interpret DI decrease with N_eff increase as reduced contribution concentration.

## Intervention strength analysis
- Uses `cos_delta_corrected_vs_base` and `rel_delta_change` to verify non-trivial intervention.
- Near-zero `rel_delta_change` indicates effectively no correction despite operator invocation.

## Graph informativeness analysis
- Graph-specific claim requires update/signed > random/shuffled/uniform/identity under same seeds/settings.
- If controls match graph variants, treat improvement as generic smoothing (Case C).

## Decision among Case A/B/C/D
- **Selected case: A**
- Graph methods beat controls with non-trivial delta perturbation and stay competitive.

## Recommended next step
- Expand with stronger graph-source ablations and robustness checks before novelty claims.

## Requested direct answers
1. Does update-derived graph beat graph controls? See Stage 1 table (or NOT_RUN if missing).
2. Does signed conflict graph beat positive/update graph? See `signed_conflict_graph_correction` vs `update_graph_correction`.
3. Does graph-based correction beat graph-free dominance correction? Compare graph rows vs dominance rows.
4. Does generic smoothing explain the gains? Check random/uniform/shuffled proximity to update/signed.
5. Are CR/CA/DI/N_eff useful for explaining failure? Check per-round logs for difficult seeds/rounds.
6. Which case is currently supported? **A** (provisional if coverage is incomplete).
