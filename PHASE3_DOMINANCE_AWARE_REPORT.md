# PHASE3_DOMINANCE_AWARE_REPORT

## 1) Phase 1 and Phase 2 Conclusions
- Phase 1: dominance showed the strongest negative association with final accuracy, conflict was weaker, cancellation was weak.
- Phase 2: update-induced graph did not beat graph controls.
- Phase 2 mean final accuracy: update 0.8008, random 0.8095, shuffled 0.8103, uniform 0.8087, identity 0.8108, FedAvgM 0.8123.

## 2) Why Graph-Centered Direction Was Deprioritized
- Graph-smoothing with update graph did not show consistent gains over random/shuffled/uniform/identity controls.
- This phase focuses on direct dominance control in server aggregation without graph smoothing.

## 3) Dominance Metrics
- `q_i = p_i ||g_i^t||`
- `qbar_i = q_i / (sum_j q_j + epsilon)`
- `DI_t = max_i qbar_i`
- `N_eff_t = 1 / (sum_i qbar_i^2 + epsilon)`

## 4) Methods and Equations
- FedAvgM baseline: `Delta_t = sum_i p_i g_i^t` (server momentum applied).
- Uniform weighting: `Delta_t = sum_i (1/N) g_i^t`.
- Norm clipping: `g_i_clipped = g_i * min(1, c / (||g_i|| + epsilon))`, `Delta_t = sum_i p_i g_i_clipped`.
- Contribution cap: if `p_i||g_i|| > cap`, scale `g_i` to satisfy `p_i||g_i|| = cap`, then aggregate with `p_i`.
- Soft reweighting: `alpha_i = p_i exp(-tau_d qbar_i) / sum_j p_j exp(-tau_d qbar_j)`, `Delta_t = sum_i alpha_i g_i`.
- Triggered soft reweighting: same alpha only when `DI_t > threshold`, otherwise FedAvg weights.

## 5) Commands Used
- `.venv311\Scripts\python.exe run_general_experiment.py --method dominance_aware --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --server-learning-rate 1.0 --server-momentum 0.9 --out-dir experiments_current\phase3_dominance --run-tag phase3_norm_clip_p75_seed42 --dominance-mode norm_clip --dominance-clip-percentile 0.75`
- `.venv311\Scripts\python.exe run_general_experiment.py --method dominance_aware --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --server-learning-rate 1.0 --server-momentum 0.9 --out-dir experiments_current\phase3_dominance --run-tag phase3_norm_clip_p75_seed43 --dominance-mode norm_clip --dominance-clip-percentile 0.75`
- `.venv311\Scripts\python.exe run_general_experiment.py --method dominance_aware --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --server-learning-rate 1.0 --server-momentum 0.9 --out-dir experiments_current\phase3_dominance --run-tag phase3_norm_clip_p75_seed44 --dominance-mode norm_clip --dominance-clip-percentile 0.75`

## 6) Experiment Settings
- dataset: `fashionmnist`
- partition: `dirichlet` (`alpha=0.03`)
- num_clients: `5`
- rounds: `10`
- seeds: `42, 43, 44`

## 7) Final Accuracy by Method and Seed
| method | seed42 | seed43 | seed44 | mean | std |
| --- | --- | --- | --- | --- | --- |
| norm_clip_p75 | 0.7955 | 0.7826 | 0.8245 | 0.8009 | 0.0175 |

## 8) Mean/Std Final Accuracy by Method
| method | mean_acc | std_acc | mean_DI_raw | mean_DI_corrected | mean_N_eff_raw | mean_N_eff_corrected |
| --- | --- | --- | --- | --- | --- | --- |
| norm_clip_p75 | 0.8009 | 0.0175 | 0.3116 | 0.3024 | 4.2939 | 4.3228 |

## 9) Round-wise Accuracy/Loss Curves
- Round-level curves are saved in `experiments_current/phase3_dominance/phase3_round_diagnostics.csv` and `.json`.

## 10) DI_t and N_eff_t Curves
- Raw and corrected `DI_t` / `N_eff_t` are logged per round in the same round-diagnostics files.

## 11) Whether Each Method Reduces Dominance
- Positive `mean_DI_reduction` indicates lower dominance than raw FedAvg weighting.
- Positive `mean_N_eff_gain` indicates broader effective participation.
| method | mean_DI_reduction | mean_N_eff_gain |
| --- | --- | --- |
| norm_clip_p75 | 0.0092 | 0.0289 |

## 12) Whether Reducing Dominance Improves Accuracy
- corr(final_accuracy, mean_DI_t_raw): `-0.9251`
- corr(final_accuracy, mean_N_eff_t_raw): `0.9170`
- corr(final_accuracy, mean_DI_reduction): `0.7315`

## 13) Comparison Against Simple Baselines
- Included simple baselines: uniform weighting, norm clipping, contribution cap.

## 14) Recommendation
- Baseline row missing; rerun Phase 3 before interpretation.

## Raw Outputs
- `experiments_current/phase3_dominance/phase3_round_diagnostics.csv`
- `experiments_current/phase3_dominance/phase3_run_summary.csv`
- `experiments_current/phase3_dominance/phase3_method_comparison.csv`
- `experiments_current/phase3_dominance/phase3_round_diagnostics.json`
- `experiments_current/phase3_dominance/phase3_run_summary.json`

_Generated at: 2026-05-10T20:40:01.200241_