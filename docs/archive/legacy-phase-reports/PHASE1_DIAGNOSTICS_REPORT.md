# PHASE 1 Diagnostics Report

## Exact Commands Used

```bash
python run_general_suite.py --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 10 --local-epochs 1 --seeds 42 43 44 --variants fedavg fedavgm --out-dir ./experiments_current/phase1_diag_alpha003_n5_r10 --reuse-existing-results false
python scripts/analysis/phase1_diagnostics_report.py --input-dir ./experiments_current/phase1_diag_alpha003_n5_r10 --report-path ./PHASE1_DIAGNOSTICS_REPORT.md --round-csv-path ./experiments_current/phase1_diag_alpha003_n5_r10/phase1_round_diagnostics.csv --run-csv-path ./experiments_current/phase1_diag_alpha003_n5_r10/phase1_run_summary.csv --command-used "python run_general_suite.py --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 10 --local-epochs 1 --seeds 42 43 44 --variants fedavg fedavgm --out-dir ./experiments_current/phase1_diag_alpha003_n5_r10 --reuse-existing-results false"
```

## Dataset / Configuration

- dataset: fashionmnist
- partition: dirichlet
- alpha: 0.030
- num_clients: 5
- rounds: 10
- seeds: 42, 43, 44
- methods: fedavg, fedavgm
- optional extension (`alpha=0.1/0.3`, seeds 45/46) was not executed in this run

## FedAvg / FedAvgM Final Results

| method | mean final acc | std final acc | mean CR | mean CA | mean DI | mean N_eff |
|---|---:|---:|---:|---:|---:|---:|
| fedavg | 0.7700 | 0.0426 | 0.6733 | 0.5607 | 0.3058 | 4.3286 |
| fedavgm | 0.8111 | 0.0184 | 0.4400 | 0.5040 | 0.3120 | 4.2834 |

## Seed-by-Seed Comparison

| seed | FedAvg acc | FedAvgM acc | gap (M-A) | FedAvg mean CR | FedAvg mean CA | FedAvg mean DI | FedAvg min N_eff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8075 | 0.8219 | 0.0144 | 0.7000 | 0.5583 | 0.3021 | 3.7912 |
| 43 | 0.7104 | 0.7851 | 0.0747 | 0.6300 | 0.5240 | 0.3430 | 3.5697 |
| 44 | 0.7920 | 0.8262 | 0.0342 | 0.6900 | 0.5998 | 0.2721 | 4.6990 |

## Per-Round Diagnostic Summaries (mean across seeds)

| method | mean pair-neg frac (CR) | mean cancellation (CA) | mean dominance (DI) | mean effective clients |
|---|---:|---:|---:|---:|
| fedavg | 0.6733 | 0.5607 | 0.3058 | 4.3286 |
| fedavgm | 0.4400 | 0.5040 | 0.3120 | 4.2834 |

## Correlation Checks

- corr(final_acc, mean_CR): -0.4657
- corr(final_acc, mean_CA): -0.0621
- corr(final_acc, mean_DI): -0.5984
- corr(final_acc, min_N_eff): 0.3362
- corr(FedAvgM gain, FedAvg mean_CR): -0.9810
- corr(FedAvgM gain, FedAvg mean_CA): -0.6168
- corr(FedAvgM gain, FedAvg mean_DI): 0.7227

## Interpretation

High CR/CA/DI should coincide with lower final accuracy or unstable rounds if the interaction-pathology hypothesis is strong.
With only 3 seeds and one alpha setting, this evidence is limited and should be treated as directional (not definitive).
In this run set, the largest absolute cross-run correlation among CR/CA/DI is for **dominance** (abs corr = 0.5984).

- most important observed pathology: **dominance**
- Phase 2 decision: **at least one interaction pathology shows tentative signal; Phase 2 is justified, but only as a conservative follow-up.**

## Raw Diagnostic Exports

- round-level CSV: `experiments_current/phase1_diag_alpha003_n5_r10/phase1_round_diagnostics.csv`
- run-level CSV: `experiments_current/phase1_diag_alpha003_n5_r10/phase1_run_summary.csv`
- round-level JSON: `experiments_current/phase1_diag_alpha003_n5_r10/phase1_round_diagnostics.json`
- run-level JSON: `experiments_current/phase1_diag_alpha003_n5_r10/phase1_run_summary.json`