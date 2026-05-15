# PHASE2_GRAPH_INFORMATIVENESS_REPORT

## 1) Phase 1 Summary
- Dataset: FashionMNIST
- Partition: Dirichlet label skew (`alpha=0.03`)
- Clients: 5
- Seeds: 42, 43, 44
- Rounds: 10
- Methods: FedAvg / FedAvgM
- FedAvg final accuracy mean: 0.7700
- FedAvgM final accuracy mean: 0.8111
- corr(final_acc, mean_CR) = -0.4657
- corr(final_acc, mean_CA) = -0.0621
- corr(final_acc, mean_DI) = -0.5984
- Interpretation (tentative): dominance strongest, conflict secondary, cancellation weak, sample size small.

## 2) Exact Commands Used
- `./.venv311/Scripts/python.exe run_general_experiment.py --method fedavgm --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant identity --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_fedavgm_seed42`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method fedavgm --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant identity --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_fedavgm_seed43`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method fedavgm --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant identity --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_fedavgm_seed44`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant update --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_update_seed42_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant update --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_update_seed43_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant update --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_update_seed44_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant random --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_random_seed42_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant random --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_random_seed43_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant random --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_random_seed44_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant shuffled --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_shuffled_seed42_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant shuffled --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_shuffled_seed43_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant shuffled --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_shuffled_seed44_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant uniform --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_uniform_seed42_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant uniform --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_uniform_seed43_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant uniform --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_uniform_seed44_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 42 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant identity --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_identity_seed42_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 43 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant identity --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_identity_seed43_l0p05`
- `./.venv311/Scripts/python.exe run_general_experiment.py --method graph_smooth --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.3 --num-clients 20 --rounds 10 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seed 44 --graph-preset signed_conflict_knn --graph-mode dense --graph-source update --graph-variant identity --graph-smoothing-lambda 0.05 --graph-laplacian-type normalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 0 --test-subset-size 0 --out-dir experiments_current\phase2_proper_n20_a0p3_signed_seed42 --run-tag phase2_graphsmooth_identity_seed44_l0p05`

## 3) Experiment Setting
- dataset: `fashionmnist`
- partition: `dirichlet`
- alpha: `0.3`
- num_clients: `20`
- seeds: `42, 43, 44`
- rounds: `10`
- base method: `FedAvgM` (plus identity/no-graph validity check)

## 4) Graph Source
- graph source used: `update`
## 5) Graph Variants
- `update`, `random`, `shuffled`, `uniform`, `identity`

## 6) Correction Operator
- `G_corrected = (I - lambda L) G`
- `Delta_corrected = sum_i p_i g_i_corrected`
- FedAvgM server momentum applied after corrected aggregation.

## 7) Lambda
- lambda: `0.05`
- laplacian: `normalized`

## 8) Final Accuracy Table
| variant | seed42 | seed43 | seed44 | mean | std |
| --- | --- | --- | --- | --- | --- |
| fedavgm_baseline | 0.7972 | 0.8043 | 0.8090 | 0.8035 | 0.0049 |
| update | 0.8032 | 0.8015 | 0.8111 | 0.8053 | 0.0042 |
| random | 0.8018 | 0.8051 | 0.8112 | 0.8060 | 0.0039 |
| shuffled | 0.8059 | 0.8032 | 0.8121 | 0.8071 | 0.0037 |
| uniform | 0.8038 | 0.8026 | 0.8130 | 0.8065 | 0.0046 |
| identity | 0.7972 | 0.8046 | 0.8141 | 0.8053 | 0.0069 |

## 9) Round-wise Curves
- Full round-wise diagnostics saved in `phase2_round_diagnostics.csv` and `phase2_round_diagnostics.json`.
- Includes: accuracy, loss, CR, CR_weighted, CA, DI, corrected update norm.

## 10) Graph Diagnostics
- Full graph diagnostics saved in `phase2_graph_diagnostics.csv`.
- Includes: density, edge-weight stats, degree stats, connected components, smoothness, `||A_t-A_{t-1}||_F`.

## 11) Validity Check (Identity vs FedAvgM)
- identity - fedavgm (mean): `0.0018`
- identity - fedavgm (std): `0.0023`
- Expected behavior: near-zero gap indicates implementation consistency.

## 12) Main Conclusion
- Graph-information hypothesis is **weak in this setting**: update graph does not consistently beat random/shuffled/uniform/identity across seeds.
- This conclusion is specific to the current low-sample setup (3 seeds, 10 rounds).

## 13) Recommendation
- Do not proceed to complex graph correction yet.
- Prefer dominance-only correction, diagnostic-only direction, or generic regularization baselines first.

_Generated at: 2026-05-11T18:05:55.169227_