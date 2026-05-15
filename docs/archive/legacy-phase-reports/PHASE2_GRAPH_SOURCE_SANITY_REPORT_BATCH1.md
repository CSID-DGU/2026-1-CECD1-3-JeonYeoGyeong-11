# PHASE2_GRAPH_SOURCE_SANITY_REPORT

## Purpose
- Check whether graph source choice changes informativeness before complex operator tuning.

## Suite Settings
- dataset: `fashionmnist`
- partition: `dirichlet` (alpha=0.03)
- clients: `5`
- rounds: `2`
- seeds: `42`
- variants: `update, random, identity`

## Commands Run
- `.venv311/Scripts/python.exe scripts/analysis/phase2_graph_informativeness.py --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 2 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seeds 42 --graph-source update --variants update random identity --graph-smoothing-lambda 0.05 --graph-laplacian-type unnormalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 2000 --test-subset-size 1000 --out-dir experiments_current\phase2_graph_source_sanity_batch1\update --report-path experiments_current\phase2_graph_source_sanity_batch1\update\PHASE2_GRAPH_INFORMATIVENESS_REPORT.md --reuse-existing-results true`
- `.venv311/Scripts/python.exe scripts/analysis/phase2_graph_informativeness.py --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 2 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seeds 42 --graph-source ema_update --variants update random identity --graph-smoothing-lambda 0.05 --graph-laplacian-type unnormalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 2000 --test-subset-size 1000 --out-dir experiments_current\phase2_graph_source_sanity_batch1\ema_update --report-path experiments_current\phase2_graph_source_sanity_batch1\ema_update\PHASE2_GRAPH_INFORMATIVENESS_REPORT.md --reuse-existing-results true`
- `.venv311/Scripts/python.exe scripts/analysis/phase2_graph_informativeness.py --dataset fashionmnist --partition dirichlet --dirichlet-alpha 0.03 --num-clients 5 --rounds 2 --local-epochs 1 --batch-size 64 --model cnn --lr 0.01 --momentum 0.9 --weight-decay 0.0005 --seeds 42 --graph-source classifier_head_update --variants update random identity --graph-smoothing-lambda 0.05 --graph-laplacian-type unnormalized --graph-zero-diagonal true --compression-dim 256 --compression-seed 0 --graph-seed 0 --server-learning-rate 1.0 --server-momentum 0.9 --train-subset-size 2000 --test-subset-size 1000 --out-dir experiments_current\phase2_graph_source_sanity_batch1\classifier_head_update --report-path experiments_current\phase2_graph_source_sanity_batch1\classifier_head_update\PHASE2_GRAPH_INFORMATIVENESS_REPORT.md --reuse-existing-results true`

## Source-Level Summary
| graph_source | fedavgm | update | best_control | update-best_control | update-fedavgm | beats_controls |
| --- | --- | --- | --- | --- | --- | --- |
| update | 0.1110 | 0.1110 | 0.1110 | 0.0000 | 0.0000 | no |
| ema_update | 0.1110 | 0.1110 | 0.1110 | 0.0000 | 0.0000 | no |
| classifier_head_update | 0.1110 | 0.1110 | 0.1110 | 0.0000 | 0.0000 | no |

## Interpretation Guide
- `update-best_control > 0`: source-specific update graph beats all control variants on mean.
- `update-fedavgm > 0`: source-specific update graph exceeds FedAvgM baseline.

_Generated at: 2026-05-11T12:09:47.390682_