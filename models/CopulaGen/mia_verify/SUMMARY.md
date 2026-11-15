# MeLoMIA-ND verification and attacks on our BRCA releases (2026-11-15)
Attack code: ~/privacy/MIA_on_bulkRNAseq_CAMDA2026 (unchanged), settings = configs/experiments/paper_grid_brca.yaml
(melomia_nd: K=30, n_noise 600, optuna 60, xgb/rf/lgbm/cat/mlp, per-record calibration; tag k30_n600_prc, params identical
to the reported run).  Targets = our releases imported with models/CopulaGen/mia_import_targets.py.  Mean AUC (SD) over 5 splits.

| release | MeLoMIA-ND | TPR@1%FPR | MahalaMIA | MAMA-MIA v2 |
|---|---|---|---|---|
| TabPFN-3.5, no DP (ours) | 0.723 (0.033) | 0.057 | 0.579 | 0.616 |
| TabPFN-3.5, no DP (reported, repo runs) | 0.747 | 0.10 | 0.569 | 0.631 |
| MIAV + TabPFN-3.5, no DP | 0.974 | 0.811 | 0.989 | 0.931 |
| TabPFN-3.5 on LowRankDP eps30 context | 0.517 (0.013) | 0.016 | 0.507 | 0.563 |
| LowRankDP eps30 | 0.522 (0.011) | 0.016 | 0.506 | 0.566 |
| TabPFN-PE eps30 | 0.510 (0.019) | 0.014 | 0.507 | 0.553 |
| TabPFN-3.5 on LowRankDP eps10 context | 0.513 (0.008) | 0.016 | 0.507 | 0.554 |
| LowRankDP eps10 | 0.510 (0.011) | 0.012 | 0.503 | 0.561 |
| TabPFN-PE eps10 | 0.509 (0.018) | 0.013 | 0.505 | 0.533 |
| Private-PGM eps10 | 0.507 (0.023) | 0.015 | 0.504 | 0.512 |

Per split, ours vs reported (MeLoMIA-ND on TabPFN-3.5): 0.681/0.745/0.757/0.697/0.736 vs 0.794/0.742/0.739/0.724/0.738.
Fidelity/utility: ~/privacy/Pro-GENE-GEN-MPC/models/results_copula/TABPFN/BRCA_summary.md
