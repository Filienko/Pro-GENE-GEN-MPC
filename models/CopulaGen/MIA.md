# Membership-inference attacks: what was run, from where, and the results

All numbers are membership-inference **ROC AUC** (0.5 = chance; our bar: < 0.6), mean over the 5 CAMDA splits unless
noted, with the worst fold in brackets where given.  Members = a split's training patients (871), non-members = its
held-out patients (218).  Result files live in `models/results_copula/` (git-ignored) and
`~/privacy/mia_verify/results/runs/`; the summary tables are `BRCA_compare.md`, `BIO_compare.md`, `ADULT_compare.md`
and `RANKS_per_metric.md` in this folder.

## 1. The attacks

### From the CAMDA attack repository (`~/privacy/MIA_on_bulkRNAseq_CAMDA2026`, used unchanged)

Run through `~/privacy/mia_verify/` (data shim `env.sh`, config `configs/verify_brca.yaml`; repo copy in
`mia_verify/`).  Our releases are imported as targets `ours@run=<run>` by `mia_import_targets.py`.  BRCA only (the
repo is wired to the CAMDA datasets).

| Attack | What it exploits | Settings |
|---|---|---|
| **MeLoMIA-ND** | Synthetic-shadow attack: 30 NoisyDiffusion shadow models trained on synthetic data; per-sample denoising-loss features, meta-classifier ensemble, per-record calibration. The repo's strongest attack (0.75 on TabPFN-3.5 BRCA in their paper). | K = 30 shadows, n_noise 600, 60 Optuna trials, classifiers xgb / rf / lgbm / cat / mlp, per-record calibration (tag `k30_n600_prc`) |
| **MAMA-MIA v2** | Marginal-based: how much more likely a candidate is under the release's one- and two-way marginals than under a reference. | public cliques, automatic edges, class centring |
| **MahalaMIA** | Mahalanobis distance of a candidate to the release (ridge covariance). Tailored to Gaussian generators: breaks the CAMDA MVN completely. | ridge 1e-4, class shrink 0.5 |

### Our own (this folder)

| Attack | File | What it exploits | Used on |
|---|---|---|---|
| **kNN-LR** | `knn_lr_attack.py` | Generator-aware for TabPFN's column-by-column generation: a member's later genes are predicted unusually well by synthetic rows whose earlier genes match it. k-NN likelihood ratio against the real candidate pool (k = 40, 150 conditioning genes). | BRCA, Bioresponse, Adult |
| **Fused** | `fused_attack.py` | Rank average of MeLoMIA-ND, kNN-LR and MAMA-MIA v2 (no labels used). | BRCA |
| TabPFN-LR | `tabpfn_lr_attack.py` | Same idea as kNN-LR with TabPFN itself as the density model. Exploratory (fold 3); superseded by kNN-LR, not in the tables. | BRCA |
| **eval_suite attacks** | `eval_suite.py` | Mahalanobis (`mahala_auc`), distance to closest record (`dcr_auc`), DCR ratio against a reference half (`dcrr_auc`), DOMIAS-style density ratio in PCA-20 (`domias_auc`), marginal density ratio (`marg_auc`). | BRCA, Bioresponse |
| **Adult attacks** | `adult_eval.py` | DCR, DCR ratio, Mahalanobis, DOMIAS on a one-hot / range-scaled encoding. | Adult |
| **MAMA-lite** | `mama_lite.py` | Our re-implementation of MAMA-MIA (4 / 10 quantile bins). | BRCA, Bioresponse, Adult |
| **PPML-Huskies red team** | `redteam_attacks.py` (wraps the red team's code in `~/privacy/PPML-Huskies-Red-Team/src`) | `simple`: MahalaMIA + MAMA-MIA 1-/2-way; `melomia`: MeLoMIA with ND synth-shadows (no Optuna); `aware`: **generator-aware synth-shadow** -- shadows are the attacked generator itself (CopulaGen or DP-MVN), features = per-sample likelihood terms under it. | BRCA, COMBINED (Nov 11-12) |

"Strongest attack" in the tables = the highest AUC of the available attacks on each fold, then averaged.  BRCA uses
MeLoMIA-ND, MAMA-MIA v2, MahalaMIA, kNN-LR and fused; Bioresponse and Adult only have our attacks, which are weaker.

## 2. Results

### BRCA, full suite (5 folds)

| Generator | MeLoMIA-ND | MAMA-MIA v2 | MahalaMIA | kNN-LR | Fused | **Strongest (worst)** | Fused TPR@1%FPR |
|---|---|---|---|---|---|---|---|
| MVN (CAMDA, noise 0.7) | 0.924 | 0.514 | **1.000** | 0.543 | 0.722 | **1.000 (1.000)** | 0.293 |
| TabPFN-3.5 | 0.723 | 0.616 | 0.579 | 0.693 | 0.768 | 0.773 (0.818) | 0.140 |
| TabPFN hybrid, cov 50% | 0.570 | 0.541 | 0.533 | 0.530 | 0.584 | 0.589 (0.611) | 0.024 |
| TabPFN hybrid, dpcov 50% | 0.541 | 0.570 | 0.519 | 0.539 | 0.575 | 0.577 (0.606) | 0.044 |
| TabPFN hybrid, dpcov 65% | 0.545 | 0.564 | 0.525 | 0.542 | 0.577 | 0.578 (0.600) | 0.027 |
| TabPFN-3.5 prompted with DP-MVN ε30 | 0.517 | 0.563 | 0.507 | 0.521 | 0.547 | 0.565 (0.587) | 0.025 |
| DP-MVN ε=10000 | 0.644 | 0.602 | 0.553 | 0.582 | 0.673 | 0.675 (0.695) | 0.049 |
| DP-MVN ε=1000 | 0.627 | 0.603 | 0.547 | 0.580 | 0.666 | 0.669 (0.683) | 0.048 |
| DP-MVN ε=300 | 0.589 | 0.589 | 0.529 | 0.560 | 0.623 | 0.623 (0.645) | 0.048 |
| DP-MVN ε=200 | 0.568 | 0.583 | 0.529 | 0.563 | 0.608 | 0.610 (0.621) | 0.046 |
| DP-MVN ε=100 | 0.540 | 0.594 | 0.519 | 0.548 | 0.590 | 0.598 (0.611) | 0.042 |
| DP-MVN ε=70 | 0.534 | 0.578 | 0.517 | 0.543 | 0.576 | 0.584 (0.610) | 0.029 |
| DP-MVN ε=50 | 0.516 | 0.585 | 0.518 | 0.536 | 0.571 | 0.586 (0.598) | 0.046 |
| DP-MVN ε=30 | 0.522 | 0.566 | 0.506 | 0.526 | 0.555 | 0.569 (0.590) | 0.022 |
| DP-MVN ε=10 | 0.510 | 0.561 | 0.503 | 0.515 | 0.539 | 0.561 (0.611) | 0.024 |
| DP-MVN ε=1 | 0.507 | 0.503 | 0.495 | 0.502 | 0.504 | 0.516 (0.528) | 0.018 |
| DP-MVN ε=0.5 | 0.504 | 0.498 | 0.501 | 0.494 | 0.499 | 0.512 (0.529) | 0.016 |
| Private-PGM ε=10 | 0.506 | 0.512 | 0.504 | 0.511 | 0.512 | 0.520 (0.543) | 0.019 |

DP-MVN with copula marginals (`dp_mvn_copula.py`): BRCA re-run with the wrapper in progress (`copula_eval.sh`); its
rows will appear in `BRCA_compare.md`.

Generator-aware synth-shadow (`redteam_attacks.py aware`, shadows = DP-MVN itself; older `--between` variant of the
v2 settings, ensemble of xgb / rf / cat):

| | BRCA | COMBINED |
|---|---|---|
| DP-MVN ε=10 | 0.506 | 0.502 |
| DP-MVN ε=30 | 0.514 | 0.511 |
| non-DP CopulaGen (positive check) | 0.699 | 0.547 |

### BRCA, two-stage ablations (fold 3 only)

Stage 1 makes synthetic rows; stage 2 is trained / prompted on them plus the given share of real-derived cluster rows
(5-member averages with DP-MVN-shaped noise).  ND = the CAMDA NoisyDiffusion generator.

| Pipeline | Real-derived | MeLoMIA-ND | MAMA-MIA v2 | MahalaMIA | Strongest |
|---|---|---|---|---|---|
| ND on real patients | -- | 0.975 | 0.572 | 0.781 | 0.975 |
| ND -> ND | 0 % | 0.645 | 0.548 | 0.606 | 0.645 |
| ND + cluster rows -> ND | 65 % | 0.617 | 0.509 | 0.620 | 0.620 |
| ND + cluster rows -> TabPFN | 65 % | 0.688 | 0.517 | 0.559 | 0.688 |
| TabPFN -> TabPFN | 0 % | 0.728 | 0.616 | 0.572 | >= 0.728 (kNN-LR / fused not run) |
| DP-MVN ε30 -> ND | 0 % | 0.507 | 0.503 | 0.509 | 0.520 |
| DP-MVN ε30 + cluster rows -> ND | 65 % | 0.563 | 0.499 | 0.544 | 0.584 |
| cluster rows, no stage 2 | 100 % | -- | -- | -- | 0.669 (kNN-LR 0.665, DCR 0.660) |
| cluster rows -> TabPFN | 100 % | 0.556 | 0.575 | 0.520 | 0.601 |

Ratio series (TabPFN prompted with ~870 rows, share of cluster rows vs DP-MVN ε30 rows), strongest attack: 0 % 0.546,
25 % 0.548, 50 % 0.573, 75 % 0.576, 100 % 0.601.

### Bioresponse (5 folds; our attacks only)

| Generator | kNN-LR | MAMA-lite | MahalaMIA | DCR | **Strongest (worst)** |
|---|---|---|---|---|---|
| MVN | 0.504 | 0.503 | 0.702 | 0.508 | 0.702 (0.732) |
| TabPFN-3.5 (fixed generator) | 0.524 | 0.522 | 0.518 | 0.535 | 0.564 (0.572) |
| TabPFN hybrid dpcov 65% | 0.519 | 0.508 | 0.532 | 0.565 | 0.595 (0.615) |
| cluster rows, no TabPFN | 0.527 | 0.510 | 0.565 | 0.613 | 0.640 (0.668) |
| DP-MVN ε=1000 | 0.508 | 0.507 | 0.512 | 0.515 | 0.554 (0.573) |
| DP-MVN ε=100 | 0.506 | 0.508 | 0.511 | 0.509 | 0.550 (0.575) |
| DP-MVN ε=30 | 0.504 | 0.505 | 0.507 | 0.506 | 0.547 (0.569) |
| DP-MVN ε=10 | 0.503 | 0.506 | 0.510 | 0.509 | 0.551 (0.571) |
| Private-PGM ε=10 | 0.503 | 0.513 | 0.505 | 0.497 | 0.535 (0.553) |

### Adult (5 folds; our attacks only)

| Generator | DCR | kNN-LR | MAMA-lite | **Strongest (worst)** |
|---|---|---|---|---|
| TabPFN-3.5 | 0.517 | 0.520 | 0.518 | 0.538 (0.559) |
| TabPFN hybrid dpcov 65% | 0.519 | 0.506 | 0.499 | 0.527 (0.536) |
| MVN | 0.501 | 0.508 | 0.502 | 0.532 (0.550) |
| DP-MVN ε=1000 | 0.510 | 0.511 | 0.503 | 0.536 (0.562) |
| DP-MVN ε=30 | 0.500 | 0.511 | 0.501 | 0.527 (0.552) |
| DP-MVN ε=10 | 0.502 | 0.510 | 0.504 | 0.534 (0.556) |
| DP-MVN-copula ε=1000 | 0.501 | 0.510 | 0.513 | 0.533 (0.551) |
| DP-MVN-copula ε=30 | 0.501 | 0.511 | 0.514 | 0.535 (0.553) |
| DP-MVN-copula ε=10 | 0.507 | 0.515 | 0.514 | 0.541 (0.560) |
| Private-PGM ε=10 | 0.498 | 0.506 | 0.500 | 0.525 (0.545) |

### Earlier red-team runs (`redteam_attacks.py`, Nov 11-12, `results_copula/REDTEAM/`)

| Target | MAMA-MIA 1+2-way (10 bins) | MahalaMIA | MeLoMIA (ND shadows) | aware synth-shadow |
|---|---|---|---|---|
| Real training data (upper bound) | 0.994 | 1.000 | -- | -- |
| CAMDA MVN | 0.522 | 0.938 | 0.939 | -- |
| non-DP CopulaGen | 0.748 | 0.546 | 0.641 | 0.699 |
| DP-CopulaGen ε=30 | 0.578 | 0.508 | -- | -- |
| Private-PGM ε=10 / ε=∞ | 0.527 / 0.528 | 0.501 / 0.502 | -- | -- |
| DP-MVN v2 ε=30 | 0.558 | 0.511 | -- | 0.514 |

## 3. What the attacks show

* **MVN leaks completely** (MahalaMIA 1.000): with fewer patients than genes per subtype its covariance has rank
  n_c - 1, so the release lies in the members' span.  DP-MVN's low-rank factors + independent per-gene residuals make
  the release full rank; MahalaMIA stays at 0.55 even at ε=10000.
* **TabPFN-3.5 is the most attackable usable generator** (fused 0.768, MeLoMIA-ND 0.723 reproducing the repo's 0.75).
  The fused attack is the strongest on it; kNN-LR alone reaches 0.693.
* **On DP-MVN the strongest attack is MAMA-MIA v2 at ε ≤ 100 and MeLoMIA-ND / fused above that.**  DP-MVN crosses the
  0.6 bar on the mean above ε=100 and on the worst fold from ε=70 (ε=10 also has one fold at 0.611 -- fold noise of
  about ±0.02); it levels off at ~0.67 for ε ≥ 1000 -- structural, not noise.
* **A leaky first stage stays leaky**: no second stage (TabPFN or ND) brings a non-private first stage under 0.6.
* **Bioresponse and Adult**: our attacks barely separate the methods (≈ 0.53-0.56 for everything except MVN and the
  real-derived cluster rows); privacy there is weakly measured.  MeLoMIA-ND and MAMA-MIA v2 would need porting.
* No new membership-inference attack was developed for DP-MVN; the generator-aware synth-shadow above is the tailored
  one (0.51 at ε 10-30).  For ε ≤ ~5 the formal bound TPR ≤ e^ε · FPR + δ is the stronger statement.
