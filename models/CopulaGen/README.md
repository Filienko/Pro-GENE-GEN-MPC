# CopulaGen: high-fidelity synthetic bulk RNA-seq

Private-PGM (star structure, 4 bins) has good utility but poor fidelity: values sit on 4 bin means per gene (W1 ≈ 0.23), and genes are independent given the label (within-class co-expression r = 0).
CopulaGen keeps P-PGM's marginal-based design but makes it continuous and adds dependence:

- **Marginals**: for every class and gene, x = m + s · q(u). The shape q is a smoothed quantile function on 41 knots with Gaussian tails beyond the 2 % / 98 % knots. It is shrunk toward the Gaussian shape (`--gauss_mix`), which limits marginal memorization.
- **Copula**: training rows are mapped to normal scores, and the scores follow the FactorGen low-rank model z = Wᵀf + e. That is k shared factors, class covariances shrunk toward the pooled within-class one, and a diagonal residual.
- **Score sampler**: `--factor gauss` (CPU, seconds) or `--factor diffusion` (the FactorDiffusion DDPM on the k factor scores, GPU). The diffusion sampler gives the best precision and coverage and the best DE preservation.
- **DP-CopulaGen** (`--dp_epsilon`) works under zCDP with three Gaussian measurements:
  - noisy class counts;
  - per-class, per-gene histograms (the quantity P-PGM measures, sensitivity √d), fitted by weighted probit regression and shrunk toward the pooled fit;
  - a noisy k×k second moment of clipped normal-score projections onto a DP-PCA basis, or onto an external cohort's basis.

Like P-PGM and FactorGen, the non-DP CopulaGen learns its statistics from the training data without noise. Privacy here is empirical, measured by the attacks below. In DP-CopulaGen the bin edges default to training quantiles, which is P-PGM's convention; `--dp_edges external` takes them from a non-member cohort instead.

## Run

```bash
source env.sh   # data paths + CAMDA splits on this machine; edit for other machines
B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS"
# recommended (highest fidelity), ~10 s / fold BRCA, ~35 s / fold COMBINED on one GPU
python run_copulagen.py $B --k 30 --knots 41 --tail 0.02 --gauss_mix 0.7 --factor diffusion --device cuda --out_dir $OUT/BRCA/copula_diff_k30
# CPU-only, ~2 s / fold
python run_copulagen.py $B --k 30 --knots 41 --tail 0.02 --gauss_mix 0.5 --out_dir $OUT/BRCA/copula_gauss_k30
# DP version
python run_copulagen.py $B --dp_epsilon 10 --k 20 --bins 6 --dp_split 0.02 0.68 0.30 --out_dir $OUT/BRCA/dpcopula_eps10
```
For COMBINED, use `$COMB_TSV $COMB_ANN $COMB_SPLITS` and `--label_col cancer_type`.
The output layout is identical to `run_factorgen.py` (synthetic data/labels, `membership_labels_split_k.csv`, `split_k_ids.json`).

## Evaluate

- `eval_suite.py`: fidelity, utility and privacy in one table (see its docstring; `--oracle` adds real-train-as-synthetic as the ceiling).
- `evalpar.sh DS out.csv "1 2 3 4 5" dir...`: runs `eval_suite.py` in parallel, one process per run directory.
- `mamamia_eval.py`: the Huskies red-team MAMA-MIA scorer, using the same reference/target protocol.

## Results (5 CAMDA folds, held-out real as reference)

**The MAMA-MIA column in the two tables below is invalid** (shared encoder cache across parallel runs, one-way marginals only); use the "Red-team attacks" section instead. All privacy columns are AUCs; 0.5 means no leak. The marg attack is a 30-bin per-gene likelihood ratio, a stronger MAMA-MIA.

**TCGA-BRCA**

| run | W1 ↓ | precision | coverage | within-class corr r | DE TPR / FPR | DE ρ | TSTR acc | Mahala | DCR-ratio | DOMIAS | marg | MAMA-MIA (10 bins) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real train (oracle) | 0.074 | 0.84 | 0.99 | 0.77 | 1 / 0 | 1 | 0.859 | 1.00 | 1.00 | 0.97 | 0.75 | – |
| **copula_diff_k30** | **0.081** | **0.90** | **0.99** | 0.73 | 0.94 / 0.02 | 0.987 | 0.846 | 0.55 | 0.53 | 0.51 | 0.57 | 0.53 |
| copula_gauss_k30 | 0.082 | 0.85 | 0.97 | 0.74 | 0.92 / 0.03 | 0.975 | 0.828 | 0.55 | 0.52 | 0.51 | 0.58 | 0.53 |
| copula_gauss_k50 | 0.080 | 0.76 | 0.92 | 0.74 | 0.92 / 0.03 | 0.974 | 0.848 | 0.57 | 0.53 | 0.50 | 0.57 | 0.53 |
| FactorGauss k50 | 0.094 | 0.61 | 0.79 | 0.74 | 0.91 / 0.04 | 0.970 | 0.840 | 0.57 | 0.53 | 0.52 | 0.53 | 0.50 |
| FactorDiffusion k50 | 0.091 | 0.66 | 0.87 | 0.73 | 0.95 / 0.02 | 0.988 | 0.827 | 0.58 | 0.53 | 0.51 | 0.53 | 0.54 |
| P-PGM star, no noise | 0.229 | (1.0)* | 0.87 | 0.00 | 0.80 / 0.02 | 0.957 | 0.808 | 0.50 | 0.50 | 0.46 | 0.53 | 0.53 |
| P-PGM star ε=10 | 0.244 | (1.0)* | 0.84 | 0.00 | 0.64 / 0.14 | 0.630 | 0.784 | 0.50 | 0.50 | 0.47 | 0.53 | – |
| CAMDA MVN | 0.163 | 0.33 | 0.51 | 0.74 | 0.87 / 0.04 | 0.957 | 0.839 | 0.96 | 0.55 | 0.53 | 0.51 | – |

**TCGA-COMBINED**

| run | W1 ↓ | precision | coverage | within-class corr r | DE TPR / FPR | TSTR acc | Mahala | DCR-ratio | DOMIAS | marg | MAMA-MIA (10 bins) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real train (oracle) | 0.044 | 0.92 | 0.99 | 0.85 | 1 / 0 | 0.995 | 0.69 | 1.00 | 0.87 | 0.59 | – |
| **copula_diff_k30** | 0.053 | **0.91** | **0.97** | 0.78 | 0.97 / 0.02 | 0.991 | 0.51 | 0.51 | 0.52 | 0.51 | 0.50 |
| copula_gauss_k30 | **0.051** | 0.76 | 0.81 | 0.77 | 0.96 / 0.03 | 0.988 | 0.51 | 0.51 | 0.50 | 0.52 | 0.51 |
| FactorDiffusion k50 | 0.062 | 0.80 | 0.93 | 0.79 | 0.97 / 0.02 | 0.991 | 0.52 | 0.52 | 0.51 | 0.50 | 0.50 |
| FactorGauss k50 | 0.070 | 0.55 | 0.60 | 0.79 | 0.96 / 0.03 | 0.990 | 0.52 | 0.51 | 0.50 | 0.50 | 0.50 |
| P-PGM star, no noise | 0.271 | (1.0)* | 0.52 | 0.00 | 0.89 / 0.03 | 0.943 | 0.50 | 0.50 | 0.49 | 0.50 | 0.51 |
| P-PGM star ε=10 | 0.275 | (1.0)* | 0.48 | 0.00 | 0.77 / 0.09 | 0.930 | 0.50 | 0.50 | 0.49 | 0.50 | 0.50 |

\* P-PGM samples collapse onto 4 points per gene, so precision is ≈1 while recall is ≈0. Fold-to-fold SD of the attack AUCs is 0.01–0.04.

**DP comparison at matched ε** (P-PGM star vs DP-CopulaGen, k=20, 6 bins)

| dataset | ε | method | W1 | within-class corr r | DE TPR / FPR | TSTR acc |
|---|---|---|---|---|---|---|
| BRCA | 10 | P-PGM | 0.244 | 0.00 | 0.64 / 0.14 | 0.784 |
| BRCA | 10 | DP-CopulaGen | 0.233 | 0.24 | 0.54 / 0.22 | 0.756 |
| BRCA | 30 | P-PGM | 0.232 | 0.00 | 0.72 / 0.07 | 0.811 |
| BRCA | 30 | DP-CopulaGen | 0.130 | 0.45 | 0.68 / 0.14 | 0.805 |
| COMBINED | 10 | P-PGM | 0.275 | 0.00 | 0.77 / 0.09 | 0.930 |
| COMBINED | 10 | DP-CopulaGen | 0.182 | 0.46 | 0.75 / 0.15 | 0.899 |
| COMBINED | 30 | P-PGM | 0.272 | 0.00 | 0.85 / 0.05 | 0.944 |
| COMBINED | 30 | DP-CopulaGen | 0.109 | 0.59 | 0.84 / 0.07 | 0.973 |

All DP attack AUCs are 0.44–0.53 for both methods.

## CAMDA KL and feature overlap (official evaluator code, `camda_kl_overlap.py`)

KL uses `Statistics.compute_kl_divergences` (50 bins per gene, ε = 1e-10). "KL(real‖syn)" is CAMDA's `kl_mean_*`; "KL(syn‖real)" is the inverted direction. Overlap is `count_feature_overlap` on the top-10 logreg coefficients per class. REAL_TEST trains the classifier on the held-out real split, so it is the sampling-noise floor for overlap. Means over 5 folds; the reference is the held-out test split.

| run | BRCA KL(real‖syn) | BRCA KL(syn‖real) | BRCA overlap | COMB KL(real‖syn) | COMB KL(syn‖real) | COMB overlap |
|---|---|---|---|---|---|---|
| real train (floor) | 0.236 | 0.651 | (1.00) | 0.053 | 0.124 | (1.00) |
| real test (overlap floor) | – | – | 0.17 | – | – | 0.44 |
| **copula_diff_k30** | 0.290 | **0.680** | 0.22 | **0.068** | **0.140** | **0.44** |
| copula_gauss_k30 | 0.278 | 0.713 | 0.25 | 0.068 | 0.143 | 0.41 |
| FactorDiffusion k50 | 0.306 | 0.735 | 0.24 | 0.076 | 0.151 | 0.46 |
| FactorGauss k50 | 0.309 | 0.768 | 0.25 | 0.092 | 0.158 | 0.46 |
| CAMDA MVN | 0.220 | 1.237 | 0.37 | – | – | – |
| P-PGM star, no noise | 18.50 | 2.301 | 0.17 | 17.65 | 1.892 | 0.23 |
| P-PGM star ε=30 | 18.50 | 2.304 | 0.09 | 17.65 | 1.893 | 0.25 |
| P-PGM star ε=10 | 18.50 | 2.313 | 0.06 | 17.65 | 1.896 | 0.20 |
| DP-CopulaGen ε=30 | 0.317 | 0.806 | 0.08 | 0.094 | 0.275 | 0.31 |
| DP-CopulaGen ε=10 | 0.318 | 1.141 | 0.06 | 0.087 | 0.439 | 0.22 |

P-PGM's KL(real‖syn) ≈ 18 is driven by the ε = 1e-10 smoothing. Its samples occupy only 4 of the 49 bins per gene, and each empty bin costs about log(1e10) per unit of real mass. MVN gets the best KL(real‖syn) on BRCA because it is over-dispersed: it covers every bin. It is the worst non-P-PGM run on the inverted KL. BRCA overlap is noisy: two disjoint real splits share only 17 % of their top features.

## Verification (Nov 11 2026)

Checks run on the 5-fold results above; outputs are in `results_copula/VERIFY/`.

- **Data and splits**: the expression matrix and labels are identical to CAMDA's own `X_train_real_split_k.csv` / `y_train_real_split_k.csv`. Every run uses the CAMDA YAML ids, train and test never share a sample or a patient, and the 5 test folds partition all samples.
- **Baseline**: the original Huskies P-PGM submissions, scored with `eval_suite.py` on their own splits, match the `pgm_star` baseline. BRCA (ε=7): W1 0.232, DE 0.61 / 0.15, TSTR 0.791. COMBINED (ε=10, 12 project labels mapped to the 10 tissues): W1 0.272, DE 0.74 / 0.08, TSTR 0.946.
- **Metrics**: TSTR equals the official CAMDA evaluator's `accuracy_synthetic` for every run. The quantile-based W1 is about 3 % below scipy's exact W1 for all methods (0.083 vs 0.086). Every table number matches the per-fold CSVs.
- **Reproducibility**: rerunning seed 0 gives bit-identical output (Gaussian and diffusion samplers). Over 3 seeds the SD of the 5-fold mean is ≤ 0.0003 (W1), ≤ 0.005 (TSTR) and ≤ 0.005 (attack AUCs).
- **Tuning**: the configuration was chosen on BRCA folds 1–2 and applied to COMBINED unchanged. BRCA folds 3–5 are no worse than folds 1–2 (W1 0.079 vs 0.085).
- **DP accounting**: the logged noise scales equal the zCDP formulas, the budget shares sum to 1, and P-PGM's RDP calibration gives the same σ for the same ε. Bin edges and the class list still come from the training data, as in P-PGM.
- **MAMA-MIA has little power here**: with the real training split released as "synthetic", it reaches only 0.645 (10 bins) on BRCA. A score of 0.53 is therefore weak evidence.
- **Small classes leak more** (`class_attack.py`, the attacker knows the label, low-rank class Mahalanobis, AUC per class):

| run | BRCA Normal (n=32) | BRCA Her2 (65) | BRCA mean over classes | COMB Skin (68) | COMB mean over classes |
|---|---|---|---|---|---|
| real train released | 0.92 | 0.86 | 0.72 | 0.81 | 0.59 |
| copula_diff_k30 | 0.61 | 0.59 | 0.57 | 0.57 | 0.53 |
| FactorDiffusion k50 | 0.55 | 0.57 | 0.55 | 0.56 | 0.53 |
| P-PGM star, no noise | 0.50 | 0.54 | 0.51 | 0.51 | 0.51 |
| DP-CopulaGen ε=30 | 0.52 | 0.53 | 0.52 | 0.53 | 0.52 |
| DP-CopulaGen ε=10 | 0.51 | 0.52 | 0.51 | 0.51 | 0.51 |

- **Out-of-range values**: 0.04 % of CopulaGen values lie more than 0.5 outside the training range of their gene (about a quarter of rows contain at least one), and a few are negative.

## Red-team attacks (PPML-Huskies code, Nov 11 2026)

`redteam_attacks.py` runs the red team's attacks with their protocol: the membership test set is the whole real pool, and members are the fold's training samples. Outputs are in `results_copula/REDTEAM/`. AUC, mean of 5 folds unless noted.

| attack | dataset | copula_diff_k30 | copula, Gaussian marginals (folds 1–2) | FactorDiffusion k50 | DP-CopulaGen ε=30 | DP-CopulaGen ε=10 | P-PGM no noise | P-PGM ε=10 | CAMDA MVN | real train released |
|---|---|---|---|---|---|---|---|---|---|---|
| MAMA-MIA, 1-way + 2-way gene × label, 10 bins | BRCA | **0.748** | 0.591 | 0.569 | 0.578 | 0.529 | 0.528 | 0.527 | 0.522 | 0.994 |
| MAMA-MIA, same, 4 bins | BRCA | **0.699** | 0.585 | 0.568 | 0.577 | 0.535 | 0.569 | 0.541 | 0.525 | 0.915 |
| MahalaMIA, no aux | BRCA | 0.546 | 0.535 | 0.575 | 0.508 | 0.504 | 0.502 | 0.501 | 0.938 | 1.000 |
| MAMA-MIA, 1-way + 2-way, 10 bins | COMBINED | **0.664** | – | 0.549 | 0.537 | 0.517 | 0.512 | 0.509 | 0.511 | 0.981 |
| MAMA-MIA, same, 4 bins | COMBINED | **0.638** | – | 0.554 | 0.539 | 0.519 | 0.552 | 0.530 | 0.512 | 0.862 |
| MahalaMIA, aux reference | COMBINED | 0.543 | – | 0.561 | 0.512 | 0.505 | 0.505 | 0.503 | 0.901 | 0.899 |

Shadow-model attacks against copula_diff_k30 (15 base shadows on BRCA, 10 on COMBINED):

| attack | BRCA AUC / TPR@10 % | COMBINED AUC / TPR@10 % |
|---|---|---|
| MeLoMIA: red-team NoisyDiffusion synth-shadows and loss features, default classifiers, no Optuna | 0.641 / 0.205 | 0.534 / 0.120 |
| CopulaGen-aware synth-shadow: likelihood terms of a CopulaGen refitted on the synthetic data | 0.699 / 0.281 | 0.547 / 0.129 |

- CopulaGen is clearly more exposed than P-PGM to MAMA-MIA and to shadow-model attacks, most of all on BRCA. Only MahalaMIA is defeated by the low-rank design.
- The nonparametric marginal shape is the main MAMA-MIA leak: with `--marginal gauss` the BRCA score drops from 0.75 to 0.59, at W1 0.091 instead of 0.085 (folds 1–2).
- DP-CopulaGen at ε=10 is at P-PGM's level under every attack here.
- The MeLoMIA positive control did not work: a stand-in diffusion target (the red team's unconditional shadow model trained on real data) scored 0.49, so this simplified pipeline is not validated against the real NoisyDiffusion generator.

## LowRankDP (Nov 11 2026): the recommended DP generator

`lowrank_dp.py` / `run_lrdp.py` (`lrdp_full.sh` reproduces everything below; outputs in `results_copula/LRDP/`).
All three methods add noise of the same order to every class (a record sits in one class, so class-wise measurements compose in parallel for P-PGM and DP-CopulaGen too). LowRankDP differs in *what* it measures: clipped sums that give the class mean and spread directly, instead of histograms from which they must be reconstructed. On BRCA this halves the error of the class means relative to DP-CopulaGen (Basal at eps=10: 0.19 vs 0.40 expression units) with a smaller budget share. Relative to P-PGM the class means are only 10–30 % more accurate on BRCA and 30–45 % on COMBINED, and class spreads are similar (`results_copula/LRDP/class_moment_errors.csv`). The fidelity gain over P-PGM comes mainly from continuous values (P-PGM puts every gene on 4 values) and from the modelled co-expression. Its Gaussian measurements (zCDP):

- per-class sums of standardised expression in all genes, rows L2-clipped;
- per-class sums of absolute deviations for the per-gene spread (about a quarter of the noise of squared deviations);
- a k x k within-class covariance per class in a basis from a public cohort (`--external_counts`) or from DP-PCA;
- the standardisation anchor (coarse fixed-bin histogram, or the public cohort) and the clipping thresholds (noisy quantiles on a fixed grid).

Class shifts are shrunk gene by gene (empirical Bayes), and small classes borrow the pooled covariance. Nothing is taken from the training data without noise except the list of class names. The number of generated rows is the rounded sum of the noisy class counts (the tables below were produced before this change, with exactly as many rows as the training split). The gene range [0, 25] used for the private anchor and output clipping is a fixed assumption about VST values.

```bash
python run_lrdp.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS \
    --dp_epsilon 30 --out_dir $OUT/LRDP/BRCA/lrdp_priv_eps30          # add --external_counts REF.tsv for the public variant
```
About 1.5 s per fold on BRCA, CPU only.

5 folds. MAMA-MIA is the red-team protocol (1-way + 2-way, 10 bins; COMBINED on folds 1–3).

| dataset | eps | method | W1 | within-class corr r | DE TPR / FPR | TSTR acc | KL(real‖syn) | KL(syn‖real) | feature overlap | MAMA-MIA | MahalaMIA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRCA | 30 | **LowRankDP** | **0.102** | 0.49 | **0.75 / 0.05** | **0.831** | 0.384 | 0.682 | 0.14 | 0.566 | 0.51 |
| BRCA | 30 | LowRankDP + public | 0.103 | 0.63 | 0.73 / 0.05 | 0.821 | 0.384 | 0.692 | 0.23 | 0.560 | 0.51 |
| BRCA | 30 | DP-CopulaGen | 0.130 | 0.45 | 0.68 / 0.14 | 0.805 | 0.317 | 0.806 | 0.08 | 0.578 | 0.51 |
| BRCA | 30 | P-PGM | 0.232 | 0.00 | 0.72 / 0.07 | 0.811 | 18.50 | 2.304 | 0.09 | – | – |
| BRCA | 10 | LowRankDP | 0.153 | 0.12 | 0.58 / 0.07 | 0.790 | 0.536 | 0.848 | 0.11 | 0.532 | 0.51 |
| BRCA | 10 | LowRankDP + public | 0.148 | 0.59 | 0.54 / 0.07 | 0.793 | 0.431 | 0.838 | 0.15 | 0.526 | 0.50 |
| BRCA | 10 | P-PGM | 0.244 | 0.00 | 0.64 / 0.14 | 0.784 | 18.50 | 2.313 | 0.06 | 0.527 | 0.50 |
| COMBINED | 30 | **LowRankDP** | **0.068** | 0.68 | 0.91 / 0.04 | **0.986** | 0.105 | 0.135 | 0.33 | 0.541 | 0.52 |
| COMBINED | 30 | LowRankDP + public | 0.068 | 0.73 | 0.92 / 0.04 | 0.985 | 0.105 | 0.139 | 0.34 | 0.545 | 0.51 |
| COMBINED | 30 | DP-CopulaGen | 0.109 | 0.59 | 0.84 / 0.07 | 0.973 | 0.094 | 0.275 | 0.31 | 0.537 | 0.51 |
| COMBINED | 30 | P-PGM | 0.272 | 0.00 | 0.85 / 0.05 | 0.944 | 17.65 | 1.893 | 0.25 | – | – |
| COMBINED | 10 | LowRankDP | 0.085 | 0.52 | 0.81 / 0.06 | 0.968 | 0.121 | 0.152 | 0.27 | 0.526 | 0.51 |
| COMBINED | 10 | LowRankDP + public | 0.082 | 0.68 | 0.82 / 0.06 | 0.965 | 0.120 | 0.150 | 0.31 | 0.531 | 0.51 |
| COMBINED | 10 | P-PGM | 0.275 | 0.00 | 0.77 / 0.09 | 0.930 | 17.65 | 1.896 | 0.20 | 0.509 | 0.50 |

- MeLoMIA with synth-shadows (15 base shadows, no Optuna) against LowRankDP eps=30 on BRCA: AUC 0.495, TPR@10 % 0.100. On COMBINED (10 shadows, 5 folds): AUC 0.504 ± 0.016, TPR@10 % 0.108. The same pipeline gave 0.641 on the non-DP CopulaGen.
- Positive control: the same MeLoMIA pipeline against the CAMDA MVN baseline (`--gen mvn`, per-class full covariance, noise level 0.7) on BRCA gives AUC 0.939; MahalaMIA gives 0.944 and MAMA-MIA (10 bins) 0.517. So the pipeline has power, and the attacks that break MVN find nothing in LowRankDP.
- Model-aware shadow attack (`aware` stage; shadows are LowRankDP at the same eps, features are likelihood terms of a CopulaGen refitted on the synthetic data) against LowRankDP eps=30: AUC 0.517 ± 0.018 on BRCA (15 shadows) and 0.505 ± 0.011 on COMBINED (10 shadows). The meta-classifier reaches only 0.53 / 0.50 on the attacker's own held-out shadows. Small BRCA classes (Her2, Normal) reach 0.54 with fold SD 0.10–0.14.
- The same model-aware attack against LowRankDP eps=10 (shadows trained at eps=10): AUC 0.503 ± 0.025 on BRCA and 0.497 ± 0.007 on COMBINED; 0.50 on the attacker's own held-out shadows for both.
- Not run: MAMA-MIA on P-PGM eps=30, and a likelihood-ratio attack on the released class statistics.
- The shrinkage and budget split were tuned on BRCA folds 1–2.

### `--between`: more co-expression without public data (optional, off by default)

Private-only DP-PCA keeps few components (BRCA: 1 at eps=10, 2 at eps=30; COMBINED: 4 and 10), because the noise on the 978 x 978 Gram matrix hides the rest. `--between` uses the row space of the released class means (C-1 directions, post-processing, no budget) as the first basis vectors and removes them from the residuals before DP-PCA. Share of the real within-class variance captured by the basis (folds 1-3): BRCA 4.5 % -> 11.6 % (eps=10) and 16 % -> 21 % (eps=30); COMBINED 22 % -> 30 % and 36 % -> 41 %. Thirty exact components would capture about 57 %.

| Dataset | eps | Variant | within-class corr r | coex TPR / FER | TSTR acc | W1 | MAMA-MIA |
|---|---|---|---|---|---|---|---|
| BRCA | 30 | private | 0.489 | 0.25 / 0.11 | 0.831 | 0.102 | 0.566 |
| BRCA | 30 | private `--between` | 0.532 | 0.35 / 0.20 | 0.793 | 0.102 | 0.569 |
| BRCA | 10 | private | 0.118 | 0.08 / 0.08 | 0.790 | 0.153 | 0.532 |
| BRCA | 10 | private `--between` | 0.182 | 0.14 / 0.17 | 0.777 | 0.154 | 0.536 |
| COMBINED | 30 | private | 0.675 | 0.68 / 0.13 | 0.986 | 0.068 | 0.541 |
| COMBINED | 30 | private `--between` | 0.691 | 0.74 / 0.20 | 0.985 | 0.068 | 0.538 |
| COMBINED | 10 | private | 0.520 | 0.46 / 0.11 | 0.968 | 0.085 | 0.526 |
| COMBINED | 10 | private `--between` | 0.553 | 0.54 / 0.19 | 0.965 | 0.086 | 0.524 |

- Co-expression improves on both datasets; the false edge rate rises to the level of the public-cohort variant.
- BRCA TSTR accuracy drops (0.831 -> 0.793 at eps=30). Cause not yet investigated. COMBINED is unaffected.
- `--basis_share 0.35` gives a little more co-expression on BRCA at eps=30 (r 0.575) but worse W1 at eps=10 (0.170).
- Reproduce with `lrdp_between.sh` (resumable). Full tables: `results_copula/LRDP/between_summary.txt`.

### `--v2`: recommended preset (Nov 12 2026)

`run_lrdp.py --v2` = three changes, selected on folds 1-3 (`lrdp_screen.sh`, two rounds) and confirmed on 5 folds (`lrdp_v2.sh`):

1. Exact Gaussian-DP accounting (`--accounting gdp`). Every release is a Gaussian mechanism, so the composition is mu-GDP with mu^2 = 2 rho, converted to (eps, delta) exactly. Same stated (eps, delta), 17 % more rho at eps=30 and 29 % more at eps=10 (noise x0.93 and x0.88).
2. Clip threshold at the 0.75 quantile of the row norms instead of 0.9 (`--clip_q 0.75`).
3. Private-only: `--between` and shares mean 0.45 / basis 0.20 / ccov 0.10. With a public cohort only 1 and 2 apply.

| Data | eps | Variant | W1 | within-class corr r | coex TPR / FER | DE TPR / FPR | TSTR acc | KL real‖syn / syn‖real (test) | MAMA-MIA | MahalaMIA | model-aware |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRCA | 30 | v1 private | 0.102 | 0.489 | 0.25 / 0.11 | 0.75 / 0.05 | 0.831 | 0.384 / 0.682 | 0.566 | 0.510 | 0.517 |
| BRCA | 30 | **v2 private** | 0.098 | 0.596 | 0.38 / 0.17 | 0.77 / 0.04 | 0.832 | 0.400 / 0.634 | 0.569 | 0.511 | 0.514 |
| BRCA | 30 | v2 public | 0.099 | 0.619 | 0.41 / 0.16 | 0.75 / 0.04 | 0.817 | 0.409 / 0.631 | 0.561 | 0.507 | – |
| BRCA | 10 | v1 private | 0.153 | 0.118 | 0.08 / 0.08 | 0.58 / 0.07 | 0.790 | 0.536 / 0.848 | 0.532 | 0.505 | 0.503 |
| BRCA | 10 | **v2 private** | 0.130 | 0.397 | 0.21 / 0.18 | 0.63 / 0.06 | 0.778 | 0.495 / 0.714 | 0.552 | 0.510 | 0.506 |
| BRCA | 10 | v2 public | 0.129 | 0.596 | 0.32 / 0.16 | 0.58 / 0.06 | 0.808 | 0.437 / 0.726 | 0.541 | 0.505 | – |
| COMBINED | 30 | v1 private | 0.068 | 0.675 | 0.68 / 0.13 | 0.91 / 0.04 | 0.986 | 0.105 / 0.135 | 0.541 | 0.502 | 0.505 |
| COMBINED | 30 | **v2 private** | 0.070 | 0.702 | 0.74 / 0.18 | 0.91 / 0.03 | 0.983 | 0.113 / 0.124 | 0.541 | 0.504 | 0.511 |
| COMBINED | 30 | v2 public | 0.070 | 0.737 | 0.73 / 0.14 | 0.90 / 0.03 | 0.988 | 0.113 / 0.127 | 0.541 | 0.499 | – |
| COMBINED | 10 | v1 private | 0.085 | 0.520 | 0.46 / 0.11 | 0.81 / 0.06 | 0.968 | 0.121 / 0.152 | 0.526 | 0.500 | 0.497 |
| COMBINED | 10 | **v2 private** | 0.081 | 0.619 | 0.57 / 0.15 | 0.83 / 0.04 | 0.977 | 0.129 / 0.133 | 0.527 | 0.500 | 0.502 |
| COMBINED | 10 | v2 public | 0.079 | 0.704 | 0.72 / 0.18 | 0.84 / 0.05 | 0.975 | 0.126 / 0.134 | 0.530 | 0.498 | – |

- 5 folds; MAMA-MIA and MahalaMIA on COMBINED use folds 1-3. Model-aware = shadow attack with LowRankDP shadows of the same preset and eps.
- Gains: co-expression everywhere, W1 at eps=10, DE false positives, KL syn‖real.
- Costs: KL real‖syn rises slightly (tighter clipping narrows the tails), COMBINED W1 at eps=30 0.068 -> 0.070, BRCA TSTR at eps=10 0.790 -> 0.778, BRCA top-feature overlap at eps=30 7.2 -> 5.6 of 50, MAMA-MIA on BRCA eps=10 0.532 -> 0.552 (within fold SD 0.02-0.03).
- Rejected in the screen: winsorising at |z| <= 3 (worse W1 on COMBINED), clip quantile 0.97 and 0.60, and reshaping marginals toward the released pooled histogram (`--marg`, worse W1 on both datasets).
- BRCA TSTR under `--between`: v1 under-represents the within-class variance along the class-separating axes (5-25 % of the real value; 44-83 % with `--between`), so its synthetic subtypes are easier to separate than real ones. Real-to-real accuracy is 0.859.
- The settings were selected on folds 1-3 of the evaluation data; this is not charged to eps.

### DP audit (Nov 11 2026)

- Every statement in `LowRankDP.fit` that reads the private arrays was listed. Each one is either a Gaussian-mechanism release (class counts, anchor histogram, four clipping-quantile histograms, class sums, absolute-deviation sums, Gram matrix, per-class factor covariance) or a deterministic transform of the data with earlier releases that feeds the next release. Nothing else derived from the private data is stored or returned.
- Budget recomputed from the noise actually drawn (sum of sensitivity^2 / 2 sigma^2 over all releases): 10 releases private-only, 7 with a public cohort, 11 with the vote step. The sum equals the zCDP budget exactly, i.e. eps = 10.000 and 30.000 at delta = 1e-5, with and without `--between`.
- All result runs have `noise_off = []` (the ablation switch that removes noise was not used).
- Outside the guarantee, to state in a paper:
  - Result runs use a fixed public seed for reproducibility. A real release must use `--seed -1` (OS entropy): with a known seed the noise can be regenerated.
  - The class label set is read from the data and is assumed public.
  - The range [0, 25], the quantile grid and the fixed constants were chosen by someone who had seen the data; budget shares and shrinkage were tuned on BRCA folds 1-2. None of this is charged to eps.
  - The input is a VST matrix computed jointly over all samples, so one patient slightly affects the other rows. The guarantee is with respect to the VST matrix as given. This applies to every generator on this benchmark.
  - The run directory also contains `split_*_ids.json` and `membership_labels_split_*.csv` for evaluation. They are not DP and must not be published with the synthetic data.
  - With a public cohort, patients of the private pool are removed from it by ID. This uses pool membership (not training-fold membership).
  - Standard implementation caveats: floating-point Gaussian sampling and a non-cryptographic RNG (NumPy PCG64).
- DP-CopulaGen is not DP end to end: its bin edges are training quantiles taken without noise (`dp_copula.py`, `fit`).
- A Private-Evolution-style vote step (`--vote`) changed nothing measurable and is off by default.
- The public cohort (the red team's COMBINED reference file, minus every patient of the private pool) mainly adds co-expression.

## Caveats

- Nonparametric marginals leak slightly more under the 30-bin marginal attack on BRCA (0.57–0.58 vs 0.53 for P-PGM). `--gauss_mix` trades this off: at 1.0 the marginals are Gaussian, giving 0.55 on BRCA folds 1–2 at a W1 cost of +0.01. Under the red team's two-way MAMA-MIA the gap is much larger (see "Red-team attacks").
- At ε=10, DP-CopulaGen marginals are about as noisy as P-PGM's (σ ≈ 21.5 counts per cell for 978 genes). It wins on W1 and co-expression but loses some DE-TPR and TSTR. `--eb` (empirical-Bayes shrinkage of the class shifts) lowers W1 and DE-FPR further, at a cost in DE-TPR. At ε=1 neither method recovers per-class marginals for 978 genes.
- The Huskies MAMA-MIA encoder caches its bin thresholds in `data/experiment_artifacts/competition_thresholds_*` under the working directory. Parallel runs overwrite each other's file, which produced the earlier `np.digitize` errors. `mamamia_eval.py` and `redteam_attacks.py` now pass a per-process cache name.
