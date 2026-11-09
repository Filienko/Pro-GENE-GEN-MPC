# FactorGen: low-rank factor generators for private, high-fidelity synthetic RNA-seq

FactorGen targets the gap in the CAMDA 2025 benchmark (Öztürk et al., bioRxiv 2026.03.02.707794):
- **Fidelity without privacy:** MVN, CVAE/CVAE-GMM and Embedded Diffusion have high fidelity, but membership inference breaks them.
- **Privacy without fidelity:** P-PGM and NMF are private, but they lose co-expression.

All three generators here produce a synthetic sample of class `c` as

```
x = mu_c + W^T s + e,     s in R^k (factor scores),   e ~ N(0, diag(r_c))
```

- `W` is a fixed rank-k basis, either fitted on within-class-centred training data or learned on an external non-member cohort.
- `mu_c` and `r_c` are per-class means and per-gene residual variances, shrunk toward the pooled within-class estimate (`--shrink`).
- Only the k-dim factor scores `s` are modelled. The 978-dim residual is drawn independently, so the synthetic
  data is full-rank and does not sit in the span of the training patients.

| `--method` | how the factor scores are generated | hardware |
|---|---|---|
| `gauss` | per-class Gaussian, or a small mixture (`--n_components`) | CPU, seconds |
| `cvae` | small conditional VAE: β=1, z=8, Gaussian decoder sampled with its noise, early stopping | GPU or CPU |
| `diffusion` | small class-conditional MLP DDPM (~0.2M params), early stopping, truncated sampling | GPU or CPU |
| `mvn_camda` | reproduction of the CAMDA MVN baseline, for comparison only | CPU |

## Why: flaws in existing methods that the attacks exploit

| Method | Flaw | Attack signal |
|---|---|---|
| MVN (CAMDA) | Per-class covariance from n_c (100–370) < 978 genes has rank n_c−1. Every synthetic row is the mean plus a mix of that class's training rows. The added noise `N(0, 0.5·Σ̂)` stays in the same subspace. | Members lie *inside* the span of the synthetic rows. MahalaMIA (`pinv`) and the off-span residual reach AUC = TPR = 1.00. GAN-leaks misses it (0.54), so the paper's "moderate risk" for MVN is too low. |
| CVAE | β = 0.001 (effectively an autoencoder), z = 128, about 735 epochs, outputs the decoder mean with no noise. | Samples lie on a manifold fitted to the members: lower reconstruction loss for members (KLeak) and closer distances (GAN-leaks 0.70). |
| CVAE-GMM | The GMM prior is fitted on the posterior means of the real training data. | Sampling lands next to members' codes. |
| Embedded Noisy Diffusion | 25M parameters on about 870 rows, DP noise multiplier 1e-5 (no protection). Likely aggravators (not tested here): SMOTE (interpolated copies of minority-class members) and a QuantileTransformer whose inverse pins genes to the training order statistics. | Lower denoising loss for members (MeLoMIA-ND 0.97). |
| NMF / P-PGM | Private *because* genes are sampled independently within a cluster or label. | Same cause: co-expression is lost. |

FactorGen sits in between. It keeps gene–gene dependence only through k shared factors (k ≪ n_class), and keeps everything else independent per gene.

## Install (GPU VM)

```bash
git clone -b claude/loving-lovelace-bso0vj https://github.com/Filienko/Pro-GENE-GEN-MPC.git
```

```bash
cd Pro-GENE-GEN-MPC/models/FactorGen
```

```bash
pip install numpy pandas pyyaml torch
```

`torch` is only needed for `cvae` and `diffusion`. The `gauss` method needs only numpy, pandas and pyyaml.

## Run

Inputs:
- **Expression:** the CAMDA count TSV (genes × samples). The pre-processed TSVs are in `data/aml/BLUE_TCGA-*`.
- **Labels:** the CAMDA annotation CSV (`samplesID` + `Subtype` for BRCA, `cancer_type` for COMBINED), from the ELSA download (`data/meta/...`).
- **Splits (recommended):** the CAMDA splits YAML (`data_splits/split_indices/<dataset>_splits.yaml`), so the folds match the challenge exactly.
  Without it, a stratified 5-fold split is generated (`--split_seed 42`).

Set the paths once (adjust to the VM):

```bash
BRCA_TSV=../../data/aml/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv
BRCA_ANN=/path/to/data/meta/TCGA-BRCA_primary_tumor_subtypes.csv
BRCA_SPLITS=/path/to/data_splits/split_indices/TCGA-BRCA_splits.yaml
COMB_TSV=../../data/aml/BLUE_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv
COMB_ANN=/path/to/data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv
COMB_SPLITS=/path/to/data_splits/split_indices/TCGA-COMBINED_splits.yaml
```

FactorGauss, the recommended default, on BRCA (all 5 folds, seconds on CPU):

```bash
python run_factorgen.py --method gauss --k 50 --shrink 0.3 --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir results/BRCA/gauss_k50_s0.3
```

FactorDiffusion on GPU:

```bash
python run_factorgen.py --method diffusion --k 50 --shrink 0.3 --device cuda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir results/BRCA/diffusion_k50_s0.3
```

FactorCVAE on GPU:

```bash
python run_factorgen.py --method cvae --k 50 --shrink 0.3 --device cuda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir results/BRCA/cvae_k50_s0.3
```

The CAMDA MVN baseline, for side-by-side comparison:

```bash
python run_factorgen.py --method mvn_camda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir results/BRCA/mvn_camda
```

For COMBINED, swap the variables and use `--label_col cancer_type`, for example:

```bash
python run_factorgen.py --method gauss --k 50 --shrink 0.3 --counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type --splits_yaml $COMB_SPLITS --out_dir results/COMBINED/gauss_k50_s0.3
```

Recommended sweep, combining each method with each setting (each gauss run takes seconds; the neural ones take about a minute per fold on GPU):

```bash
for m in gauss diffusion cvae; do for k in 20 50 100; do for s in 0 0.3; do python run_factorgen.py --method $m --k $k --shrink $s --device cuda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir results/BRCA/${m}_k${k}_s${s}; done; done; done
```

Optional, more private setting: learn the basis on an external non-member cohort. For BRCA, the non-BRCA tumours in COMBINED work;
overlapping sample IDs are dropped automatically. This lowered MahalaMIA from 0.55 to about 0.52–0.54 but cost fidelity (see the results below).

```bash
python run_factorgen.py --method gauss --k 200 --external_counts $COMB_TSV --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir results/BRCA/gauss_ext_k200
```

### Outputs per run directory (CAMDA naming)
- `synthetic_data_split_{k}.csv`: genes only, same size as the training split (`--n_synth` to change).
- `synthetic_labels_split_{k}.csv`: one label column. KLeak reads it by replacing `_data_` with `_labels_`.
- `membership_labels_split_{k}.csv`: `sample_id, membership_label` (1 = member), the red-team `--mmb_labels_file` format.
- `split_{k}_ids.json`, `config.json`.

### Main options
| option | default | meaning |
|---|---|---|
| `--k` | 50 | number of shared factors: larger k gives more co-expression and more leakage |
| `--shrink` | 0.3 | shrinks class factor covariance and residual variance toward the pooled **within-class** estimate (helps small subtypes) |
| `--resid_scale` | 1.0 | multiplier on the independent residual variance (>1 is more private, lower fidelity) |
| `--n_components` | 1 | `gauss`: mixture components per class (components are shrunk toward the class covariance) |
| `--sigma_stop` | 0.1 | `diffusion`: stop the reverse chain at this noise level instead of denoising onto training points |
| `--moment_match` | `mean` (diffusion), `none` (cvae) | re-centre generated scores, or also match the shrunk class covariance (`cov`) |
| `--max_epochs --patience --lr --batch_size --hidden` | per model | training; both neural models early-stop on a held-out 10% of the training split |

## Quick pre-screen before the full attack suite

`sanity_check.py` is not a new attack. It reuses the red-team MahalanobisNoAux (`pinv`) and GAN-leaks scoring, plus the off-span diagnostic.
It also reports per-gene mean and SD error, co-expression TPR and false-edge rate (|r| > 0.3 and 0.5, against the train and held-out networks),
and train-on-synthetic / test-on-real (TSTR) accuracy with a shrinkage LDA:

```bash
python sanity_check.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --syn_dir results/BRCA/mvn_camda results/BRCA/gauss_k50_s0.3 results/BRCA/diffusion_k50_s0.3 results/BRCA/cvae_k50_s0.3 --out_csv results/BRCA/sanity.csv
```

## Full evaluation

Red-team suite ([c-jarrell/PPML-Huskies-Red-Team](https://github.com/c-jarrell/PPML-Huskies-Red-Team), from `src/mia/`, with `attack_model` set in `config.yaml`):

```bash
python red_team.py /abs/path/results/BRCA/gauss_k50_s0.3/synthetic_data_split_1.csv /abs/path/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv gauss_k50 --mmb_labels_file /abs/path/results/BRCA/gauss_k50_s0.3/membership_labels_split_1.csv
```

For COMBINED attacks that need a reference set, add `--reference_file <reference TSV>`.

For CAMDA utility, DE and co-expression: copy a run directory to `data_splits/<dataset>/synthetic/<generator>/<experiment>/`
in the CAMDA starter, then run its `evaluate.py` and the `eval/bio_eval` R scripts as for the other generators.

## Local results (BRCA, folds 1–2, CPU)

These numbers use **k-means pseudo-subtypes**, because the real labels are not in this repo. Re-run with the real annotation file.
MIA columns are AUC / TPR at FPR = 0.1 (members = train split, non-members = held-out split).

| | MahalaMIA | GAN-leaks | off-span | mean / SD err | co-expr r>0.3 TPR / false-edge (train net.) | co-expr r>0.5 TPR / false-edge | TSTR rel. |
|---|---|---|---|---|---|---|---|
| mvn_camda | **1.00 / 1.00** | 0.54 / 0.12 | **1.00 / 1.00** | 0.046 / 0.141 | 0.68 / 0.04 | 0.58 / 0.02 | 0.97 |
| gauss k=50 | 0.55 / 0.10 | 0.53 / 0.12 | 0.59 / 0.13 | 0.045 / 0.042 | 0.90 / 0.17 | 0.86 / 0.23 | 0.98 |
| diffusion k=50 | 0.58 / 0.13 | 0.53 / 0.11 | 0.60 / 0.13 | 0.045 / 0.042 | 0.88 / 0.15 | 0.83 / 0.17 | **1.01** |
| cvae k=50 | 0.59 / 0.13 | 0.53 / 0.12 | 0.60 / 0.13 | 0.053 / 0.042 | 0.87 / 0.14 | 0.82 / 0.15 | 0.99 |
| gauss, external basis k=100 | 0.53 / 0.10 | – | 0.53 | 0.053 SD | 0.67 / 0.12 | 0.54 / – | 0.93 |
| gauss, external basis k=200 | 0.54 / 0.13 | – | 0.55 | 0.053 SD | 0.71 / 0.13 | 0.59 / – | 0.93 |

Reference: a real held-out sample scored against the training network gives a false-edge rate of 0.39 / 0.52. That held-out sample is smaller than the synthetic sets, so its correlations are noisier.

Observations:
- **On the attack scores checked here, the CAMDA MVN is fully identifiable.** Its low false-edge rate comes from copying the training network.
- **The FactorGen variants are near chance on the same scores**, while recovering more true co-expression edges than MVN and matching its utility.
  TPR at FPR = 0.01 was about 0.02 or less for all three (noisy: only about 2 of the 218 non-members set that threshold).
- **FactorCVAE:** the validation ELBO was best at the very start of training. On these near-Gaussian, decorrelated factor scores the CVAE
  adds no structure that generalizes, so early stopping keeps it at roughly a per-class Gaussian. Prefer `gauss` unless the validation ELBO improves on your data.
- **FactorDiffusion:** dropout in the denoiser inflated sample variance by about 1.4×, so dropout now defaults to 0. Remaining mean bias is removed by `--moment_match mean`.
- **External basis:** the most private option, but other-tumour factors fit breast subtypes less well (lower co-expression TPR and utility).

## Caveats
- These are empirical results: no formal DP guarantee. A basis fitted on members still carries some membership signal for an attacker
  who knows the generator family and k. The external-basis option removes that part.
- Fidelity metrics computed against the training members (discriminative score, distance-to-closest, the training co-expression network)
  reward copying. Report the held-out versions alongside them.
