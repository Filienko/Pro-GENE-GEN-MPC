# How to run every proposed generator (GPU VM guide)

All generators read the CAMDA inputs and write CAMDA-format outputs to `--out_dir`, so the red-team suite and the
CAMDA evaluation can run on them unchanged:

| file | used by |
|---|---|
| `synthetic_data_split_{k}.csv` (genes only) | red_team.py, evaluate.py, sanity_check.py |
| `synthetic_labels_split_{k}.csv` (one label column) | KLeak (finds it by replacing `_data_` with `_labels_`), evaluate.py |
| `membership_labels_split_{k}.csv` (`sample_id, membership_label`, 1 = member) | red_team.py `--mmb_labels_file` |
| `split_{k}_ids.json`, `config.json` | sanity_check.py, bookkeeping |

| method | script | what it is | hardware | time / fold (BRCA, measured on CPU) |
|---|---|---|---|---|
| `mvn_camda` | FactorGen/run_factorgen.py | the CAMDA MVN baseline (comparison only) | CPU | ~1 s |
| `gauss` | FactorGen/run_factorgen.py | per-class Gaussian on k shared factors + independent per-gene residual | CPU | ~2 s |
| `cvae` | FactorGen/run_factorgen.py | small CVAE on the factor scores | GPU or CPU | ~5 s CPU |
| `diffusion` | FactorGen/run_factorgen.py | small conditional DDPM on the factor scores | GPU or CPU | ~25 s CPU |
| `star` | StructPGM/run_structpgm.py | current P-PGM structure: (gene, label) marginals | CPU | ~3 s |
| `tree` | StructPGM/run_structpgm.py | Chow-Liu gene tree: (gene_i, gene_j, label) | CPU | ~3 s |
| `hub` | StructPGM/run_structpgm.py | gene modules + eigengenes: (gene, eigengene, label) + eigengene tree | CPU | ~3 s |
| `copula` | CopulaGen/run_copulagen.py | class-conditional low-rank Gaussian copula, smoothed per-class marginals, Gaussian or diffusion factor sampler; DP variant via `--dp_epsilon` (see CopulaGen/README.md) | CPU / GPU for diffusion | ~2 s (gauss), ~10 s GPU (diffusion) |
| `lrdp` | CopulaGen/run_lrdp.py | LowRankDP: DP end to end; per-class gene sums and spreads (cost of one pooled sum), empirical-Bayes shrinkage, low-rank within-class covariance from DP-PCA or a public cohort (`--external_counts`). Recommended DP generator (see CopulaGen/README.md) | CPU | ~1.5 s |

## 1. Setup

```bash
git clone -b claude/loving-lovelace-bso0vj https://github.com/Filienko/Pro-GENE-GEN-MPC.git
```

```bash
cd Pro-GENE-GEN-MPC/models
```

```bash
pip install numpy pandas pyyaml scipy networkx torch
```

Set the input paths. The TSVs are in the repo; the annotation CSVs and splits YAMLs come from the CAMDA/ELSA download.
Without `--splits_yaml`, a stratified 5-fold split with seed 42 is generated.

```bash
export BRCA_TSV=$PWD/../data/aml/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv
export BRCA_ANN=/path/to/data/meta/TCGA-BRCA_primary_tumor_subtypes.csv
export BRCA_SPLITS=/path/to/data_splits/split_indices/TCGA-BRCA_splits.yaml
export COMB_TSV=$PWD/../data/aml/BLUE_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv
export COMB_ANN=/path/to/data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv
export COMB_SPLITS=/path/to/data_splits/split_indices/TCGA-COMBINED_splits.yaml
export OUT=$PWD/results
```

## 2. FactorGen (run from `models/FactorGen`)

```bash
cd FactorGen
```

Each generator has a BRCA command. For COMBINED, swap in `$COMB_TSV $COMB_ANN $COMB_SPLITS`, use `--label_col cancer_type`, and use a COMBINED output directory.

CAMDA MVN baseline:

```bash
python run_factorgen.py --method mvn_camda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/mvn_camda
```

FactorGauss (recommended default):

```bash
python run_factorgen.py --method gauss --k 50 --shrink 0.3 --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/gauss_k50
```

FactorDiffusion (GPU):

```bash
python run_factorgen.py --method diffusion --k 50 --shrink 0.3 --device cuda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/diffusion_k50
```

FactorCVAE (GPU):

```bash
python run_factorgen.py --method cvae --k 50 --shrink 0.3 --device cuda --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/cvae_k50
```

COMBINED example:

```bash
python run_factorgen.py --method gauss --k 50 --shrink 0.3 --counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type --splits_yaml $COMB_SPLITS --out_dir $OUT/COMBINED/gauss_k50
```

Optional more-private variant: learn the basis from non-member tumours. For BRCA these are the non-BRCA COMBINED samples; overlapping IDs are dropped automatically.

```bash
python run_factorgen.py --method gauss --k 200 --external_counts $COMB_TSV --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/gauss_ext_k200
```

Key options:
- `--k`: number of factors. Default 50; more factors means more co-expression and more leakage.
- `--shrink`: shrinkage toward the pooled within-class statistics. Default 0.3.
- Adaptive shrinkage is on by default: λ_c = max(shrink, k/n_c), for small subtypes such as Normal-like. Turn it off with `--no_adaptive_shrink`.
- `--resid_scale`: multiplier on the residual noise (>1 is more private).
- `--n_components`: Gaussian mixture components per class, for `gauss`.
- Diffusion: `--sigma_stop`, `--moment_match mean|cov|none`, `--T`.
- CVAE: `--z_dim`, `--beta`.
- Neural models: `--max_epochs --patience --lr --batch_size --hidden`.

## 3. StructPGM (run from `models/StructPGM`)

```bash
cd ../StructPGM
```

4 bins per gene (the current P-PGM discretization) by default. `--epsilon inf` means no noise (empirical privacy); a finite value gives the Gaussian mechanism with RDP calibration.

Current P-PGM structure (baseline):

```bash
python run_structpgm.py --structure star --epsilon inf --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/pgm_star_epsinf
```

Chow-Liu gene tree:

```bash
python run_structpgm.py --structure tree --epsilon inf --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/pgm_tree_epsinf
```

Module hubs (recommended PGM variant):

```bash
python run_structpgm.py --structure hub --modules 40 --second_module --epsilon inf --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/pgm_hub2_epsinf
```

ε sweep, for each structure × ε ∈ {1, 10, inf}:

```bash
for s in star tree hub; do for e in 1 10 inf; do python run_structpgm.py --structure $s --epsilon $e --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/BRCA/pgm_${s}_eps${e}; done; done
```

Key options:
- `--modules` (hub, default 40) and `--second_module` (hub: also tie each gene to a neighbouring eigengene).
- `--bins` and `--hub_bins`: both default 4.
- `--inverse mean|smooth`: bin → value. Default `mean` (training bin mean, as before); `smooth` samples a truncated normal inside the bin.
- `--inference direct|mbi`. `direct` (default) samples the decomposable model straight from the (noisy) clique tables: exact for ε = inf, seconds to run.
  `mbi` runs Private-PGM mirror descent. It needs ≥1000 `--num_iters` to converge on 978 genes, which is about 45–50 min per fold on CPU; try `--backend torch` on the GPU.
- `--external_counts`: take bin edges, the tree and the modules from a non-member cohort.
- `--delta`.

## 4. Quick pre-screen, then full evaluation

From `models/FactorGen`. This reuses the existing MahalaMIA (`pinv`) and GAN-leaks scores and adds fidelity, co-expression and TSTR checks; it is not a new attack.

```bash
cd ../FactorGen && python sanity_check.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --syn_dir $OUT/BRCA/* --out_csv $OUT/BRCA/sanity.csv
```

Red-team suite, from `PPML-Huskies-Red-Team/src/mia` with `attack_model` set in `config.yaml`, once per fold and method:

```bash
python red_team.py $OUT/BRCA/gauss_k50/synthetic_data_split_1.csv $BRCA_TSV gauss_k50 --mmb_labels_file $OUT/BRCA/gauss_k50/membership_labels_split_1.csv
```

For CAMDA utility, DE and co-expression, copy a run directory to `data_splits/<dataset>/synthetic/<generator>/<experiment>/` and run `evaluate.py` and `eval/bio_eval`.

## 5. What to expect (local check: BRCA, k-means pseudo-subtypes, CPU, mean of folds 1–2)

These are pre-screen numbers, not the red-team suite. MAMA-MIA, KLeak and synth_shadow still need to run on the VM, with the real labels.

| method | MahalaMIA AUC | GAN-leaks AUC | co-expr r>0.3 TPR / false-edge | co-expr r>0.5 TPR / false-edge | gene SD error | TSTR rel. |
|---|---|---|---|---|---|---|
| mvn_camda | **1.00** | 0.54 | 0.68 / 0.04 | 0.58 / 0.02 | 0.141 | 0.97 |
| gauss k=50 | 0.57 | 0.53 | **0.90 / 0.18** | **0.87 / 0.25** | 0.042 | 0.99 |
| diffusion k=50 | 0.58 | 0.53 | 0.88 / 0.15 | 0.83 / 0.17 | 0.042 | 1.01 |
| cvae k=50 | 0.59 | 0.53 | 0.87 / 0.14 | 0.82 / 0.15 | 0.042 | 0.99 |
| pgm star, ε=inf | 0.53 | 0.52 | 0.07 / 0.00 | 0.00 / 0.00 | 0.078 | 1.10 |
| pgm tree, ε=inf | 0.50 | 0.52 | 0.14 / 0.00 | 0.15 / 0.00 | 0.077 | 1.05 |
| pgm hub, ε=inf | 0.53 | 0.52 | 0.25 / 0.02 | 0.20 / 0.01 | 0.078 | 1.01 |
| pgm hub + second module, ε=inf | 0.54 | 0.52 | **0.28 / 0.02** | **0.21 / 0.01** | 0.077 | 0.99 |
| pgm star / tree / hub, ε=10 | 0.51–0.52 | 0.51 | ≤ 0.06 | ≈ 0 | 0.09 | 0.97–1.09 |

The gauss row uses the current defaults, with adaptive shrinkage on. The diffusion and cvae rows were measured before adaptive shrinkage was added;
for gauss it changed every number by ≤ 0.02.
TSTR rel. above 1 means the shrinkage-LDA trained on synthetic data scored higher on held-out real data than the one trained on real data. It reflects the classifier, not a problem with the generator.

Notes:
- **StructPGM.** Hub roughly triples the co-expression of the current star structure at the same 4 bins, with near-chance attack scores. At ε = 10,
  noise of about 25 counts per cell, spread over about 1,000 cliques, wipes out the gene–gene signal in every structure. The richer structures pay off only at large ε or without DP,
  so MAMA-MIA should decide which setting is acceptable.
- **FactorCVAE.** It collapses to roughly a per-class Gaussian: its validation ELBO is best at the start of training. Prefer `gauss`.
- **Structure and bins come from training data.** In both packages, structure choices (basis, tree, modules) and bin edges/means are computed from the training data without noise,
  as in the existing P-PGM. The formal DP guarantee of StructPGM covers the marginals only; use `--external_counts` to remove that dependence.
