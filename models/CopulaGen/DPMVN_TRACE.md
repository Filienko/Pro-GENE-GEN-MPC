# DP-MVN tracing attack (`dpmvn_trace_attack.py`)

A membership-inference attack tailored to DP-MVN (`dp_mvn.py`).  BRCA, 5 CAMDA folds; all numbers are ROC AUC
(members = a fold's 871 training patients, non-members = its 218 held-out patients).  Settings were fixed on fold 3
only.  Per-fold results: `results_copula/TRACE/BRCA_trace_all.csv`; per-record scores: `results_copula/TRACE/BRCA/*_scores.csv`.

## Idea

DP-MVN releases, through its synthetic rows, one mean per class and gene (the "mean" step, 45 % of the budget).
A member x of class c moves its class mean by about w·x / n_c, where w is its row-clipping factor.  That is 1/n_c of
its own deviation, in all d = 978 genes at once.  One gene says nothing; the sum over 978 genes does.  This is the
"trace amounts" setting of genomic tracing (Homer et al. 2008; Sankararaman et al. 2009; Dwork et al. 2015):

    T_i = (x_i - r_i)' Σ⁻¹ (m_c - r_i)

Here m_c is the synthetic class mean, r_i is the leave-one-out mean of the pool's class-c candidates, and Σ is the
noise covariance of m_c.  T_i is centred on 0 for a non-member and on tr(Σ⁻¹ Σ_x) / n_c for a member, which gives a
separation of about

    sqrt( d σ² / (n_c σ² + σ_DP²) )   standard deviations.

DP-MVN's mean noise (σ_mean = 11.4 at ε=30, 23.6 at ε=10, in clipped-z sum units) is small next to the synthetic
resampling noise of the larger classes (√n_c ≈ 21 for LumA), so most of this separation survives.  None of the
earlier attacks uses it.  MahalaMIA and DCR are distances, not inner products against the release.  DOMIAS works in
PCA-20 and MAMA-MIA in 1-/2-way marginals, both too few dimensions.  The synth-shadow attacks learn from ND or DP-MVN
shadows without this statistic.

## Threat model

The attacker has the same information as kNN-LR and MAMA-MIA here: the synthetic release, the candidate pool
(members and non-members, with subtype labels), and the DP-MVN algorithm with its ε, read from the release's
`config.json`.  There are no shadow models, and no non-member data beyond the pool.  The member share of the pool is
taken as synthetic size / pool size.  `--no_dp` gives the ε-unknown variant.

## What is modelled (all estimated from the pool or the release)

* **Row clipping.**  The mechanism averages rows clipped to C_z, the 75 % quantile of the z-norms.  The centre a,
  scale b and C_z are recomputed on the pool with dp_mvn's own anchor formula, noiselessly.  Each candidate's
  contribution is clip(x).
* **Noise covariance Σ** of v = m_c - r, made of three parts:
  * Sampling noise of the synthetic class mean.  This is a factor model of the synthetic rows, like the generator:
    a shared basis of k̂ factors, a per-class factor covariance shrunk toward the pooled one, and a diagonal residual.
    k̂ is where the flat noise bulk of the standardised within-class spectrum starts; that is the release's k_eff,
    and k̂ = k_eff in every check.
  * Inflation by (1 + non-member share) for the pool reference.
  * The DP noise of the mean step, σ = C_z / sqrt(2 ρ · 0.45) with ρ from ε (dp_mvn's exact Gaussian-DP accounting).
* **Shrinkage.**  The generator shrinks noisy class shifts toward the pooled mean (empirical Bayes).  The attacker
  re-derives the shrinkage weight per class and gene from the synthetic shifts.  A member's weight in gene g is then
  coef_g = 1/n + s_cg (1/n_c - 1/n).
* **Per-record calibration.**  t_i = T_i / sd_i is N(0, 1) for a non-member; for a member it is shifted by δ_i.  The
  score is the log-LR δ_i t_i - δ_i² / 2, which is comparable across classes.  Before that, a moment self-calibration
  on the pool rescales t and δ, using the known member share, so a mis-specified term down-weights itself.
* **Second moment (`trace_var`).**  The same test on |z - class mean| (the MAD step), against the synthetic per-gene
  spread, with a diagonal noise model.  `trace` = `trace_mean` + `trace_var`.

Runtime is seconds per fold, CPU only.

## Results (BRCA, 5 folds)

"Previous strongest" = the highest of MeLoMIA-ND, MAMA-MIA v2, MahalaMIA, kNN-LR and fused, as given in `MIA.md`.

| Release | **Tracing** (worst fold) | TPR@1%FPR | mean part | spread part | k̂ | previous strongest |
|---|---|---|---|---|---|---|
| DP-MVN ε=10000 | **0.889** (0.914) | 0.337 | 0.857 | 0.741 | 30 | 0.675 |
| DP-MVN ε=1000 | **0.886** (0.907) | 0.350 | 0.859 | 0.717 | 30 | 0.669 |
| DP-MVN ε=300 | **0.861** (0.874) | 0.284 | 0.839 | 0.716 | 26 | 0.623 |
| DP-MVN ε=200 | **0.854** (0.868) | 0.201 | 0.829 | 0.712 | 19 | 0.610 |
| DP-MVN ε=100 | **0.842** (0.872) | 0.303 | 0.811 | 0.704 | 10 | 0.598 |
| DP-MVN ε=70 | **0.805** (0.830) | 0.192 | 0.783 | 0.670 | 7 | 0.584 |
| DP-MVN ε=50 | **0.793** (0.822) | 0.146 | 0.764 | 0.667 | 7 | 0.586 |
| DP-MVN ε=30 | **0.737** (0.758) | 0.111 | 0.720 | 0.628 | 3 | 0.569 |
| DP-MVN ε=20 | **0.701** (0.724) | 0.080 | | | 2 | -- |
| DP-MVN ε=10 | **0.633** (0.642) | 0.042 | 0.633 | 0.545 | 1 | 0.561 |
| DP-MVN ε=8 | **0.601** (0.608) | 0.026 | | | 1 | -- |
| DP-MVN ε=5 | 0.539 (0.557) | 0.020 | | | 1 | -- |
| DP-MVN ε=1 | 0.503 (0.534) | 0.013 | | | | 0.516 |
| DP-MVN ε=0.5 | 0.514 (0.540) | 0.012 | | | | 0.512 |
| DP-MVN-copula ε=1000 | 0.862 (0.868) | 0.313 | | | 30 | -- |
| DP-MVN-copula ε=100 | 0.785 (0.797) | 0.168 | | | 9 | -- |
| DP-MVN-copula ε=30 | 0.718 (0.750) | 0.077 | | | 4 | -- |
| DP-MVN-copula ε=10 | 0.613 (0.642) | 0.043 | | | 1 | -- |
| TabPFN hybrid, dpcov 65% | **0.735** (0.749) | 0.116 | | | | 0.578 |
| TabPFN-3.5 prompted with DP-MVN ε30 | **0.670** (0.688) | 0.054 | | | | 0.565 |
| TabPFN-3.5 (real prompt) | 0.708 (0.715) | 0.103 | | | | 0.773 |
| MVN (CAMDA, noise 0.7) | 0.661 (0.710) | 0.089 | | | | 1.000 |
| Private-PGM ε=10 | 0.542 (0.560) | 0.018 | | | | 0.520 |

DP-MVN ε=30, per fold: 0.728 / 0.754 / 0.758 / 0.718 / 0.729 (TPR@10%FPR 0.31 / 0.38 / 0.44 / 0.23 / 0.31).

## Checks

* **Null control.**  I fitted DP-MVN afresh (`dp_mvn.DPMVN`) on a random 871 of the fold-3 pool, 3 seeds per ε.
  Scored against the true membership: AUC 0.878-0.895 at ε=1000 and 0.770-0.777 at ε=30.  Scored against an
  independent random 80/20 labelling: 0.486-0.516.  k̂ equalled the generator's k_eff in all six fits.  The attack
  therefore picks up real membership, not an artefact of the pool or the classes.
* **Ablations** (5 folds, tracing AUC):

  | | full | no clipping model | ε unknown (no DP noise model) | no shrinkage model |
  |---|---|---|---|---|
  | ε=30 | 0.737 | 0.671 | 0.706 | 0.740 |
  | ε=100 | 0.842 | 0.757 | 0.832 | 0.846 |
  | ε=1000 | 0.886 | 0.818 | 0.887 | 0.887 |

  Results: `results_copula/TRACE/BRCA_trace_ablate.csv`.
* The attack is consistent with the DP guarantee: at ε ≤ 1 it is at chance.

## What it means for DP-MVN

* Against this attack, DP-MVN stays under the 0.6 bar only at **ε ≲ 8**.  At ε=30, the setting recommended so far,
  it reaches 0.737 with TPR@1%FPR 0.11.  The flat 0.67 ceiling seen earlier at ε ≥ 1000 was a limit of those attacks,
  not of the leak.
* The leak is the class-mean step.  Any release that carries DP-MVN means or real cluster means inherits it, so the
  TabPFN hybrids fail too (dpcov 65 %: 0.735).  Post-processing by TabPFN only dilutes it (DP-MVN ε30 → TabPFN:
  0.670).
* Untested levers that act directly on this statistic:
  * more noise on the mean step (its share, or a smaller C_z: clipping at the 75 % norm quantile leaves most members
    unclipped);
  * coarser class means (only between-class shifts that clear the noise; pooled mean elsewhere);
  * fewer synthetic rows (more resampling noise in m_c).  This weakens the attack at a fidelity cost and does not
    change the formal ε.

## Usage

```bash
source models/CopulaGen/env.sh
python models/CopulaGen/dpmvn_trace_attack.py --counts $BRCA_TSV --annot $BRCA_ANN \
    --syn_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30 $OUT/EPSWEEP/BRCA/lrdp_eps100 --folds 1 2 3 4 5 \
    --out_csv $OUT/TRACE/BRCA/run.csv            # [--eps E] [--q K] [--no_dp | --no_clip | --no_shrink]
```

Per-record scores (`trace`, `trace_mean`, `trace_var`, ...) are written next to `--out_csv` as
`<out>_<run>_f<fold>_scores.csv`, with columns sample_id / score / member, so they can be fused with the other attacks.
It needs `dp_mvn.py` on the path for the accounting, and falls back to an identical local copy if it is missing.
