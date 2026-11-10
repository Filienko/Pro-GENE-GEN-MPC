#!/usr/bin/env python
"""Generate synthetic bulk RNA-seq with the FactorGen family, fold by fold.

Outputs (CAMDA layout, directly usable by the red-team suite and evaluate.py):
    <out_dir>/synthetic_data_split_{k}.csv      genes only, one row per synthetic sample
    <out_dir>/synthetic_labels_split_{k}.csv    one label column
    <out_dir>/split_{k}_ids.json               member (train) / non-member (test) sample ids
    <out_dir>/membership_labels_split_{k}.csv   same, in the red-team --mmb_labels_file format
    <out_dir>/config.json                       all arguments + per-fold info

Examples
--------
python run_factorgen.py --method gauss \
    --counts ../../data/aml/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv \
    --annot  /path/TCGA-BRCA_primary_tumor_subtypes.csv --label_col Subtype \
    --splits_yaml /path/TCGA-BRCA_splits.yaml --k 50 --shrink 0.3 \
    --out_dir results/BRCA/factor_gauss_k50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_core import (fit_factor_model, load_expression, load_labels,  # noqa: E402
                         load_splits, save_synthetic)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--counts", required=True, help="CAMDA count TSV (genes x samples)")
    d.add_argument("--annot", default=None, help="annotation CSV with sample id + label columns")
    d.add_argument("--sample_col", default="samplesID")
    d.add_argument("--label_col", default="Subtype", help="'Subtype' (BRCA) or 'cancer_type' (COMBINED)")
    d.add_argument("--splits_yaml", default=None,
                   help="CAMDA splits YAML (data_splits/split_indices/<dataset>_splits.yaml). "
                        "If omitted, a stratified K-fold split is generated with --split_seed.")
    d.add_argument("--n_splits", type=int, default=5)
    d.add_argument("--split_seed", type=int, default=42)
    d.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds, e.g. 1 2")
    d.add_argument("--out_dir", required=True)

    m = p.add_argument_group("model (shared)")
    m.add_argument("--method", required=True, choices=["gauss", "cvae", "diffusion", "mvn_camda"],
                   help="mvn_camda reproduces the CAMDA MVN baseline for comparison")
    m.add_argument("--k", type=int, default=50, help="number of shared factors")
    m.add_argument("--shrink", type=float, default=0.3,
                   help="shrinkage of class factor cov / residual var toward pooled within-class")
    m.add_argument("--no_adaptive_shrink", action="store_true",
                   help="disable lambda_c = max(shrink, k/n_c) for small classes")
    m.add_argument("--external_counts", default=None,
                   help="optional TSV (genes x samples) of a NON-member cohort used to learn the basis "
                        "(e.g. other TCGA tumours). Samples that also occur in --counts are dropped.")
    m.add_argument("--resid_scale", type=float, default=1.0, help="multiplier on residual variance")
    m.add_argument("--n_synth", type=int, default=-1, help="-1 = same size as the training split")
    m.add_argument("--uniform_y", action="store_true", help="balanced classes instead of training proportions")
    m.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("gauss")
    g.add_argument("--n_components", type=int, default=1, help="Gaussian mixture components per class")

    n = p.add_argument_group("cvae / diffusion")
    n.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    n.add_argument("--max_epochs", type=int, default=None)
    n.add_argument("--patience", type=int, default=None)
    n.add_argument("--batch_size", type=int, default=None)
    n.add_argument("--lr", type=float, default=None)
    n.add_argument("--hidden", type=int, default=None)
    n.add_argument("--z_dim", type=int, default=8, help="CVAE latent size")
    n.add_argument("--beta", type=float, default=1.0, help="CVAE KL weight")
    n.add_argument("--T", type=int, default=200, help="diffusion steps")
    n.add_argument("--moment_match", default=None, choices=["none", "mean", "cov"],
                   help="post-process generated factor scores: re-centre (mean) or also match the shrunk "
                        "class covariance (cov). Default: none for cvae, mean for diffusion")
    n.add_argument("--dropout", type=float, default=None, help="diffusion denoiser dropout (default 0)")
    n.add_argument("--sigma_stop", type=float, default=0.1,
                   help="diffusion: stop the reverse chain at this noise-to-signal level (0 = full)")
    return p.parse_args()


def mvn_camda(X, y, n_total, rng, noise_level=0.5):
    """CAMDA MVN baseline (per-class full covariance + noise drawn from the same
    covariance).  Included only as a comparison point."""
    classes, counts = np.unique(y, return_counts=True)
    Xs, ys = [], []
    for c, cnt in zip(classes, counts):
        Z = X[y == c]
        C = np.cov(Z, rowvar=False)
        w, V = np.linalg.eigh((C + C.T) / 2)
        L = V * np.sqrt(np.maximum(w, 0))
        n_c = int(round(n_total * cnt / len(y)))
        Xs.append(Z.mean(0) + rng.standard_normal((n_c, X.shape[1])) @ L.T * np.sqrt(1 + noise_level))
        ys += [c] * n_c
    X, y = np.vstack(Xs), np.array(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def nn_kwargs(a, keys):
    return {k: getattr(a, k) for k in keys if getattr(a, k, None) is not None}


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    expr = load_expression(a.counts)
    labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[labels.index]
    genes = list(expr.columns)
    print(f"[FactorGen] data {expr.shape}, classes: "
          f"{dict(zip(*np.unique(labels.values, return_counts=True)))}")

    external = None
    if a.external_counts:
        ext = load_expression(a.external_counts)
        overlap = ext.index.intersection(expr.index)
        ext = ext.drop(index=overlap)
        ext = ext.reindex(columns=genes)
        if ext.isna().any().any():
            raise ValueError("external cohort is missing some genes of --counts")
        external = ext.values
        print(f"[FactorGen] external basis cohort {ext.shape} ({len(overlap)} overlapping samples dropped)")

    splits = load_splits(a.splits_yaml, expr.index, labels, a.n_splits, a.split_seed)
    folds = a.folds or sorted(splits)
    log = {"args": vars(a), "folds": {}}

    for f in folds:
        tr_ids, te_ids = splits[f]
        tr_ids = [s for s in tr_ids if s in expr.index]
        te_ids = [s for s in te_ids if s in expr.index]
        X, y = expr.loc[tr_ids].values, labels.loc[tr_ids].values
        n_total = len(tr_ids) if a.n_synth < 0 else a.n_synth
        rng = np.random.default_rng(a.seed + f)
        t0 = time.time()
        print(f"\n[FactorGen] fold {f}: train {len(tr_ids)}, test {len(te_ids)}, method {a.method}")

        info = {}
        if a.method == "mvn_camda":
            Xs, ys = mvn_camda(X, y, n_total, rng)
        else:
            fm = fit_factor_model(X, y, k=a.k, shrink=a.shrink, external=external,
                                  adaptive_shrink=not a.no_adaptive_shrink)
            info["shrink_per_class"] = {c: round(v, 3) for c, v in fm.shrink_used.items()}
            if a.method == "gauss":
                from factor_gauss import FactorGauss
                gen = FactorGauss(n_components=a.n_components, resid_scale=a.resid_scale,
                                  seed=a.seed + f).fit(fm)
            elif a.method == "cvae":
                from factor_cvae import FactorCVAE
                kw = nn_kwargs(a, ["max_epochs", "patience", "batch_size", "lr", "hidden", "moment_match"])
                gen = FactorCVAE(z_dim=a.z_dim, beta=a.beta, resid_scale=a.resid_scale,
                                 seed=a.seed + f, device=a.device, **kw).fit(fm)
                info["epochs"] = gen.epochs_trained
            else:
                from factor_diffusion import FactorDiffusion
                kw = nn_kwargs(a, ["max_epochs", "patience", "batch_size", "lr", "hidden",
                                   "moment_match", "dropout"])
                gen = FactorDiffusion(T=a.T, sigma_stop=a.sigma_stop, resid_scale=a.resid_scale,
                                      seed=a.seed + f, device=a.device, **kw).fit(fm)
                info.update(epochs=gen.epochs_trained, n_params=gen.n_params, t_stop=gen.t_stop)
            Xs, ys = gen.sample(n_total, uniform_y=a.uniform_y)
            info["k"] = int(fm.W.shape[0])

        fx, fy = save_synthetic(a.out_dir, f, Xs, ys, genes, a.label_col)
        with open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w") as fh:
            json.dump({"train": tr_ids, "test": te_ids}, fh)
        # ground truth for the red-team suite (--mmb_labels_file): 1 = member (train), 0 = non-member
        pd.DataFrame({"sample_id": tr_ids + te_ids,
                      "membership_label": [1] * len(tr_ids) + [0] * len(te_ids)}
                     ).to_csv(os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
        info.update(n_synth=len(ys), seconds=round(time.time() - t0, 1))
        log["folds"][f] = info
        print(f"[FactorGen] fold {f}: wrote {fx} ({Xs.shape}) in {info['seconds']}s")

    with open(os.path.join(a.out_dir, "config.json"), "w") as fh:
        json.dump(log, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
