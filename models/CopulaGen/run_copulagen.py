#!/usr/bin/env python
"""Generate synthetic bulk RNA-seq with CopulaGen, fold by fold (CAMDA output layout,
identical to run_factorgen.py: synthetic_data/labels_split_k.csv, split_k_ids.json,
membership_labels_split_k.csv, config.json).

Example
-------
python run_copulagen.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype \
    --splits_yaml $BRCA_SPLITS --k 50 --out_dir results/BRCA/copula_k50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402
from copula_gen import CopulaGen  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", required=True)
    p.add_argument("--annot", default=None)
    p.add_argument("--sample_col", default="samplesID")
    p.add_argument("--label_col", default="Subtype")
    p.add_argument("--splits_yaml", default=None)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--shrink", type=float, default=0.3)
    p.add_argument("--knots", type=int, default=21)
    p.add_argument("--tail", type=float, default=0.05, help="outer knot probability (Gaussian tails beyond)")
    p.add_argument("--shape_shrink", type=float, default=0.0)
    p.add_argument("--gauss_mix", type=float, default=0.0, help="shrink marginal shape toward Gaussian")
    p.add_argument("--n0", type=float, default=50.0, help="shape shrinkage lambda_c >= n0/(n0+n_c)")
    p.add_argument("--marginal", default="quantile", choices=["quantile", "gauss"])
    p.add_argument("--factor", default="gauss", choices=["gauss", "gmm", "kde", "diffusion"])
    p.add_argument("--n_components", type=int, default=3)
    p.add_argument("--kde_h", type=float, default=0.5)
    p.add_argument("--resid_scale", type=float, default=1.0)
    p.add_argument("--external_counts", default=None)
    p.add_argument("--external_annot", default=None, help="labels of the external cohort (within-group basis)")
    p.add_argument("--external_label_col", default="cancer_type")
    p.add_argument("--dp_epsilon", type=float, default=None,
                   help="use DP-CopulaGen with this epsilon (inf = same pipeline, no noise)")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--bins", type=int, default=8, help="DP: histogram bins per gene")
    p.add_argument("--eb", action="store_true",
                   help="DP: empirical-Bayes shrinkage of class shifts (lower W1 / DE-FPR, lower DE-TPR)")
    p.add_argument("--dp_edges", default="data", choices=["data", "external"])
    p.add_argument("--dp_split", type=float, nargs=3, default=[0.02, 0.49, 0.49],
                   help="DP budget shares: class counts, marginals, copula")
    p.add_argument("--clip_mult", type=float, default=1.5)
    p.add_argument("--device", default=None)
    p.add_argument("--sigma_stop", type=float, default=0.1, help="diffusion: truncated reverse chain")
    p.add_argument("--n_synth", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def write_fold(out_dir, f, Xs, ys, genes, label_col, tr_ids, te_ids):
    save_synthetic(out_dir, f, Xs, ys, genes, label_col)
    with open(os.path.join(out_dir, f"split_{f}_ids.json"), "w") as fh:
        json.dump({"train": tr_ids, "test": te_ids}, fh)
    pd.DataFrame({"sample_id": tr_ids + te_ids,
                  "membership_label": [1] * len(tr_ids) + [0] * len(te_ids)}
                 ).to_csv(os.path.join(out_dir, f"membership_labels_split_{f}.csv"), index=False)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    expr = load_expression(a.counts)
    labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[labels.index]
    genes = list(expr.columns)
    external = external_y = None
    if a.external_counts:
        ext = load_expression(a.external_counts)
        ext = ext.drop(index=ext.index.intersection(expr.index)).reindex(columns=genes)
        external = ext.values
        if a.external_annot:
            ea = pd.read_csv(a.external_annot)
            external_y = ea.set_index(ea[a.sample_col].astype(str))[a.external_label_col].astype(str
                                                                                    ).reindex(ext.index).fillna("na").values
    splits = load_splits(a.splits_yaml, expr.index, labels, a.n_splits, a.split_seed)
    log = {"args": vars(a), "folds": {}}
    for f in a.folds or sorted(splits):
        tr_ids = [s for s in splits[f][0] if s in expr.index]
        te_ids = [s for s in splits[f][1] if s in expr.index]
        X, y = expr.loc[tr_ids].values, labels.loc[tr_ids].values
        t0 = time.time()
        if a.dp_epsilon is not None:
            from dp_copula import DPCopulaGen
            gen = DPCopulaGen(epsilon=a.dp_epsilon, delta=a.delta, k=a.k, bins=a.bins, knots=a.knots, tail=a.tail,
                              clip_mult=a.clip_mult, shrink=a.shrink, seed=a.seed + f, external=external,
                              external_y=external_y, edges=a.dp_edges,
                              split=tuple(a.dp_split), eb_shrink=a.eb).fit(X, y)
        else:
          gen = CopulaGen(k=a.k, shrink=a.shrink, knots=a.knots, tail=a.tail, shape_shrink=a.shape_shrink,
                        n0=a.n0, marginal=a.marginal, factor=a.factor, n_components=a.n_components,
                        kde_h=a.kde_h, resid_scale=a.resid_scale, seed=a.seed + f, external=external,
                        device=a.device, diff_kw={"sigma_stop": a.sigma_stop}, gauss_mix=a.gauss_mix).fit(X, y)
        Xs, ys = gen.sample(len(tr_ids) if a.n_synth < 0 else a.n_synth)
        write_fold(a.out_dir, f, Xs, ys, genes, a.label_col, tr_ids, te_ids)
        log["folds"][f] = {"seconds": round(time.time() - t0, 1), "shape_lambda": gen.marg.lam,
                           "dp_info": getattr(gen, "info", None)}
        print(f"[CopulaGen] fold {f}: {Xs.shape} in {time.time() - t0:.1f}s", flush=True)
    with open(os.path.join(a.out_dir, "config.json"), "w") as fh:
        json.dump(log, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
