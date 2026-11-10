#!/usr/bin/env python
"""Generate synthetic bulk RNA-seq with StructPGM (Private-PGM with star / tree / hub structure).

Same inputs and CAMDA-format outputs as models/FactorGen/run_factorgen.py:
    synthetic_data_split_{k}.csv, synthetic_labels_split_{k}.csv,
    membership_labels_split_{k}.csv, split_{k}_ids.json, config.json

Example
-------
python run_structpgm.py --structure hub --epsilon 10 \
    --counts ../../data/aml/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv \
    --annot /path/TCGA-BRCA_primary_tumor_subtypes.csv --label_col Subtype \
    --splits_yaml /path/TCGA-BRCA_splits.yaml --out_dir results/BRCA/pgm_hub_eps10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402
from struct_pgm import StructPGM  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", required=True, help="CAMDA count TSV (genes x samples)")
    p.add_argument("--annot", default=None, help="annotation CSV with sample id + label columns")
    p.add_argument("--sample_col", default="samplesID")
    p.add_argument("--label_col", default="Subtype")
    p.add_argument("--splits_yaml", default=None)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--structure", default="hub", choices=["star", "tree", "hub"],
                   help="star = current P-PGM (gene-label); tree = Chow-Liu gene tree; hub = gene modules + eigengenes")
    p.add_argument("--epsilon", type=float, default=float("inf"), help="DP epsilon (inf = no noise)")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--bins", type=int, default=4, help="quantile bins per gene (4 = current P-PGM)")
    p.add_argument("--hub_bins", type=int, default=4, help="bins per eigengene (hub only)")
    p.add_argument("--modules", type=int, default=40, help="number of gene modules (hub only)")
    p.add_argument("--second_module", action="store_true",
                   help="hub: also tie each gene to its best tree-adjacent second eigengene")
    p.add_argument("--inverse", default="mean", choices=["mean", "smooth"],
                   help="bin -> value: training bin mean (current) or truncated normal within the bin")
    p.add_argument("--num_iters", type=int, default=1000, help="mirror-descent iterations of mbi")
    p.add_argument("--inference", default="direct", choices=["direct", "mbi"],
                   help="direct = exact sampling of the decomposable model from (noisy) clique tables "
                        "(seconds); mbi = Private-PGM mirror descent (slow, needs >= 1000 iters)")
    p.add_argument("--backend", default="numpy", choices=["numpy", "torch"],
                   help="mbi backend; torch uses the GPU when available (much faster)")
    p.add_argument("--external_counts", default=None,
                   help="non-member cohort TSV used for bin edges and structure (tree / modules)")
    p.add_argument("--n_synth", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    expr = load_expression(a.counts)
    labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[labels.index]
    genes = list(expr.columns)
    print(f"[StructPGM] data {expr.shape}, classes: "
          f"{dict(zip(*np.unique(labels.values, return_counts=True)))}")

    external = None
    if a.external_counts:
        ext = load_expression(a.external_counts)
        ext = ext.drop(index=ext.index.intersection(expr.index)).reindex(columns=genes)
        if ext.isna().any().any():
            raise ValueError("external cohort is missing some genes of --counts")
        external = ext.values
        print(f"[StructPGM] external cohort for edges/structure: {ext.shape}")

    splits = load_splits(a.splits_yaml, expr.index, labels, a.n_splits, a.split_seed)
    log = {"args": vars(a), "folds": {}}
    for f in (a.folds or sorted(splits)):
        tr_ids = [s for s in splits[f][0] if s in expr.index]
        te_ids = [s for s in splits[f][1] if s in expr.index]
        X, y = expr.loc[tr_ids].values, labels.loc[tr_ids].values
        n_total = len(tr_ids) if a.n_synth < 0 else a.n_synth
        t0 = time.time()
        print(f"\n[StructPGM] fold {f}: train {len(tr_ids)}, test {len(te_ids)}")
        gen = StructPGM(structure=a.structure, bins=a.bins, hub_bins=a.hub_bins, modules=a.modules,
                        second_module=a.second_module, inverse=a.inverse, epsilon=a.epsilon,
                        delta=a.delta, num_iters=a.num_iters, backend=a.backend,
                        inference=a.inference, seed=a.seed + f)
        gen.fit(X, y, genes, external=external)
        t_fit = time.time() - t0
        Xs, ys = gen.sample(n_total)
        fx, _ = save_synthetic(a.out_dir, f, Xs, ys, genes, a.label_col)
        with open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w") as fh:
            json.dump({"train": tr_ids, "test": te_ids}, fh)
        pd.DataFrame({"sample_id": tr_ids + te_ids,
                      "membership_label": [1] * len(tr_ids) + [0] * len(te_ids)}
                     ).to_csv(os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
        info = dict(cliques=len(gen.cliques), cells=gen.n_cells, sigma=gen.sigma,
                    fit_seconds=round(t_fit, 1), seconds=round(time.time() - t0, 1))
        log["folds"][f] = info
        print(f"[StructPGM] fold {f}: wrote {fx} {Xs.shape} | {info}")
    with open(os.path.join(a.out_dir, "config.json"), "w") as fh:
        json.dump(log, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
