#!/usr/bin/env python
"""MAMA-MIA, the scoring of MIA_on_bulkRNAseq_CAMDA2026/mia/attacks/mamamia.py (Golob et al. reimplementation),
ported to read our run directories.  Same choices as that file: quantile bin edges and the auxiliary baseline
both from the full labelled real pool, one-way per gene + two-way gene x label + label marginal, aggregation =
mean ratio p_syn / p_aux.  Candidates of fold k = its training (members) + test (non-members) samples.

python mama_lite.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --bins 4 10 --syn_dir d1 d2 --out_csv x.csv
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FactorGen"))
from factor_core import load_expression, load_labels  # noqa: E402

EPS = 1e-10
p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--syn_dir", nargs="+", required=True); p.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
p.add_argument("--bins", type=int, nargs="+", default=[4, 10]); p.add_argument("--out_csv", required=True)
a = p.parse_args()

expr = load_expression(a.counts)
lab = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[lab.index]
classes = sorted(lab.unique()); C = len(classes)
X_aux = expr.values.astype(np.float64); y_aux = np.array([classes.index(c) for c in lab])
G = X_aux.shape[1]; gi = np.arange(G)


def digitize(X, edges, k):
    return np.clip(np.stack([np.searchsorted(edges[j], X[:, j], side="right") for j in range(X.shape[1])], 1), 0, k - 1)


def oneway(b, k):
    codes = (gi * k)[None, :] + b
    return np.bincount(codes.ravel(), minlength=G * k).reshape(G, k) / len(b)


def twoway(b, y, k):
    codes = (gi * k * C)[None, :] + b * C + y[:, None]
    return np.bincount(codes.ravel(), minlength=G * k * C).reshape(G, k, C) / len(b)


def ratio(ps, pa):
    return np.maximum(ps, EPS) / np.maximum(pa, EPS)


rows = []
for d in a.syn_dir:
    run = os.path.basename(os.path.normpath(d))
    for f in a.folds:
        fx = os.path.join(d, f"synthetic_data_split_{f}.csv")
        if not os.path.exists(fx):
            continue
        Xs = pd.read_csv(fx).values.astype(np.float64)
        ys = np.array([classes.index(c) for c in pd.read_csv(os.path.join(d, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]])
        ids = json.load(open(os.path.join(d, f"split_{f}_ids.json")))
        cand = ids["train"] + ids["test"]; mem = np.r_[np.ones(len(ids["train"])), np.zeros(len(ids["test"]))]
        Xc = expr.loc[cand].values; yc = np.array([classes.index(c) for c in lab.loc[cand]])
        for k in a.bins:
            edges = np.quantile(X_aux, np.linspace(0, 1, k + 1)[1:-1], axis=0).T
            bc, bs, ba = digitize(Xc, edges, k), digitize(Xs, edges, k), digitize(X_aux, edges, k)
            s1 = ratio(oneway(bs, k)[gi, bc], oneway(ba, k)[gi, bc]).sum(1)
            s2 = ratio(twoway(bs, ys, k)[gi, bc, yc[:, None]], twoway(ba, y_aux, k)[gi, bc, yc[:, None]]).sum(1)
            s0 = ratio(np.bincount(ys, minlength=C)[yc] / len(ys), np.bincount(y_aux, minlength=C)[yc] / len(y_aux))
            score = (s1 + s2 + s0) / (2 * G + 1)
            rows.append({"run": run, "fold": f, "bins": k, "auc": roc_auc_score(mem, score)})
        print(run, f, " ".join(f"k{r['bins']}={r['auc']:.3f}" for r in rows if r["run"] == run and r["fold"] == f), flush=True)
pd.DataFrame(rows).to_csv(a.out_csv, index=False)
