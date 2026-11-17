#!/usr/bin/env python
"""Cheap stand-in for tabpfn_lr_attack.py: the same "prefix -> next gene" likelihood ratio, with a k-nearest-neighbour
conditional model instead of TabPFN, so it can use every gene in the generation order in seconds.

For target gene j (column order of the generator: subtype, then genes in file order) and candidate x:
  neighbours of x among the release (same subtype) on a random subset F(j) of the genes generated before j;
  Gaussian log-density of x_j under those neighbours' j-values (kernel mean / variance), minus the same with the
  real candidate pool as reference (cross-fitted, x never its own neighbour).  Score = sum over targets.

python knn_lr_attack.py --counts ... --syn_dir DIR --folds 1 2 3 4 5 --out_csv x.csv [--k 10 --n_feat 50]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--syn_dir", nargs="+", required=True); p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--k", type=int, default=10); p.add_argument("--n_feat", type=int, default=50)
p.add_argument("--min_pos", type=int, default=100); p.add_argument("--n_targets", type=int, default=0, help="0 = all")
p.add_argument("--kfold", type=int, default=5); p.add_argument("--same_class", type=int, default=1)
p.add_argument("--seed", type=int, default=0); p.add_argument("--out_csv", required=True)
a = p.parse_args()

expr = load_expression(a.counts); lab = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[lab.index]; genes = list(expr.columns); classes = sorted(lab.unique())
sd_all = expr.values.std(0) + 1e-6


def logdens(Q, yq, cq, R, yr, cr, feats, j, k):
    """log N(x_j; mean, var) of each query's k nearest reference rows on feats (same class if asked)."""
    out = np.empty(len(Q))
    for c in np.unique(cq):
        qi = np.where(cq == c)[0] if a.same_class else np.arange(len(Q))
        ri = np.where(cr == c)[0] if a.same_class else np.arange(len(R))
        if a.same_class and len(ri) < k + 1:
            ri = np.arange(len(R))
        A, B = Q[qi][:, feats], R[ri][:, feats]
        d = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None]
        nn = np.argpartition(d, k, axis=1)[:, :k]
        v = R[ri][:, j][nn]; m, s2 = v.mean(1), v.var(1) + (0.05 * sd_all[j - 1]) ** 2
        out[qi] = -0.5 * (np.log(2 * np.pi * s2) + (Q[qi][:, j] - m) ** 2 / s2)
        if not a.same_class:
            break
    return out


rows = []
for sd_ in a.syn_dir:
    run = os.path.basename(os.path.normpath(sd_))
    for f in a.folds:
        if not os.path.exists(os.path.join(sd_, f"synthetic_data_split_{f}.csv")):
            continue
        rng = np.random.default_rng(a.seed + f)
        ids = json.load(open(os.path.join(sd_, f"split_{f}_ids.json")))
        cand = ids["train"] + ids["test"]; mem = np.r_[np.ones(len(ids["train"])), np.zeros(len(ids["test"]))]
        C = np.column_stack([np.zeros(len(cand)), expr.loc[cand, genes].values / sd_all])
        cC = np.array([classes.index(c) for c in lab.loc[cand]])
        S = np.column_stack([np.zeros(sum(1 for _ in open(os.path.join(sd_, f"synthetic_data_split_{f}.csv"))) - 1),
                             pd.read_csv(os.path.join(sd_, f"synthetic_data_split_{f}.csv"))[genes].values / sd_all])
        cS = np.array([classes.index(c) for c in pd.read_csv(os.path.join(sd_, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]])
        kf = rng.permutation(len(C)) % a.kfold
        pos = np.arange(max(a.min_pos, 2), C.shape[1])
        targets = np.sort(rng.choice(pos, a.n_targets, replace=False)) if a.n_targets else pos
        L = np.zeros((len(C), len(targets)))
        for t, j in enumerate(targets):
            feats = np.sort(rng.choice(np.arange(1, j), min(a.n_feat, j - 1), replace=False))
            ls = logdens(C, None, cC, S, None, cS, feats, j, a.k)
            lr = np.empty(len(C))
            for q in range(a.kfold):
                m_ = kf == q
                lr[m_] = logdens(C[m_], None, cC[m_], C[~m_], None, cC[~m_], feats, j, a.k)
            L[:, t] = ls - lr
        for name, sc in {"knn_lr": L.sum(1), "knn_lr_clip": np.clip(L, -3, 3).sum(1)}.items():
            fpr, tpr, _ = roc_curve(mem, sc)
            rows.append({"run": run, "fold": f, "score": name, "auc": roc_auc_score(mem, sc),
                         "tpr@0.01": float(np.interp(0.01, fpr, tpr)), "tpr@0.1": float(np.interp(0.1, fpr, tpr)),
                         "k": a.k, "n_feat": a.n_feat, "n_targets": len(targets)})
        pd.DataFrame({"sample_id": cand, "knn_lr": L.sum(1), "knn_lr_clip": np.clip(L, -3, 3).sum(1), "member": mem}).to_csv(
            a.out_csv.replace(".csv", f"_{run}_f{f}_scores.csv"), index=False)
        print(run, f, " ".join(f"{r['score']}={r['auc']:.3f}/{r['tpr@0.01']:.3f}" for r in rows[-2:]), flush=True)
pd.DataFrame(rows).to_csv(a.out_csv, index=False)
