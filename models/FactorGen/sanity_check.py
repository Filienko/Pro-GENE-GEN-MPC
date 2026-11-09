#!/usr/bin/env python
"""Fast pre-screen of synthetic data before running the full red-team suite.

This is NOT a new attack.  It re-uses the scoring formulas of attacks that
already exist (red-team MahalanobisNoAux with pinv, GAN-leaks) plus one
diagnostic (residual outside the span of the synthetic rows, the property
that makes the CAMDA MVN fully identifiable), and reports fidelity.

For each fold in <syn_dir> (written by run_factorgen.py):
  privacy : AUC and TPR@FPR=0.1 / 0.01, members = train ids, non-members = test ids
  fidelity: per-gene mean / SD error vs held-out real, correlation-matrix MAE vs held-out,
            co-expression TPR and false-edge rate at |r|>0.3 / 0.5
            against the TRAIN network and against the HELD-OUT network
  utility : train-on-synthetic / test-on-real accuracy with a shrinkage LDA
            (relative to train-on-real), numpy only

Usage:
python sanity_check.py --counts <TSV> --syn_dir <out_dir> [--folds 1 2]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_core import load_expression  # noqa: E402


def auc(s_in, s_out):
    s = np.r_[s_in, s_out]
    r = s.argsort().argsort() + 1
    return (r[:len(s_in)].sum() - len(s_in) * (len(s_in) + 1) / 2) / (len(s_in) * len(s_out))


def tpr_at(s_in, s_out, fpr):
    return float((s_in > np.quantile(s_out, 1 - fpr)).mean())


def psd(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.maximum(w, 0)) @ V.T


def mahalanobis_noaux(S, T):
    """Red-team MahalanobisNoAux score: 1 / Mahalanobis distance to the synthetic
    mean under pinv(synthetic covariance)."""
    mu = S.mean(0)
    Ci = np.linalg.pinv(psd(np.cov(S, rowvar=False)))
    D = T - mu
    return 1 / (np.sqrt(np.abs(np.einsum("ij,jk,ik->i", D, Ci, D))) + 1e-10)


def gan_leaks(S, T):
    """GAN-leaks: negative distance to the closest synthetic record."""
    out = np.empty(len(T))
    s2 = (S ** 2).sum(1)
    for i in range(0, len(T), 256):
        B = T[i:i + 256]
        d2 = (B ** 2).sum(1)[:, None] + s2[None] - 2 * B @ S.T
        out[i:i + 256] = -np.sqrt(np.maximum(d2, 0).min(1))
    return out


def off_span(S, T):
    """Diagnostic: residual norm outside the affine span of the synthetic rows
    (members of the CAMDA MVN have ~0 residual)."""
    Sc = S - S.mean(0)
    _, s, Vt = np.linalg.svd(Sc, full_matrices=False)
    r = int((s > s[0] * 1e-8).sum())
    D = T - S.mean(0)
    return -np.linalg.norm(D - (D @ Vt[:r].T) @ Vt[:r], axis=1)


def coex(S, R_ref, iu, thr):
    Rs = np.corrcoef(S, rowvar=False)[iu]
    real, syn = np.abs(R_ref) > thr, np.abs(Rs) > thr
    return (real & syn).sum() / max(real.sum(), 1), (syn & ~real).sum() / max(real.sum(), 1)


def lda_fit(X, y, alpha=0.5):
    classes = np.unique(y)
    mu = np.array([X[y == c].mean(0) for c in classes])
    Z = np.vstack([X[y == c] - mu[i] for i, c in enumerate(classes)])
    C = np.cov(Z, rowvar=False)
    C = (1 - alpha) * C + alpha * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C)
    return classes, mu, P


def lda_acc(model, X, y):
    classes, mu, P = model
    sc = X @ P @ mu.T - 0.5 * np.einsum("ij,jk,ik->i", mu, P, mu)[None]
    return float((classes[sc.argmax(1)] == y).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", required=True)
    p.add_argument("--syn_dir", required=True, nargs="+", help="one or more run_factorgen output dirs")
    p.add_argument("--annot", default=None, help="needed for the TSTR utility check")
    p.add_argument("--sample_col", default="samplesID")
    p.add_argument("--label_col", default="Subtype")
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--out_csv", default=None)
    a = p.parse_args()

    expr = load_expression(a.counts)
    labels = None
    if a.annot:
        ann = pd.read_csv(a.annot)
        labels = ann.set_index(ann[a.sample_col].astype(str))[a.label_col].astype(str)
    iu = None
    rows = []
    for sd in a.syn_dir:
        files = sorted(glob.glob(os.path.join(sd, "synthetic_data_split_*.csv")))
        for fx in files:
            f = int(re.findall(r"split_(\d+)", fx)[-1])
            if a.folds and f not in a.folds:
                continue
            ids = json.load(open(os.path.join(sd, f"split_{f}_ids.json")))
            Sdf = pd.read_csv(fx)
            S = Sdf.values.astype(float)
            Xtr = expr.loc[ids["train"], Sdf.columns].values
            Xte = expr.loc[ids["test"], Sdf.columns].values
            if iu is None:
                iu = np.triu_indices(S.shape[1], 1)
            r = {"run": os.path.basename(os.path.normpath(sd)), "fold": f}
            for name, fn in [("mahala", mahalanobis_noaux), ("ganleaks", gan_leaks), ("offspan", off_span)]:
                si, so = fn(S, Xtr), fn(S, Xte)
                r[f"{name}_auc"] = auc(si, so)
                r[f"{name}_tpr@0.1"] = tpr_at(si, so, 0.1)
                r[f"{name}_tpr@0.01"] = tpr_at(si, so, 0.01)
            r["mean_err"] = np.abs(S.mean(0) - Xte.mean(0)).mean()
            r["sd_err"] = np.abs(S.std(0) - Xte.std(0)).mean()
            R_tr = np.corrcoef(Xtr, rowvar=False)[iu]
            R_te = np.corrcoef(Xte, rowvar=False)[iu]
            r["corr_mae_test"] = np.abs(np.corrcoef(S, rowvar=False)[iu] - R_te).mean()
            for thr in (0.3, 0.5):
                r[f"coex{thr}_tpr_train"], r[f"coex{thr}_fer_train"] = coex(S, R_tr, iu, thr)
                r[f"coex{thr}_tpr_test"], r[f"coex{thr}_fer_test"] = coex(S, R_te, iu, thr)
            if labels is not None:
                ys = pd.read_csv(fx.replace("_data_", "_labels_")).iloc[:, 0].astype(str).values
                ytr, yte = labels.loc[ids["train"]].values, labels.loc[ids["test"]].values
                acc_syn = lda_acc(lda_fit(S, ys), Xte, yte)
                acc_real = lda_acc(lda_fit(Xtr, ytr), Xte, yte)
                r["tstr_acc"], r["trtr_acc"], r["tstr_rel"] = acc_syn, acc_real, acc_syn / acc_real
            rows.append(r)
            print(f"{r['run']:28s} fold {f}: Mahala AUC {r['mahala_auc']:.3f} | GAN-leaks AUC "
                  f"{r['ganleaks_auc']:.3f} | off-span AUC {r['offspan_auc']:.3f} | coex>0.3 TPR/FER(train) "
                  f"{r['coex0.3_tpr_train']:.2f}/{r['coex0.3_fer_train']:.2f}"
                  + (f" | TSTR rel {r['tstr_rel']:.3f}" if 'tstr_rel' in r else ""))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== mean over folds ===")
    print(df.drop(columns="fold").groupby("run").mean().round(3).T)
    if a.out_csv:
        df.to_csv(a.out_csv, index=False)
        print(f"saved {a.out_csv}")


if __name__ == "__main__":
    main()
