#!/usr/bin/env python
"""Fidelity / utility / privacy evaluation of synthetic bulk RNA-seq runs.

Reads run directories in the CAMDA layout written by run_factorgen.py,
run_structpgm.py and run_copulagen.py:
    synthetic_data_split_{k}.csv, synthetic_labels_split_{k}.csv, split_{k}_ids.json

Reference = the held-out real split (test ids) unless stated otherwise.

fidelity  w1           mean per-gene 1-Wasserstein distance (paper metric), pooled
          w1_cls       same, within each class, averaged over classes (weighted by size)
          mean_err / sd_err   per-gene mean / SD absolute error
          mmd          RBF-kernel MMD^2 on standardised genes (median heuristic)
          disc_auc     5-fold CV AUC of a logistic regression real-test vs synthetic (0.5 = indistinguishable)
          prec / rec / dens / cov   k-NN precision, recall, density, coverage (k=5, PCA-50 space)
          corr_mae     mean |r_syn - r_real| over gene pairs
          coex{t}_tpr / _fer   co-expression edges |r|>t found / false edges per true edge
          de_tpr / de_fpr / de_rho  one-vs-rest DE genes (Welch t, BH 5%, |diff|>0.5) of real TRAIN
                                    recovered in synthetic; de_rho = Spearman of per-gene effect sizes
utility   tstr_acc / tstr_f1 / trtr_acc   CAMDA logistic regression (scaler fitted on real train)
privacy   members = train ids, non-members = test ids.  Reference half of the test set is used
          as attacker auxiliary data where an attack needs one; scores on the other half.
          mahala_auc   MahalanobisNoAux (pinv synthetic cov)
          dcr_auc      GAN-leaks / distance to closest synthetic record
          dcrr_auc     distance ratio  d(x, syn) / d(x, ref)
          domias_auc   DOMIAS-style log p_syn(x) - log p_ref(x) (Gaussian KDE, PCA-20)
          marg_auc     marginal likelihood ratio sum_g log p_syn(x_g)/p_ref(x_g) (MAMA-MIA-like, targets
                       marginal-based generators such as Private-PGM)
          dcr_share    fraction of synthetic rows closer to train than to an equal-size test subset (ideal 0.5)
Use --oracle to add a pseudo-run 'REAL_TRAIN' that treats the real training split as synthetic data.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FactorGen"))
from factor_core import load_expression  # noqa: E402


warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- helpers
def auc(s_in, s_out):
    s = np.r_[s_in, s_out]
    r = stats.rankdata(s)
    return (r[:len(s_in)].sum() - len(s_in) * (len(s_in) + 1) / 2) / (len(s_in) * len(s_out))


def tpr_at(s_in, s_out, fpr):
    return float((s_in > np.quantile(s_out, 1 - fpr)).mean())


def sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0)


def nn_dist(A, B, k=1, exclude_self=False):
    D = np.sqrt(sqdist(A, B))
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, 1)[:, k - 1]


def w1_per_gene(A, B, q=200):
    """1-Wasserstein per column via quantile functions."""
    qs = (np.arange(q) + 0.5) / q
    return np.abs(np.quantile(A, qs, axis=0) - np.quantile(B, qs, axis=0)).mean(0)


def mmd_rbf(A, B, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    Z = np.vstack([A, B])
    D = sqdist(Z, Z)
    g = 1.0 / np.median(D[D > 0])
    K = np.exp(-g * D)
    n = len(A)
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    m = len(B)
    return ((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
            - 2 * Kab.mean())


def prdc(real, fake, k=5):
    """Naeem et al. 2020 precision / recall / density / coverage."""
    r_rad = nn_dist(real, real, k + 1)  # k-th neighbour excluding self (self is distance 0)
    f_rad = nn_dist(fake, fake, k + 1)
    D_rf = np.sqrt(sqdist(real, fake))
    prec = (D_rf < r_rad[:, None]).any(0).mean()
    rec = (D_rf < f_rad[None]).any(1).mean()
    dens = (D_rf < r_rad[:, None]).sum(0).mean() / k
    cov = (D_rf.min(1) < r_rad).mean()
    return prec, rec, dens, cov


def bh(p, alpha=0.05):
    n = len(p)
    o = np.argsort(p)
    thr = alpha * np.arange(1, n + 1) / n
    passed = p[o] <= thr
    out = np.zeros(n, bool)
    if passed.any():
        out[o[:np.max(np.where(passed)[0]) + 1]] = True
    return out


def de_sets(X, y, classes, min_diff=0.5):
    sets, effects = {}, {}
    for c in classes:
        a, b = X[y == c], X[y != c]
        if len(a) < 3 or len(b) < 3:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        diff = a.mean(0) - b.mean(0)
        p = np.nan_to_num(p, nan=1.0)
        sets[c] = bh(p) & (np.abs(diff) > min_diff)
        effects[c] = diff
    return sets, effects


def disc_auc(R, S, seed=0, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    if len(S) > len(R):
        S = S[rng.choice(len(S), len(R), replace=False)]
    X = np.vstack([R, S])
    y = np.r_[np.ones(len(R)), np.zeros(len(S))]
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p = m.decision_function(sc.transform(X[te]))
        aucs.append(auc(p[y[te] == 1], p[y[te] == 0]))
    a = float(np.mean(aucs))
    return max(a, 1 - a)


def tstr(Xs, ys, Xtr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(sc.transform(Xs), ys)
    pr = m.predict(sc.transform(Xte))
    return accuracy_score(yte, pr), f1_score(yte, pr, average="macro")


def psd_pinv(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, 0)
    inv = np.where(w > w.max() * 1e-10, 1 / np.where(w > 0, w, 1), 0)
    return (V * inv) @ V.T


def gauss_kde_logpdf(Train, Q, bw=None):
    """log density of Q under an isotropic Gaussian KDE on Train (Scott bandwidth per dim, after whitening)."""
    n, d = Train.shape
    mu, sd = Train.mean(0), Train.std(0) + 1e-9
    T, Qz = (Train - mu) / sd, (Q - mu) / sd
    h = bw if bw is not None else n ** (-1.0 / (d + 4))
    D = sqdist(Qz, T) / (2 * h * h)
    m = D.min(1, keepdims=True)
    return -m[:, 0] + np.log(np.exp(-(D - m)).sum(1)) - np.log(sd).sum()


def marg_logpdf(Train, Q, lo, hi, bins=30):
    """sum over genes of log histogram-density (with add-half smoothing) of Q under Train marginals,
    on a fixed per-gene grid [lo, hi]."""
    out = np.zeros(len(Q))
    w = (hi - lo) / bins
    bt = np.clip(((Train - lo) / w).astype(int), 0, bins - 1)
    bq = np.clip(((Q - lo) / w).astype(int), 0, bins - 1)
    for g in range(Train.shape[1]):
        h = np.bincount(bt[:, g], minlength=bins) + 0.5
        h = h / h.sum()
        out += np.log(h[bq[:, g]])
    return out


# ----------------------------------------------------------------------------- per-fold evaluation
def evaluate_fold(S, ys, Xtr, ytr, Xte, yte, ctx, rng, fast=False):
    r = {}
    classes = np.unique(np.r_[ytr, yte])
    # ---------------- fidelity: marginals
    w = w1_per_gene(S, Xte)
    r["w1"] = w.mean()
    w1c, wt = 0.0, 0
    for c in classes:
        a, b = S[ys == c], Xte[yte == c]
        if len(a) >= 5 and len(b) >= 5:
            w1c += w1_per_gene(a, b).mean() * len(b)
            wt += len(b)
    r["w1_cls"] = w1c / max(wt, 1)
    r["mean_err"] = np.abs(S.mean(0) - Xte.mean(0)).mean()
    r["sd_err"] = np.abs(S.std(0) - Xte.std(0)).mean()
    # ---------------- fidelity: joint
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Sz, Tez = (S - mu) / sd, (Xte - mu) / sd
    r["mmd"] = mmd_rbf(Sz, Tez)
    r["disc_auc"] = disc_auc(Xte, S, seed=0)
    P = ctx["pca50"]
    Sp, Tep = (S - mu) @ P, (Xte - mu) @ P
    r["prec"], r["rec"], r["dens"], r["cov"] = prdc(Tep, Sp[rng.choice(len(Sp), min(len(Sp), 2 * len(Tep)), replace=False)])
    iu = ctx["iu"]
    Rs = np.corrcoef(S, rowvar=False)[iu]
    Rs = np.nan_to_num(Rs)
    Rt = ctx["R_te"]
    r["corr_mae"] = np.abs(Rs - Rt).mean()
    r["corr_r"] = np.corrcoef(Rs, Rt)[0, 1]
    for thr in (0.3, 0.5):
        real, syn = np.abs(Rt) > thr, np.abs(Rs) > thr
        r[f"coex{thr}_tpr"] = (real & syn).sum() / max(real.sum(), 1)
        r[f"coex{thr}_fer"] = (syn & ~real).sum() / max(real.sum(), 1)
    # within-class co-expression (the structure P-PGM loses; between-class means alone create pooled edges)
    rw = []
    for c in classes:
        a, b = S[ys == c], Xte[yte == c]
        if len(a) >= 20 and len(b) >= 20:
            ra = np.nan_to_num(np.corrcoef(a, rowvar=False)[iu])
            rb = np.corrcoef(b, rowvar=False)[iu]
            rw.append((np.corrcoef(ra, rb)[0, 1], len(b)))
    r["wcorr_r"] = sum(a * n for a, n in rw) / max(sum(n for _, n in rw), 1)
    # ---------------- DE preservation (reference: real train, as in the CAMDA bio metric)
    de_real, eff_real = ctx["de_real"], ctx["eff_real"]
    de_syn, eff_syn = de_sets(S, ys, list(de_real))
    tp = fp = pos = neg = 0
    rhos = []
    for c in de_real:
        if c not in de_syn:
            continue
        tp += (de_real[c] & de_syn[c]).sum()
        pos += de_real[c].sum()
        fp += (~de_real[c] & de_syn[c]).sum()
        neg += (~de_real[c]).sum()
        rhos.append(stats.spearmanr(eff_real[c], eff_syn[c])[0])
    r["de_tpr"] = tp / max(pos, 1)
    r["de_fpr"] = fp / max(neg, 1)
    r["de_rho"] = float(np.mean(rhos)) if rhos else np.nan
    # ---------------- utility
    r["tstr_acc"], r["tstr_f1"] = tstr(S, ys, Xtr, Xte, yte)
    r["trtr_acc"], r["trtr_f1"] = ctx["trtr"]
    # ---------------- privacy
    nte = len(Xte)
    perm = rng.permutation(nte)
    ref_i, out_i = perm[: nte // 2], perm[nte // 2:]
    Ref, Out = Xte[ref_i], Xte[out_i]
    In = Xtr[rng.choice(len(Xtr), min(len(Xtr), 4 * len(Out)), replace=False)]
    # Mahalanobis (whole test set vs whole train set, as in the red-team script)
    muS = S.mean(0)
    Ci = psd_pinv(np.cov(S, rowvar=False))
    def mahal(T):
        D = T - muS
        return -np.sqrt(np.abs(np.einsum("ij,jk,ik->i", D, Ci, D)))
    r["mahala_auc"] = auc(mahal(Xtr), mahal(Xte))
    d_in, d_out = nn_dist(Xtr, S), nn_dist(Xte, S)
    r["dcr_auc"] = auc(-d_in, -d_out)
    r["dcr_tpr@0.1"] = tpr_at(-d_in, -d_out, 0.1)
    din2, dout2 = nn_dist(In, S), nn_dist(Out, S)
    r["dcrr_auc"] = auc(-din2 / nn_dist(In, Ref), -dout2 / nn_dist(Out, Ref))
    # DOMIAS-style density ratio in PCA-20 space fitted on the reference half
    P20 = P[:, :20]
    Sp20, Rp20 = (S - mu) @ P20, (Ref - mu) @ P20
    def dom(T):
        Tp = (T - mu) @ P20
        return gauss_kde_logpdf(Sp20, Tp) - gauss_kde_logpdf(Rp20, Tp)
    s_in, s_out = dom(In), dom(Out)
    r["domias_auc"] = auc(s_in, s_out)
    r["domias_tpr@0.1"] = tpr_at(s_in, s_out, 0.1)
    lo, hi = ctx["lo"], ctx["hi"]
    def marg(T):
        return marg_logpdf(S, T, lo, hi) - marg_logpdf(Ref, T, lo, hi)
    s_in, s_out = marg(In), marg(Out)
    r["marg_auc"] = auc(s_in, s_out)
    # DCR share with equal-size train subset
    Tr_sub = Xtr[rng.choice(len(Xtr), nte, replace=False)]
    r["dcr_share"] = float((nn_dist(S, Tr_sub) < nn_dist(S, Xte)).mean())
    r["dcr_syn_train"] = float(nn_dist(S, Xtr).mean())
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", required=True)
    p.add_argument("--annot", required=True)
    p.add_argument("--sample_col", default="samplesID")
    p.add_argument("--label_col", default="Subtype")
    p.add_argument("--syn_dir", nargs="+", required=True)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--oracle", action="store_true", help="add REAL_TRAIN pseudo-run")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    expr = load_expression(a.counts)
    ann = pd.read_csv(a.annot)
    labels = ann.set_index(ann[a.sample_col].astype(str))[a.label_col].astype(str)
    genes = list(expr.columns)
    ctx_cache = {}
    rows = []

    def get_ctx(ids_file):
        if ids_file in ctx_cache:
            return ctx_cache[ids_file]
        ids = json.load(open(ids_file))
        Xtr = expr.loc[ids["train"], genes].values
        Xte = expr.loc[ids["test"], genes].values
        ytr = labels.loc[ids["train"]].values
        yte = labels.loc[ids["test"]].values
        iu = np.triu_indices(Xtr.shape[1], 1)
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-9
        _, _, Vt = np.linalg.svd((Xtr - mu), full_matrices=False)
        de_real, eff_real = de_sets(Xtr, ytr, list(np.unique(ytr)))
        ctx = dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, iu=iu, pca50=Vt[:50].T,
                   R_te=np.corrcoef(Xte, rowvar=False)[iu], de_real=de_real, eff_real=eff_real,
                   trtr=tstr(Xtr, ytr, Xtr, Xte, yte),
                   lo=np.minimum(Xtr.min(0), Xte.min(0)) - 0.5, hi=np.maximum(Xtr.max(0), Xte.max(0)) + 0.5)
        ctx_cache[ids_file] = ctx
        return ctx

    jobs = []
    for sd in a.syn_dir:
        for fx in sorted(glob.glob(os.path.join(sd, "synthetic_data_split_*.csv"))):
            f = int(re.findall(r"split_(\d+)", fx)[-1])
            if a.folds and f not in a.folds:
                continue
            jobs.append((os.path.basename(os.path.normpath(sd)), f, fx, os.path.join(sd, f"split_{f}_ids.json")))
    if a.oracle:
        seen = {}
        for _, f, _, idf in jobs:
            seen.setdefault(f, idf)
        jobs = [("REAL_TRAIN", f, None, idf) for f, idf in sorted(seen.items())] + jobs

    for run, f, fx, idf in jobs:
        t0 = time.time()
        ctx = get_ctx(idf)
        if fx is None:
            S, ys = ctx["Xtr"], ctx["ytr"]
        else:
            Sdf = pd.read_csv(fx)
            S = Sdf[genes].values.astype(float)
            ys = pd.read_csv(fx.replace("_data_", "_labels_")).iloc[:, 0].astype(str).values
        rng = np.random.default_rng(a.seed + f)
        r = {"run": run, "fold": f}
        r.update(evaluate_fold(S, ys, ctx["Xtr"], ctx["ytr"], ctx["Xte"], ctx["yte"], ctx, rng))
        rows.append(r)
        print(f"{run:34s} f{f} | W1 {r['w1']:.3f} W1c {r['w1_cls']:.3f} disc {r['disc_auc']:.3f} "
              f"wcorr {r['wcorr_r']:.2f} coex.3 {r['coex0.3_tpr']:.2f}/{r['coex0.3_fer']:.2f} "
              f"DE {r['de_tpr']:.2f}/{r['de_fpr']:.2f} TSTR {r['tstr_acc']:.3f} | MIA mah {r['mahala_auc']:.2f} "
              f"dcr {r['dcr_auc']:.2f} dcrr {r['dcrr_auc']:.2f} dom {r['domias_auc']:.2f} "
              f"marg {r['marg_auc']:.2f} | {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 80)
    print("\n=== mean over folds ===")
    print(df.drop(columns="fold").groupby("run", sort=False).mean().round(3).T)
    if a.out_csv:
        df.to_csv(a.out_csv, index=False)
        print(f"saved {a.out_csv}")


if __name__ == "__main__":
    main()
