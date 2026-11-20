#!/usr/bin/env python
"""Adult evaluation (no gene-specific metrics): fidelity = W1 (numeric, each scaled by its public range) and total
variation distance (categorical); utility = TSTR with a gradient-boosting classifier on income (accuracy, macro-F1,
ROC AUC; TRTR as the reference); privacy = DCR, DCR-ratio, Mahalanobis and DOMIAS-style attacks (eval_suite.py
helpers) on a one-hot / range-scaled encoding.  kNN-LR and MAMA-lite run separately (adult_eval.sh).

Every release is decoded the same way first: numeric values clipped to the public range and rounded (all Adult
numeric columns are integers), categorical codes rounded to the nearest valid level.

python adult_eval.py --syn_dir DIR [DIR ...] --folds 1 2 3 4 5 --out_csv x.csv
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels  # noqa: E402
from eval_suite import auc, nn_dist, psd_pinv, gauss_kde_logpdf  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", default=os.environ.get("ADULT_TSV")); p.add_argument("--annot", default=os.environ.get("ADULT_ANN"))
p.add_argument("--meta", default=os.environ.get("ADULT_META"))
p.add_argument("--syn_dir", nargs="+", required=True); p.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
p.add_argument("--seed", type=int, default=0); p.add_argument("--out_csv", required=True)
a = p.parse_args()
meta = json.load(open(a.meta))
X = load_expression(a.counts); lab = load_labels(a.annot, X.index, "samplesID", "Subtype"); X = X.loc[lab.index]
cols = list(X.columns); NUM = [c for c in cols if c in meta["numeric"]]; CAT = [c for c in cols if c in meta["categorical"]]
LO = np.array([meta["numeric"][c][0] for c in NUM]); HI = np.array([meta["numeric"][c][1] for c in NUM])
NL = {c: len(meta["categorical"][c]) for c in CAT}


def decode(D):
    D = D.copy()
    D[NUM] = np.rint(np.clip(D[NUM].values, LO, HI))
    for c in CAT:
        D[c] = np.clip(np.rint(D[c].values), 0, NL[c] - 1)
    return D


def encode(D):          # range-scaled numeric + one-hot categorical (scaled so a mismatch has distance 1, like a full range)
    parts = [(D[NUM].values - LO) / (HI - LO)]
    for c in CAT:
        parts.append(np.eye(NL[c])[D[c].values.astype(int)] / np.sqrt(2))
    return np.hstack(parts)


def clf_fit(D, y):
    m = HistGradientBoostingClassifier(categorical_features=[D.columns.get_loc(c) for c in CAT], random_state=0, max_iter=200)
    return m.fit(D.values, y)


rows = []
for sd in a.syn_dir:
    run = os.path.basename(os.path.normpath(sd))
    for f in a.folds:
        fx = os.path.join(sd, f"synthetic_data_split_{f}.csv")
        if not os.path.exists(fx):
            continue
        rng = np.random.default_rng(a.seed + f)
        ids = json.load(open(os.path.join(sd, f"split_{f}_ids.json")))
        Tr, Te = decode(X.loc[ids["train"], cols]), decode(X.loc[ids["test"], cols])
        ytr, yte = (lab.loc[ids["train"]] == "gt50K").astype(int).values, (lab.loc[ids["test"]] == "gt50K").astype(int).values
        S = decode(pd.read_csv(fx)[cols])
        ys = (pd.read_csv(os.path.join(sd, f"synthetic_labels_split_{f}.csv")).iloc[:, 0].astype(str) == "gt50K").astype(int).values
        r = {"run": run, "fold": f}
        # fidelity
        r["w1_num"] = float(np.mean([wasserstein_distance((S[c] - lo) / (hi - lo), (Te[c] - lo) / (hi - lo)) for c, lo, hi in zip(NUM, LO, HI)]))
        tv = []
        for c in CAT:
            ps = np.bincount(S[c].astype(int), minlength=NL[c]) / len(S); pt = np.bincount(Te[c].astype(int), minlength=NL[c]) / len(Te)
            tv.append(0.5 * np.abs(ps - pt).sum())
        r["tvd_cat"] = float(np.mean(tv))
        r["label_rate_err"] = float(abs(ys.mean() - ytr.mean()))
        # utility (TSTR / TRTR) on income
        for tag, (D, yy) in {"tstr": (S, ys), "trtr": (Tr, ytr)}.items():
            if len(np.unique(yy)) < 2:
                r[f"{tag}_acc"] = r[f"{tag}_f1"] = r[f"{tag}_auc"] = np.nan; continue
            m = clf_fit(D, yy); pr = m.predict_proba(Te.values)[:, 1]
            r[f"{tag}_acc"] = accuracy_score(yte, pr > 0.5); r[f"{tag}_f1"] = f1_score(yte, pr > 0.5, average="macro"); r[f"{tag}_auc"] = roc_auc_score(yte, pr)
        # privacy
        Es, Etr, Ete = encode(S), encode(Tr), encode(Te)
        nte = len(Ete); perm = rng.permutation(nte); Ref, Out = Ete[perm[: nte // 2]], Ete[perm[nte // 2:]]
        In = Etr[rng.choice(len(Etr), min(len(Etr), 4 * len(Out)), replace=False)]
        r["dcr_auc"] = auc(-nn_dist(Etr, Es), -nn_dist(Ete, Es))
        r["dcrr_auc"] = auc(-nn_dist(In, Es) / nn_dist(In, Ref), -nn_dist(Out, Es) / nn_dist(Out, Ref))
        mu = Es.mean(0); Ci = psd_pinv(np.cov(Es, rowvar=False))
        mah = lambda T: -np.sqrt(np.abs(np.einsum("ij,jk,ik->i", T - mu, Ci, T - mu)))
        r["mahala_auc"] = auc(mah(Etr), mah(Ete))
        U = np.linalg.svd(Etr - Etr.mean(0), full_matrices=False)[2][:10].T
        dom = lambda T: gauss_kde_logpdf((Es - Etr.mean(0)) @ U, (T - Etr.mean(0)) @ U) - gauss_kde_logpdf((Ref - Etr.mean(0)) @ U, (T - Etr.mean(0)) @ U)
        r["domias_auc"] = auc(dom(In), dom(Out))
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(a.out_csv, index=False)
