#!/usr/bin/env python
"""Model-aware membership attack on the TabPFN gene-by-gene generator (black box: release + candidate pool).

The generator (tabpfn_extensions TabPFNUnsupervisedModel, as in run_tabpfn.py) draws column j of every synthetic row
from TabPFN's predictive distribution given columns 0..j-1 (column 0 = subtype), with all members as the prompt.  Late
in the order a long prefix pins down an individual member, so the next value is pulled towards that member: the
release is locally a better model of the members than of the population.

Score of candidate x (a likelihood ratio with the generator's own model as density estimator):
    sum over target genes j of  log p_TabPFN(x_j | x_F(j); prompt = release)  -  log p_TabPFN(x_j | x_F(j); prompt = reference)
F(j) = subtype + a random subset of the genes generated before j (cheaper than all of them, and enough to pin a
member).  Reference prompt = the real candidate pool, cross-fitted in K folds so x is never in its own prompt (about
the size of the release).  No shadow models, no membership labels.

python tabpfn_lr_attack.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --syn_dir DIR --folds 3 --out_csv x.csv
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--syn_dir", required=True); p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--n_targets", type=int, default=60); p.add_argument("--n_feat", type=int, default=50)
p.add_argument("--min_pos", type=int, default=200, help="targets drawn from generation positions >= this")
p.add_argument("--kfold", type=int, default=5); p.add_argument("--n_estimators", type=int, default=4)
p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_csv", required=True)
a = p.parse_args()
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
import torch  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

expr = load_expression(a.counts); lab = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[lab.index]; genes = list(expr.columns); classes = sorted(lab.unique())


def nll(Xfit, yfit, Xq, yq, seed):
    reg = TabPFNRegressor(n_estimators=a.n_estimators, device=a.device, random_state=seed,
                          ignore_pretraining_limits=True, categorical_features_indices=[0])
    reg.fit(Xfit.astype(np.float32), yfit.astype(np.float32))
    pr = reg.predict(Xq.astype(np.float32), output_type="full")
    with torch.no_grad():
        return pr["criterion"](torch.as_tensor(pr["logits"]).float().cpu(), torch.as_tensor(yq).float()).numpy()


rows = []
for f in a.folds:
    t0 = time.time(); rng = np.random.default_rng(a.seed + f)
    ids = json.load(open(os.path.join(a.syn_dir, f"split_{f}_ids.json")))
    cand = ids["train"] + ids["test"]; mem = np.r_[np.ones(len(ids["train"])), np.zeros(len(ids["test"]))]
    C = np.column_stack([[classes.index(c) for c in lab.loc[cand]], expr.loc[cand, genes].values]).astype(np.float64)
    S = np.column_stack([[classes.index(c) for c in pd.read_csv(os.path.join(a.syn_dir, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]],
                         pd.read_csv(os.path.join(a.syn_dir, f"synthetic_data_split_{f}.csv"))[genes].values]).astype(np.float64)
    kf = rng.permutation(len(C)) % a.kfold
    targets = np.sort(rng.choice(np.arange(max(a.min_pos, 1), C.shape[1]), a.n_targets, replace=False))
    ls, lr = np.zeros((len(C), a.n_targets)), np.zeros((len(C), a.n_targets))
    for t, j in enumerate(targets):
        feats = np.r_[0, np.sort(rng.choice(np.arange(1, j), min(a.n_feat, j - 1), replace=False))]
        sd = int(rng.integers(1 << 30))
        ls[:, t] = nll(S[:, feats], S[:, j], C[:, feats], C[:, j], sd)              # prompt = release
        for k in range(a.kfold):                                                     # prompt = other candidates
            q = kf == k
            lr[q, t] = nll(C[~q][:, feats], C[~q, j], C[q][:, feats], C[q, j], sd)
        if t % 10 == 9:
            sc = (lr[:, :t + 1] - ls[:, :t + 1]).sum(1)
            print(f"  fold {f} target {t + 1}/{a.n_targets}: AUC so far {roc_auc_score(mem, sc):.3f}  {time.time() - t0:.0f}s", flush=True)
    for name, sc in {"lr": (lr - ls).sum(1), "release_only": -ls.sum(1), "lr_clip": np.clip(lr - ls, -5, 5).sum(1)}.items():
        fpr, tpr, _ = roc_curve(mem, sc)
        rows.append({"run": os.path.basename(os.path.normpath(a.syn_dir)), "fold": f, "score": name,
                     "auc": roc_auc_score(mem, sc), "tpr@0.01": float(np.interp(0.01, fpr, tpr)),
                     "tpr@0.1": float(np.interp(0.1, fpr, tpr)), "n_targets": a.n_targets, "n_feat": a.n_feat})
        print(f"[TabPFN-LR] fold {f} {name}: AUC {rows[-1]['auc']:.3f}  TPR@1% {rows[-1]['tpr@0.01']:.3f}", flush=True)
    np.savez_compressed(a.out_csv.replace(".csv", f"_f{f}.npz"), ls=ls, lr=lr, mem=mem, targets=targets)
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
