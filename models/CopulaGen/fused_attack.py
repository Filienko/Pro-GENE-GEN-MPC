#!/usr/bin/env python
"""Fused membership attack: rank average (no labels used) of
   MeLoMIA-ND and MAMA-MIA v2 (per-record scores of MIA_on_bulkRNAseq_CAMDA2026 runs, see mia_verify/) and
   kNN-LR (knn_lr_attack.py: the generator's prefix -> next-gene likelihood ratio, k=40, 150 conditioning genes).
Settings of kNN-LR were fixed on TabPFN-3.5 fold 3 only.

python fused_attack.py --runs tabpfn_35 tabpfn_35_ma5 ... --out_csv fused.csv
"""
import argparse, glob, os, subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True); p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--run_root", default=os.path.join(os.environ["OUT"], "TABPFN", "BRCA"))
p.add_argument("--extra_roots", nargs="*", default=[os.path.join(os.environ["OUT"], "LRDP", "v2", "BRCA"),
                                                    os.path.join(os.environ["OUT"], "EPSWEEP", "BRCA")])
p.add_argument("--mia_runs", default=os.path.expanduser("~/privacy/mia_verify/results/runs"))
p.add_argument("--work", default=os.path.join(os.environ["OUT"], "TABPFN", "attack_lr"))
p.add_argument("--out_csv", required=True)
a = p.parse_args()


def tpr(y, s, f):
    fp, tp, _ = roc_curve(y, s); return float(np.interp(f, fp, tp))


rows = []
for run in a.runs:
    d_run = next((os.path.join(r, run) for r in [a.run_root, *a.extra_roots] if os.path.isdir(os.path.join(r, run))), None)
    for f in a.folds:
        if d_run is None or not os.path.exists(os.path.join(d_run, f"synthetic_data_split_{f}.csv")):
            continue
        kf = os.path.join(a.work, f"knn_{run}_{run}_f{f}_scores.csv")
        if not os.path.exists(kf):
            subprocess.run([sys.executable, os.path.join(HERE, "knn_lr_attack.py"), "--counts", os.environ["BRCA_TSV"],
                            "--annot", os.environ["BRCA_ANN"], "--label_col", "Subtype", "--syn_dir", d_run, "--folds", str(f),
                            "--k", "40", "--n_feat", "150", "--same_class", "0",
                            "--out_csv", os.path.join(a.work, f"knn_{run}.csv")], check=True, stdout=subprocess.DEVNULL)
        d = pd.read_csv(kf)[["sample_id", "knn_lr_clip", "member"]]
        have = {"kNN-LR": "knn_lr_clip"}
        for att, nm in [("melomia_nd", "MeLoMIA-ND"), ("mamamia_v2", "MAMA-MIA v2")]:
            g = glob.glob(os.path.join(a.mia_runs, f"BRCA__{att}__ours@run={run}__s{f}__*", "scores.csv"))
            if g:
                d = d.merge(pd.read_csv(g[0])[["sample_id", "score"]].rename(columns={"score": att}), on="sample_id"); have[nm] = att
        rk = {nm: rankdata(d[c]) / len(d) for nm, c in have.items()}
        scores = {nm: d[c] for nm, c in have.items()}
        if {"MeLoMIA-ND"} <= set(rk):
            scores["fused: MeLoMIA+kNN"] = rk["MeLoMIA-ND"] + rk["kNN-LR"]
        if {"MeLoMIA-ND", "MAMA-MIA v2"} <= set(rk):
            scores["fused: MeLoMIA+kNN+MAMA"] = rk["MeLoMIA-ND"] + rk["kNN-LR"] + rk["MAMA-MIA v2"]
        for nm, s in scores.items():
            rows.append({"run": run, "fold": f, "attack": nm, "auc": roc_auc_score(d.member, s),
                         "tpr@0.01": tpr(d.member, s, 0.01), "tpr@0.1": tpr(d.member, s, 0.1)})
R = pd.DataFrame(rows); R.to_csv(a.out_csv, index=False)
g = R.groupby(["run", "attack"]).agg(auc=("auc", "mean"), sd=("auc", "std"), tpr1=("tpr@0.01", "mean"), n=("fold", "nunique"))
print(g.round(3).to_string())
