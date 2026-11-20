#!/usr/bin/env python
"""Adult table: W1 / TVD, TSTR, attacks (DCR, DCR-ratio, Mahalanobis, DOMIAS, kNN-LR, MAMA-lite); strongest per fold.
python adult_table.py   ->  $OUT/ADULT/ADULT_compare.md"""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
AD = os.path.join(os.environ["OUT"], "ADULT")
M = [("tabpfn_35", "TabPFN-3.5", "no"), ("tabpfn_35_dpcov65", "TabPFN hybrid dpcov 65%", "no"), ("dpcov_prompt_only", "dpcov cluster rows, no TabPFN", "no"),
     ("mvn", "MVN (CAMDA baseline)", "no"), ("lrdp_eps1000", "DP-MVN ε=1000", "ε=1000"), ("lrdp_eps100", "DP-MVN ε=100", "ε=100"),
     ("lrdp_eps30", "DP-MVN ε=30", "ε=30"), ("lrdp_eps10", "DP-MVN ε=10", "ε=10"),
     ("lrdpc_eps1000", "DP-MVN-copula ε=1000", "ε=1000"), ("lrdpc_eps100", "DP-MVN-copula ε=100", "ε=100"),
     ("lrdpc_eps30", "DP-MVN-copula ε=30", "ε=30"), ("lrdpc_eps10", "DP-MVN-copula ε=10", "ε=10"), ("pgm_eps10", "Private-PGM ε=10", "ε=10")]
ev = pd.concat([pd.read_csv(f) for f in glob.glob(f"{AD}/eval/*.csv")])
mm = [pd.read_csv(f) for f in glob.glob(f"{AD}/mama/*.csv")]
if mm:
    m = pd.concat(mm).pivot_table(index=["run", "fold"], columns="bins", values="auc").rename(columns=lambda b: f"mama{b}").reset_index()
    ev = ev.merge(m, on=["run", "fold"], how="left")
kn = []
for f in glob.glob(f"{AD}/knn/knn_*_scores.csv"):
    base = os.path.basename(f)[4:-11]; run, fold = base.rsplit("_f", 1); run = run[: len(run) // 2]
    s = pd.read_csv(f); kn.append({"run": run, "fold": int(fold), "knn_lr": roc_auc_score(s.member, s.knn_lr_clip)})
if kn:
    ev = ev.merge(pd.DataFrame(kn), on=["run", "fold"], how="left")
att = [c for c in ["dcr_auc", "dcrr_auc", "mahala_auc", "domias_auc", "knn_lr", "mama4", "mama10"] if c in ev]
ev["strongest"] = ev[att].max(axis=1)
out = ["# Adult (1089-row subset, 5 splits 870-872 / 217-219), mean over folds (worst fold for the attack)", "",
       "| Method | Formal DP | Folds | Strongest attack | DCR | kNN-LR | MAMA-lite | W1 numeric ↓ | TVD categorical ↓ | TSTR acc ↑ | TSTR macro-F1 ↑ | TSTR AUC ↑ |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for run, lab, dp in M:
    x = ev[ev.run == run]
    if not len(x): continue
    g = lambda c: f"{x[c].mean():.3f}" if c in x and x[c].notna().any() else "–"
    out.append(f"| {lab} | {dp} | {x.fold.nunique()} | {x.strongest.mean():.3f} ({x.strongest.max():.3f}) | {g('dcr_auc')} | {g('knn_lr')} | {g('mama10')} | "
               f"{g('w1_num')} | {g('tvd_cat')} | {g('tstr_acc')} | {g('tstr_f1')} | {g('tstr_auc')} |")
out.append(f"| *real train (TRTR reference)* | – | – | – | – | – | – | – | – | {ev.trtr_acc.mean():.3f} | {ev.trtr_f1.mean():.3f} | {ev.trtr_auc.mean():.3f} |")
open(f"{AD}/ADULT_compare.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
