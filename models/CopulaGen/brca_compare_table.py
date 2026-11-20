#!/usr/bin/env python
"""One BRCA comparison table (5 CAMDA folds, identical member/non-member splits for every method):
fidelity + utility from eval_suite / CAMDA KL (BRCA_all_folds.csv, written by tabpfn_report.py) and privacy from the
MIA_on_bulkRNAseq_CAMDA2026 attacks (MeLoMIA-ND, MAMA-MIA v2, MahalaMIA; mia_verify/) plus kNN-LR and the fused
attack (fused_attack.py).  Mean over folds, worst fold for the attacks.

python brca_compare_table.py      ->  $OUT/TABPFN/BRCA_compare.{md,csv}
"""
import glob, os, subprocess, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(os.environ["OUT"], "TABPFN"); MIA = os.path.expanduser("~/privacy/mia_verify/results/runs")
# (run name in BRCA_all_folds.csv, run name of the release dir / attacks, label, formal DP)
M = [("mvn", "mvn", "MVN (CAMDA baseline, noise 0.7)", "no"),
     ("tabpfn_35", "tabpfn_35", "TabPFN-3.5", "no"),
     ("tabpfn_35_ma5covlrdp", "tabpfn_35_ma5covlrdp", "TabPFN hybrid prompt, cov 50%", "no"),
     ("tabpfn_35_ma5dpcovlrdp", "tabpfn_35_ma5dpcovlrdp", "TabPFN hybrid prompt, dpcov 50%", "no"),
     ("tabpfn_35_ma5dpcovlrdp470", "tabpfn_35_ma5dpcovlrdp470", "TabPFN hybrid prompt, dpcov 65%", "no"),
     ("lrdp_eps1000", "lrdp_eps1000", "DP-MVN ε=1000", "yes (ε=1000)"),
     ("lrdp_eps300", "lrdp_eps300", "DP-MVN ε=300", "yes (ε=300)"),
     ("lrdp_eps200", "lrdp_eps200", "DP-MVN ε=200", "yes (ε=200)"),
     ("lrdp_eps100", "lrdp_eps100", "DP-MVN ε=100", "yes (ε=100)"),
     ("lrdp_eps70", "lrdp_eps70", "DP-MVN ε=70", "yes (ε=70)"),
     ("lrdp_eps50", "lrdp_eps50", "DP-MVN ε=50", "yes (ε=50)"),
     ("lrdp_eps30", "v2p_priv_eps30", "DP-MVN ε=30", "yes"),
     ("lrdpc_eps1000", "lrdpc_eps1000", "DP-MVN-copula ε=1000", "yes (ε=1000)"),
     ("lrdpc_eps100", "lrdpc_eps100", "DP-MVN-copula ε=100", "yes (ε=100)"),
     ("lrdpc_eps30", "lrdpc_eps30", "DP-MVN-copula ε=30", "yes"),
     ("lrdpc_eps10", "lrdpc_eps10", "DP-MVN-copula ε=10", "yes"),
     ("tabpfn_35_dp30", "tabpfn_35_dp30", "TabPFN-3.5 prompted with DP-MVN ε=30", "yes (ε=30)"),
     ("lrdp_eps10", "lrdp_eps10", "DP-MVN ε=10", "yes"),
     ("lrdp_eps1", "lrdp_eps1", "DP-MVN ε=1", "yes"),
     ("lrdp_eps0.5", "lrdp_eps0.5", "DP-MVN ε=0.5", "yes"),
     ("pgm_eps10", "pgm_eps10", "Private-PGM ε=10", "yes")]

ev = pd.read_csv(os.path.join(T, "BRCA_all_folds.csv"))
# fused attack (MeLoMIA-ND, MAMA-MIA v2, kNN-LR, fused) for all methods; kNN scores are cached
fz = os.path.join(T, "attack_lr", "fused_compare.csv")
subprocess.run([sys.executable, os.path.join(HERE, "fused_attack.py"), "--runs", *[r for _, r, _, _ in M], "--out_csv", fz],
               check=True, stdout=subprocess.DEVNULL)
F = pd.read_csv(fz)
rows = []
for run in [r for _, r, _, _ in M]:                       # MahalaMIA straight from its per-record scores
    for f in range(1, 6):
        g = glob.glob(os.path.join(MIA, f"BRCA__mahalamia__ours@run={run}__s{f}__*", "scores.csv"))
        if g:
            s = pd.read_csv(g[0]); fp, tp, _ = roc_curve(s.y_member, s.score)
            rows.append({"run": run, "fold": f, "attack": "MahalaMIA", "auc": roc_auc_score(s.y_member, s.score),
                         "tpr@0.01": float(np.interp(0.01, fp, tp))})
F = pd.concat([F, pd.DataFrame(rows)])

ATT = [("MeLoMIA-ND", "MeLoMIA-ND"), ("MAMA-MIA v2", "MAMA-MIA v2"), ("MahalaMIA", "MahalaMIA"), ("kNN-LR", "kNN-LR"),
       ("fused: MeLoMIA+kNN+MAMA", "Fused (3 attacks)")]
FID = [("w1", "W1 ↓"), ("disc_auc", "Discrim. AUC →0.5"), ("prec", "Precision ↑"), ("rec", "Recall ↑"),
       ("corr_mae", "Corr MAE ↓"), ("wcorr_r", "Within-subtype corr r ↑"), ("coex0.3_tpr", "Coexpr. TPR ↑"),
       ("coex0.3_fer", "Coexpr. false-edge ↓"), ("kl_real||syn_test", "KL(real‖syn) ↓"), ("kl_syn||real_test", "Inv. KL(syn‖real) ↓")]
UTIL = [("de_tpr", "DE TPR ↑"), ("de_fpr", "DE FPR ↓"), ("tstr_acc", "TSTR acc ↑"), ("tstr_f1", "TSTR macro-F1 ↑")]

out, nf = [], {}
for ev_run, run, lab, dp in M:
    r = {"Method": lab, "Formal DP": dp}
    e = ev[ev.run == ev_run]; nf[lab] = e.fold.nunique()
    if not len(e) and not (F.run == run).any():
        continue
    for c, n in FID + UTIL:
        r[n] = e[c].mean() if len(e) else np.nan
    for a_, n in ATT:
        x = F[(F.run == run) & (F.attack == a_)]
        r[n] = x.auc.mean() if len(x) else np.nan; r[n + " worst"] = x.auc.max() if len(x) else np.nan
        if a_.startswith("fused"):
            r["Fused TPR@1%FPR"] = x["tpr@0.01"].mean() if len(x) else np.nan
        r.setdefault("_folds_att", set()).update(x.fold)
    r["Folds"] = f"{nf[lab]}/{len(r.pop('_folds_att'))}"
    best = F[(F.run == run)].groupby("fold").auc.max()          # strongest attack on each fold (any of the above)
    r["Strongest attack"] = best.mean() if len(best) else np.nan; r["Strongest attack worst"] = best.max() if len(best) else np.nan
    out.append(r)
D = pd.DataFrame(out).set_index("Method")
D.to_csv(os.path.join(T, "BRCA_compare.csv"))


def fmt(v, d=3):
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def md(cols, title):
    s = [f"### {title}", "", "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, r in D.iterrows():
        s.append(f"| {m} | " + " | ".join(r[c] if isinstance(r[c], str) else fmt(r[c]) for c in cols) + " |")
    return "\n".join(s) + "\n"


priv = []
for _, n in ATT + [(None, "Strongest attack")]:
    D[n + " (worst)"] = [f"{fmt(a)} ({fmt(b)})" for a, b in zip(D[n], D[n + " worst"])]
    priv.append(n + " (worst)")
txt = ["# BRCA: fidelity, utility and privacy, 5 CAMDA folds", "",
       "Every method is trained on the same 5 member splits (871/872 members, 218/217 non-members per fold). "
       "Mean over folds; attacks show mean AUC (worst fold). Folds column = eval folds / attack folds.", "",
       md(["Formal DP", "Folds"] + priv + ["Fused TPR@1%FPR"], "Privacy (membership-inference AUC, 0.5 = chance; bar: < 0.6)"),
       md([n for _, n in FID], "Fidelity"), md([n for _, n in UTIL], "Utility")]
open(os.path.join(T, "BRCA_compare.md"), "w").write("\n".join(txt))
print("\n".join(txt))
