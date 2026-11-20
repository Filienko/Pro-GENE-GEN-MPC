#!/usr/bin/env python
"""Fidelity / utility / privacy table for a second dataset (bio_eval.sh), plus the same methods on BRCA with the
same attack battery, so the two datasets are compared like for like.

Attack battery available on both datasets ("common"): kNN-LR, MAMA-lite (4 and 10 bins) and eval_suite's
MahalaMIA, DCR, DCR-ratio, DOMIAS and marginal attacks.  "Strongest" = the highest AUC of these on each fold.
(On BRCA the challenge repo's MeLoMIA-ND / MAMA-MIA v2 / fused attacks are stronger; see BRCA_compare.md.)

python bio_table.py BIO        ->  $OUT/BIO/BIO_compare.md
"""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DS = sys.argv[1] if len(sys.argv) > 1 else "BIO"
O = os.environ["OUT"]; B = os.path.join(O, DS)
M = [("mvn", "MVN (CAMDA baseline)", "no"), ("tabpfn_35", "TabPFN-3.5", "no"),
     ("tabpfn_35_dpcov65", "TabPFN hybrid dpcov 65%", "no"), ("dpcov_prompt_only", "dpcov cluster rows, no TabPFN", "no"),
     ("lrdp_eps1000", "DP-MVN ε=1000", "ε=1000"), ("lrdp_eps100", "DP-MVN ε=100", "ε=100"), ("lrdp_eps50", "DP-MVN ε=50", "ε=50"),
     ("lrdp_eps30", "DP-MVN ε=30", "ε=30"), ("lrdp_eps10", "DP-MVN ε=10", "ε=10"), ("pgm_eps10", "Private-PGM ε=10", "ε=10")]
ATT = ["mahala_auc", "dcr_auc", "dcrr_auc", "domias_auc", "marg_auc", "mama4", "mama10", "knn_lr"]


def collect(eval_glob, mama_glob, kl_glob, knn_glob, rename=None):
    ev = pd.concat([pd.read_csv(f) for f in glob.glob(eval_glob)]) if glob.glob(eval_glob) else pd.DataFrame(columns=["run", "fold"])
    ev = ev.drop(columns=[c for c in ev.columns if c.startswith("kl_") or c in ("mama4", "mama10", "overlap_prop")], errors="ignore")
    parts = []
    for f in glob.glob(mama_glob):
        m = pd.read_csv(f)
        if {"run", "fold", "bins", "auc"} <= set(m.columns):
            parts.append(m)
    if parts:
        m = pd.concat(parts).pivot_table(index=["run", "fold"], columns="bins", values="auc").rename(columns=lambda b: f"mama{b}").reset_index()
        ev = ev.merge(m, on=["run", "fold"], how="left")
    kl = [pd.read_csv(f) for f in glob.glob(kl_glob) if os.path.getsize(f) > 0]
    if kl:
        kl = pd.concat(kl); ev = ev.merge(kl[[c for c in kl.columns if c in ("run", "fold") or c.startswith("kl_")]], on=["run", "fold"], how="left")
    kn = []
    for f in glob.glob(knn_glob):
        s = pd.read_csv(f); base = os.path.basename(f)[4:-11]          # knn_<run>_<run>_f<k>_scores.csv
        run, fold = base.rsplit("_f", 1)[0], int(base.rsplit("_f", 1)[1])
        run = run[: len(run) // 2]
        kn.append({"run": run, "fold": fold, "knn_lr": roc_auc_score(s.member, s.knn_lr_clip)})
    if kn:
        ev = ev.merge(pd.DataFrame(kn), on=["run", "fold"], how="left")
    if rename:
        ev["run"] = ev.run.replace(rename)
    ev["strongest"] = ev[[c for c in ATT if c in ev.columns]].max(axis=1)
    return ev


def table(d):
    rows = []
    for run, lab, dp in M:
        x = d[d.run == run]
        if not len(x):
            continue
        g = lambda c: x[c].mean() if c in x and x[c].notna().any() else np.nan
        rows.append({"Method": lab, "Formal DP": dp, "Folds": x.fold.nunique(),
                     "Strongest attack": f"{x.strongest.mean():.3f} ({x.strongest.max():.3f})",
                     "kNN-LR": g("knn_lr"), "MAMA-lite": g("mama10"), "MahalaMIA": g("mahala_auc"), "DCR": g("dcr_auc"),
                     "W1 ↓": g("w1"), "Discrim. AUC": g("disc_auc"), "Precision": g("prec"), "Recall": g("rec"),
                     "Within-class corr": g("wcorr_r"), "Coexpr. TPR": g("coex0.3_tpr"), "Coexpr. false-edge": g("coex0.3_fer"),
                     "KL(real‖syn)": g("kl_real||syn_test"), "Inv. KL": g("kl_syn||real_test"),
                     "DE TPR": g("de_tpr"), "DE FPR": g("de_fpr"), "TSTR acc": g("tstr_acc"), "TSTR F1": g("tstr_f1")})
    return pd.DataFrame(rows).set_index("Method")


def md(D, title):
    cols = list(D.columns)
    s = [f"### {title}", "", "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, r in D.iterrows():
        s.append(f"| {m} | " + " | ".join(v if isinstance(v, str) else ("–" if pd.isna(v) else (str(v) if isinstance(v, (int, np.integer)) else f"{v:.3f}")) for v in r) + " |")
    return "\n".join(s) + "\n"


bio = collect(f"{B}/eval/*.csv", f"{B}/mama/*.csv", f"{B}/kl/*.csv", f"{B}/knn/knn_*_scores.csv")
# TabPFN on BIO: the runs before the generator fixes (unbounded draws, class indices / truncated classes written as
# values) are dropped; the *_fix runs (--clip 2 22 --no_cat_infer, see run_tabpfn.py) take their place
bio = bio[~bio.run.isin(["tabpfn_35", "tabpfn_35_dpcov65", "tabpfn_35_clip", "tabpfn_35_dpcov65_clip"])]
bio["run"] = bio.run.replace({"tabpfn_35_fix": "tabpfn_35", "tabpfn_35_dpcov65_fix": "tabpfn_35_dpcov65"})
T = os.path.join(O, "TABPFN")
brca = collect(f"{T}/BRCA/eval/*.csv", f"{T}/BRCA/mama/*.csv", f"{T}/BRCA/kl/*.csv", f"{T}/attack_lr/knn_*_scores.csv",
               rename={"v2p_priv_eps30": "lrdp_eps30", "tabpfn_35_ma5dpcovlrdp470": "tabpfn_35_dpcov65"})
base = pd.read_csv(f"{T}/BRCA_all_folds.csv")          # BRCA eval rows of the DP-MVN / P-PGM runs live in EPSWEEP / LRDP evals
base = base[base.run.isin([r for r, _, _ in M]) & ~base.run.isin(brca.run)]
if len(base):
    kn = brca[["run", "fold", "knn_lr"]] if "knn_lr" in brca else None
    brca = pd.concat([brca, base], ignore_index=True)
    brca["strongest"] = brca[[c for c in ATT if c in brca.columns]].max(axis=1)
Tb, Tr = table(bio), table(brca)
priv = ["Formal DP", "Folds", "Strongest attack", "kNN-LR", "MAMA-lite", "MahalaMIA", "DCR"]
fid = ["W1 ↓", "Discrim. AUC", "Precision", "Recall", "Within-class corr", "Coexpr. TPR", "Coexpr. false-edge", "KL(real‖syn)", "Inv. KL"]
uti = ["DE TPR", "DE FPR", "TSTR acc", "TSTR F1"]
txt = [f"# {DS} vs BRCA, same methods, same attack battery", "",
       "Strongest attack = max AUC per fold over kNN-LR, MAMA-lite (4/10 bins), MahalaMIA, DCR, DCR-ratio, DOMIAS, marginal; "
       "mean (worst fold).  This battery is weaker than BRCA's full one (MeLoMIA-ND, MAMA-MIA v2, fused; see BRCA_compare.md).", ""]
for name, D in [(DS, Tb), ("BRCA (same battery)", Tr)]:
    txt += [f"## {name}", "", md(D[priv], "Privacy"), md(D[fid], "Fidelity"), md(D[uti], "Utility")]
open(os.path.join(B, f"{DS}_compare.md"), "w").write("\n".join(txt))
print("\n".join(txt))
