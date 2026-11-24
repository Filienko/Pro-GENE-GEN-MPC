#!/usr/bin/env python
"""Per-dataset ranking of the generators on every metric (5 folds, mean), privacy included.

BRCA privacy = strongest of MeLoMIA-ND / MAMA-MIA v2 / MahalaMIA / kNN-LR / fused per fold (BRCA_compare.csv);
BIO and ADULT privacy = strongest of the attacks available there (kNN-LR, MAMA-lite, MahalaMIA, DCR, DCR-ratio,
DOMIAS, marginal for BIO; DCR, DCR-ratio, Mahalanobis, DOMIAS, kNN-LR, MAMA-lite for ADULT).
Rank 1 = best.  Discriminator AUC is ranked by |AUC - 0.5|.

python rank_table.py   ->  $OUT/RANKS.md
"""
import os, runpy, sys, io, contextlib
import numpy as np, pandas as pd

O = os.environ["OUT"]; HERE = os.path.dirname(os.path.abspath(__file__))
LOW, HIGH, HALF = "low", "high", "half"
METHODS = [("mvn", "MVN"), ("tabpfn_35", "TabPFN-3.5"), ("tabpfn_35_dpcov65", "Hybrid dpcov 65%"),
           ("lrdp_eps10000", "DP-MVN ε10000"), ("lrdp_eps1000", "DP-MVN ε1000"), ("lrdp_eps100", "DP-MVN ε100"), ("lrdp_eps30", "DP-MVN ε30"),
           ("lrdp_eps10", "DP-MVN ε10"), ("lrdpc_eps1000", "DP-MVN-C ε1000"), ("lrdpc_eps100", "DP-MVN-C ε100"),
           ("lrdpc_eps30", "DP-MVN-C ε30"), ("lrdpc_eps10", "DP-MVN-C ε10"), ("pgm_eps10", "P-PGM ε10")]
GROUPS = {"privacy": ["Strongest attack"],
          "fidelity": ["W1", "Discrim. AUC", "Precision", "Recall", "Within-class corr", "Coexpr. TPR", "Coexpr. false edges",
                       "KL(real‖syn)", "Inv. KL", "W1 numeric", "TVD categorical"],
          "utility": ["DE TPR", "DE FPR", "TSTR acc", "TSTR F1", "TSTR AUC"]}
DIR = {"Strongest attack": LOW, "W1": LOW, "Discrim. AUC": HALF, "Precision": HIGH, "Recall": HIGH, "Within-class corr": HIGH,
       "Coexpr. TPR": HIGH, "Coexpr. false edges": LOW, "KL(real‖syn)": LOW, "Inv. KL": LOW, "W1 numeric": LOW,
       "TVD categorical": LOW, "DE TPR": HIGH, "DE FPR": LOW, "TSTR acc": HIGH, "TSTR F1": HIGH, "TSTR AUC": HIGH}
EVCOL = {"w1": "W1", "disc_auc": "Discrim. AUC", "prec": "Precision", "rec": "Recall", "wcorr_r": "Within-class corr",
         "coex0.3_tpr": "Coexpr. TPR", "coex0.3_fer": "Coexpr. false edges", "kl_real||syn_test": "KL(real‖syn)",
         "kl_syn||real_test": "Inv. KL", "de_tpr": "DE TPR", "de_fpr": "DE FPR", "tstr_acc": "TSTR acc", "tstr_f1": "TSTR F1"}


def quiet(path, argv):
    sys.argv = [path] + argv
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(path)


def brca():
    ev = pd.read_csv(f"{O}/TABPFN/BRCA_all_folds.csv")
    ev["run"] = ev.run.replace({"tabpfn_35_ma5dpcovlrdp470": "tabpfn_35_dpcov65"})
    D = ev.groupby("run")[list(EVCOL)].mean().rename(columns=EVCOL)
    C = pd.read_csv(f"{O}/TABPFN/BRCA_compare.csv").set_index("Method")
    lab = {"MVN (CAMDA baseline, noise 0.7)": "mvn", "TabPFN-3.5": "tabpfn_35", "TabPFN hybrid prompt, dpcov 65%": "tabpfn_35_dpcov65",
           "DP-MVN ε=10000": "lrdp_eps10000", "DP-MVN ε=1000": "lrdp_eps1000", "DP-MVN ε=100": "lrdp_eps100", "DP-MVN ε=30": "lrdp_eps30", "DP-MVN ε=10": "lrdp_eps10",
           "Private-PGM ε=10": "pgm_eps10", "DP-MVN-copula ε=1000": "lrdpc_eps1000", "DP-MVN-copula ε=100": "lrdpc_eps100",
           "DP-MVN-copula ε=30": "lrdpc_eps30", "DP-MVN-copula ε=10": "lrdpc_eps10"}
    for m, r in lab.items():
        if m in C.index:
            D.loc[r, "Strongest attack"] = C.loc[m, "Strongest attack"]; D.loc[r, "_worst"] = C.loc[m, "Strongest attack worst"]
    return D


def bio():
    g = quiet(os.path.join(HERE, "bio_table.py"), ["BIO"]); b = g["bio"]
    D = b.groupby("run")[[c for c in EVCOL if c in b] + ["strongest"]].mean().rename(columns={**EVCOL, "strongest": "Strongest attack"})
    D["_worst"] = b.groupby("run").strongest.max()
    return D.drop(columns=["Inv. KL"], errors="ignore")          # degenerate on zero-inflated data (about 13-14 for every method)


def adult():
    g = quiet(os.path.join(HERE, "adult_table.py"), []); e = g["ev"]
    cols = {"strongest": "Strongest attack", "w1_num": "W1 numeric", "tvd_cat": "TVD categorical", "tstr_acc": "TSTR acc",
            "tstr_f1": "TSTR F1", "tstr_auc": "TSTR AUC"}
    D = e.groupby("run")[list(cols)].mean().rename(columns=cols); D["_worst"] = e.groupby("run").strongest.max()
    return D


def rank(D):
    D = D.loc[[r for r, _ in METHODS if r in D.index]].copy()
    R = pd.DataFrame(index=D.index)
    for c in D.columns:
        if c.startswith("_") or D[c].isna().all():
            continue
        v = D[c]; key = {LOW: v, HIGH: -v, HALF: (v - 0.5).abs()}[DIR[c]]
        R[c] = key.rank(method="min")
    for gname, cols in GROUPS.items():
        cs = [c for c in cols if c in R]
        if cs:
            R[f"avg {gname}"] = R[cs].mean(1)
    R["avg all"] = R[[c for c in R if not c.startswith("avg")]].mean(1)
    return D, R


out = ["# Generator rankings per dataset (5 folds, mean; rank 1 = best)", "",
       "Privacy = strongest attack per fold, mean (worst fold). BRCA uses the full attack suite (MeLoMIA-ND, MAMA-MIA v2, "
       "MahalaMIA, kNN-LR, fused); BIO and ADULT use the attacks available there (weaker).  Average ranks: lower is better.", ""]
for name, fn in [("BRCA", brca), ("Bioresponse (BIO)", bio), ("Adult", adult)]:
    D, R = rank(fn())
    lab = dict(METHODS)
    metric_cols = [c for c in R.columns if not c.startswith("avg")]
    out += [f"## {name}", "", "| Method | " + " | ".join(metric_cols) + " | avg privacy | avg fidelity | avg utility | **avg all** |",
            "|---|" + "---|" * (len(metric_cols) + 4)]
    for r in R.sort_values("avg all").index:
        cells = []
        for c in metric_cols:
            v = D.loc[r, c]
            s = f"{v:.3f}" + (f" ({D.loc[r, '_worst']:.3f})" if c == "Strongest attack" and "_worst" in D else "")
            cells.append("–" if pd.isna(v) else f"{s} [{int(R.loc[r, c])}]")
        av = [f"{R.loc[r, k]:.1f}" if k in R else "–" for k in ("avg privacy", "avg fidelity", "avg utility")]
        out.append(f"| {lab[r]} | " + " | ".join(cells) + f" | {' | '.join(av)} | **{R.loc[r, 'avg all']:.1f}** |")
    out.append("")
open(f"{O}/RANKS.md", "w").write("\n".join(out))
# per-metric orderings (no averages): row = metric, columns = rank 1..8
SHORT = {"mvn": "MVN", "tabpfn_35": "TabPFN", "tabpfn_35_dpcov65": "Hybrid", "lrdp_eps10000": "DP-MVN ε10000", "lrdp_eps1000": "DP-MVN ε1000", "lrdp_eps100": "DP-MVN ε100",
         "lrdp_eps30": "DP-MVN ε30", "lrdp_eps10": "DP-MVN ε10", "pgm_eps10": "P-PGM ε10",
         "lrdpc_eps1000": "DP-MVN-C ε1000", "lrdpc_eps100": "DP-MVN-C ε100", "lrdpc_eps30": "DP-MVN-C ε30", "lrdpc_eps10": "DP-MVN-C ε10"}
NOTE = {LOW: "↓", HIGH: "↑", HALF: "→0.5"}
po = ["# Per-metric rankings (5 folds, mean; 1 = best)", ""]
for name, fn in [("BRCA", brca), ("Bioresponse", bio), ("Adult", adult)]:
    D, R = rank(fn())
    po += [f"## {name}", "", "| Metric | " + " | ".join(str(i) for i in range(1, len(D) + 1)) + " |", "|---|" + "---|" * len(D)]
    order = ["Strongest attack"] + [c for g in ("fidelity", "utility") for c in GROUPS[g]]
    for c in [c for c in order if c in R.columns]:
        x = R[c].dropna().sort_values(kind="stable")
        cells = [f"{SHORT[r]} {D.loc[r, c]:.3f}" for r in x.index]
        po.append(f"| {c} {NOTE[DIR[c]]} | " + " | ".join(cells + [""] * (len(D) - len(cells))) + " |")
    po.append("")
open(f"{O}/RANKS_per_metric.md", "w").write("\n".join(po)); print("\n".join(po))
