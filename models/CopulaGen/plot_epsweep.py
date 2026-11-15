#!/usr/bin/env python
"""One PNG per (dataset, metric) for the epsilon sweep: x = epsilon (log scale), y = metric, one line per method
(mean over folds, bars = fold SD).  Reads <root>/<DS>/eval.csv and, if present, <root>/redteam/simple_*.csv.
usage: python plot_epsweep.py <root>"""
import glob, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHODS = {"lrdp": ("LowRankDP (v2)", "#2a78d6", "o"), "pgm": ("Private-PGM (star)", "#eb6834", "s")}
DSNAME = {"BRCA": "TCGA-BRCA", "COMB": "TCGA-COMBINED"}
# metric -> (label, better direction)
LAB = {
    "w1": ("Wasserstein-1 per gene (pooled)", "lower"), "w1_cls": ("Wasserstein-1 per gene, within class", "lower"),
    "mean_err": ("Error of gene means", "lower"), "sd_err": ("Error of gene SDs", "lower"), "mmd": ("MMD", "lower"),
    "disc_auc": ("Real-vs-synthetic discriminator AUC", "closer to 0.5"),
    "prec": ("Precision (k-NN manifold)", "higher"), "rec": ("Recall (k-NN manifold)", "higher"),
    "dens": ("Density", "closer to 1"), "cov": ("Coverage", "higher"),
    "corr_mae": ("Gene-gene correlation MAE", "lower"), "corr_r": ("Correlation-matrix agreement r", "higher"),
    "coex0.3_tpr": ("Co-expression |r|>0.3 recovered (TPR)", "higher"),
    "coex0.3_fer": ("Co-expression |r|>0.3 false edges", "lower"),
    "coex0.5_tpr": ("Co-expression |r|>0.5 recovered (TPR)", "higher"),
    "coex0.5_fer": ("Co-expression |r|>0.5 false edges", "lower"),
    "wcorr_r": ("Within-class correlation agreement r", "higher"),
    "de_tpr": ("DE genes recovered (TPR)", "higher"), "de_fpr": ("DE false positives (FPR)", "lower"),
    "de_rho": ("DE effect-size rank correlation", "higher"),
    "tstr_acc": ("TSTR accuracy", "higher"), "tstr_f1": ("TSTR macro-F1", "higher"),
    "mahala_auc": ("Mahalanobis MIA AUC", "closer to 0.5"), "dcr_auc": ("DCR MIA AUC", "closer to 0.5"),
    "dcr_tpr@0.1": ("DCR MIA TPR @ 10% FPR", "closer to 0.1"), "dcrr_auc": ("DCR-ratio MIA AUC", "closer to 0.5"),
    "domias_auc": ("DOMIAS MIA AUC", "closer to 0.5"), "domias_tpr@0.1": ("DOMIAS TPR @ 10% FPR", "closer to 0.1"),
    "marg_auc": ("Marginal-histogram MIA AUC", "closer to 0.5"), "dcr_share": ("DCR share (closer to train)", "closer to 0.5"),
    "dcr_syn_train": ("Mean distance synthetic→train", "-"),
}
REAL_REF = {"tstr_acc": "trtr_acc", "tstr_f1": "trtr_f1"}           # real-data reference drawn as a dashed line
CHANCE = {k: 0.5 for k in ("disc_auc", "mahala_auc", "dcr_auc", "dcrr_auc", "domias_auc", "marg_auc", "dcr_share")}
CHANCE.update({"dcr_tpr@0.1": 0.1, "domias_tpr@0.1": 0.1, "mama": 0.5})

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def split(run):
    m, e = re.match(r"(\w+?)_eps([\d.]+)$", run).groups()
    return m, float(e)


def plot(df, col, ds, label, better, out, ref=None, chance=None):
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=150)
    for m, (name, c, mk) in METHODS.items():
        d = df[df.method == m].groupby("eps")[col].agg(["mean", "std"]).sort_index()
        if d.empty:
            continue
        ax.errorbar(d.index, d["mean"], yerr=d["std"].fillna(0), color=c, marker=mk, ms=6, lw=2, capsize=3,
                    elinewidth=1, label=name, markeredgecolor="#fcfcfb", markeredgewidth=1)
    if ref is not None:
        ax.axhline(ref, color="#52514e", ls="--", lw=1)
        ax.annotate("real data", (0.01, ref), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=8, color="#52514e")
    if chance is not None:
        ax.axhline(chance, color="#9a9994", ls=":", lw=1)
    ax.set_xscale("log")
    eps = sorted(df.eps.unique())
    ax.set_xticks(eps); ax.set_xticklabels([f"{e:g}" for e in eps]); ax.minorticks_off()
    ax.set_xlabel("ε  (δ = 1e-5)"); ax.set_ylabel(label)
    ax.set_title(f"{DSNAME[ds]}: {label}" + (f"  ({better} is better)" if better != "-" else ""), fontsize=10, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)



def main(ROOT):
    n = 0
    for ds in ("BRCA", "COMB"):
        f = f"{ROOT}/{ds}/eval.csv"
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df[["method", "eps"]] = df.run.apply(lambda r: pd.Series(split(r)))
        od = f"{ROOT}/plots/{ds}"; os.makedirs(od, exist_ok=True)
        for col in [c for c in df.columns if c in LAB]:
            lab, better = LAB[col]
            ref = df[REAL_REF[col]].mean() if col in REAL_REF else None
            plot(df, col, ds, lab, better, f"{od}/{col.replace('@', '_at_')}.png", ref, CHANCE.get(col)); n += 1
        rows = []                                         # MAMA-MIA, if the red-team stage ran
        for g in glob.glob(f"{ROOT}/redteam/simple_{ds}_*_eps*.csv"):
            m, e = split(os.path.basename(g)[len(f"simple_{ds}_"):-4])
            x = pd.read_csv(g); x = x[x.attack.str.startswith("MAMA-MIA 1+2")]
            rows += [{"method": m, "eps": e, "mama": v} for v in x.auc]
        if rows:
            plot(pd.DataFrame(rows), "mama", ds, "MAMA-MIA 1+2-way AUC (10 bins)", "closer to 0.5",
                 f"{od}/mama_mia.png", chance=0.5); n += 1
        s = df.groupby(["method", "eps"])[[c for c in LAB if c in df.columns]].mean().round(3)
        s.to_csv(f"{ROOT}/plots/{ds}_summary.csv")
    print(f"wrote {n} plots under {ROOT}/plots")


if __name__ == "__main__":
    main(sys.argv[1])
