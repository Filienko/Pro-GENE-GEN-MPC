#!/usr/bin/env python
"""TabPFN vs LowRankDP (several eps) vs P-PGM: merge eval_suite + MAMA-MIA results, write summary tables and one
PNG per (dataset, metric): x = epsilon of LowRankDP (line), TabPFN drawn as horizontal lines (non-private, eps = inf).
Means are over the folds TabPFN has; fold SD in brackets / as bars.   usage: python tabpfn_report.py <TABPFN root>"""
import glob, os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = sys.argv[1]; OUT = os.path.dirname(ROOT.rstrip("/"))
sys.argv = [sys.argv[0], ROOT]
from plot_epsweep import LAB, CHANCE, REAL_REF  # noqa: E402  (metric labels shared with the eps-sweep plots)

LRDP = {1: "EPSWEEP/{D}/lrdp_eps1", 5: "EPSWEEP/{D}/lrdp_eps5", 8: "EPSWEEP/{D}/lrdp_eps8", 10: "EPSWEEP/{D}/lrdp_eps10", 20: "EPSWEEP/{D}/lrdp_eps20", 30: "LRDP/v2/{D}/v2p_priv_eps30"}
EVAL = ["EPSWEEP/{D}/eval.csv", "LRDP/v2/{D}/eval_p.csv"]
TAB = {"tabpfn_35": ("TabPFN-3.5, gene by gene", "#1baf7a", "-"), "miav_35": ("MIAV + TabPFN-3.5", "#6250d6", "--"),
       "tabpfn_pe_sig0": ("TabPFN-PE, noiseless votes (ceiling)", "#b8860b", ":")}
# DP TabPFN: the prompt is a LowRankDP release (post-processing), so these sit at that release's eps
TABDP = {"tabpfn_35_dp10": ("TabPFN-3.5 on LowRankDP context", "#0f8a5f", "D", 10),
         "tabpfn_35_dp30": ("TabPFN-3.5 on LowRankDP context", "#0f8a5f", "D", 30),
         "tabpfn_35_dp10pub": ("TabPFN-3.5 on LowRankDP + public context", "#c2378a", "^", 10),
         "tabpfn_pe10": ("TabPFN-PE (private evolution)", "#b8860b", "P", 10),
         "tabpfn_pe30": ("TabPFN-PE (private evolution)", "#b8860b", "P", 30)}
PGM = "pgm_eps10"
KEEP = ["w1", "w1_cls", "mean_err", "sd_err", "disc_auc", "prec", "rec", "corr_mae", "wcorr_r", "coex0.3_tpr", "coex0.3_fer",
        "de_tpr", "de_fpr", "tstr_acc", "tstr_f1", "mahala_auc", "dcr_auc", "dcr_tpr@0.1", "dcrr_auc", "domias_auc",
        "domias_tpr@0.1", "marg_auc", "dcr_share", "mama4", "mama10",
        "kl_real||syn_train", "kl_syn||real_train", "kl_real||syn_test", "kl_syn||real_test", "overlap_prop"]
LAB = dict(LAB, mama4=("MAMA-MIA AUC, 4 bins", "closer to 0.5"), mama10=("MAMA-MIA AUC, 10 bins", "closer to 0.5"),
           **{"kl_real||syn_train": ("KL(real train || syn), CAMDA", "lower"), "kl_syn||real_train": ("Inverted KL(syn || real train)", "lower"),
              "kl_real||syn_test": ("KL(real test || syn), CAMDA", "lower"), "kl_syn||real_test": ("Inverted KL(syn || real test)", "lower"),
              "overlap_prop": ("Top-feature overlap with real (CAMDA)", "higher")})
HERE = os.path.dirname(os.path.abspath(__file__))
CAMDA_SRC = os.path.expanduser("~/privacy/PPML-Huskies-Red-Team/src")   # copy of the CAMDA evaluator (verified identical KL)
DATA = {"BRCA": ["--counts", os.environ["BRCA_TSV"], "--annot", os.environ["BRCA_ANN"], "--label_col", "Subtype"],
        "COMB": ["--counts", os.environ["COMB_TSV"], "--annot", os.environ["COMB_ANN"], "--label_col", "cancer_type"]}


def ensure_kl(D, dirs):
    """CAMDA KL / inverted KL / feature overlap for every (run dir, fold) that lacks it; cached per fold."""
    os.makedirs(f"{ROOT}/{D}/kl", exist_ok=True)
    for d in dirs:
        for fx in sorted(glob.glob(f"{d}/synthetic_data_split_*.csv")):
            f = fx.rsplit("_", 1)[1][:-4]; out = f"{ROOT}/{D}/kl/{os.path.basename(d)}_f{f}.csv"
            if not os.path.exists(out):
                subprocess.run([sys.executable, f"{HERE}/camda_kl_overlap.py", *DATA[D], "--camda_src", CAMDA_SRC,
                                "--syn_dir", d, "--folds", f, "--out_csv", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=dict(os.environ, OMP_NUM_THREADS="6"))
CHANCE = dict(CHANCE, mama4=0.5, mama10=0.5)

for D in ("BRCA", "COMB"):
    ev = [pd.read_csv(f) for f in glob.glob(f"{ROOT}/{D}/eval/*.csv")]
    if not ev:
        continue
    tab = pd.concat(ev)
    folds = sorted(tab.fold.unique())
    base = pd.concat([pd.read_csv(f"{OUT}/{p.format(D=D)}") for p in EVAL if os.path.exists(f"{OUT}/{p.format(D=D)}")])
    names = {os.path.basename(v.format(D=D)): f"lrdp_eps{e}" for e, v in LRDP.items()}
    base = base[base.run.isin(list(names) + [PGM])].copy()
    base["run"] = base.run.map(lambda r: names.get(r, r))
    d = pd.concat([tab, base])
    m = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/{D}/mama/*.csv")])
    m["run"] = m.run.map(lambda r: names.get(r, r))
    m = m.pivot_table(index=["run", "fold"], columns="bins", values="auc").rename(columns=lambda b: f"mama{b}").reset_index()
    d = d.merge(m, on=["run", "fold"], how="left")
    ensure_kl(D, [x for x in glob.glob(f"{ROOT}/{D}/*") if os.path.exists(f"{x}/config.json")]
              + [f"{OUT}/{v.format(D=D)}" for v in LRDP.values()] + [f"{OUT}/EPSWEEP/{D}/{PGM}"])
    kf = [pd.read_csv(f) for f in glob.glob(f"{ROOT}/{D}/kl/*.csv") if os.path.getsize(f) > 0]
    if kf:
        k = pd.concat(kf); k = k[~k.run.str.startswith("REAL")]
        k["run"] = k.run.map(lambda r: names.get(r, r))
        d = d.merge(k[["run", "fold"] + [c for c in k.columns if c.startswith(("kl_", "overlap_prop"))]], on=["run", "fold"], how="left")
    d.to_csv(f"{ROOT}/{D}_all_folds.csv", index=False)
    s = d[d.fold.isin(folds)]
    cols = [c for c in KEEP if c in s.columns]
    mean, sd = s.groupby("run")[cols].mean(), s.groupby("run")[cols].std()
    order = [r for r in [*TAB, *TABDP, "lrdp_eps30", "lrdp_eps20", "lrdp_eps10", "lrdp_eps8", "lrdp_eps5", "lrdp_eps1", PGM] if r in mean.index]
    mean.loc[order].round(3).to_csv(f"{ROOT}/{D}_summary.csv")
    with open(f"{ROOT}/{D}_summary.md", "w") as fh:
        nf = s.groupby("run").fold.nunique()
        fh.write(f"{D}: mean (fold SD) over the folds each run has, of {[int(x) for x in folds]}\n\n| metric | " + " | ".join(order)
                 + " |\n|---|" + "---|" * len(order) + "\n| n folds | " + " | ".join(str(nf[r]) for r in order) + " |\n")
        for c in cols:
            fh.write(f"| {c} | " + " | ".join(f"{mean.loc[r, c]:.3f} ({sd.loc[r, c]:.3f})" for r in order) + " |\n")
    od = f"{ROOT}/plots/{D}"; os.makedirs(od, exist_ok=True)
    eps = [e for e in sorted(LRDP) if f"lrdp_eps{e}" in mean.index]
    for c in cols:
        lab, better = LAB[c]
        fig, ax = plt.subplots(figsize=(5.8, 3.9), dpi=150)
        lm = [mean.loc[f"lrdp_eps{e}", c] for e in eps]; ls = [sd.loc[f"lrdp_eps{e}", c] for e in eps]
        ax.errorbar(eps, lm, yerr=[x if x == x else 0 for x in ls], color="#2a78d6", marker="o", ms=6, lw=2, capsize=3, label="LowRankDP (v2)",
                    markeredgecolor="#fcfcfb")
        if PGM in mean.index:
            ax.errorbar([10], [mean.loc[PGM, c]], yerr=[sd.loc[PGM, c]], color="#eb6834", marker="s", ms=7, lw=0,
                        elinewidth=1, capsize=3, label="Private-PGM, ε=10", markeredgecolor="#fcfcfb")
        for nm, col, mk in dict.fromkeys(v[:3] for v in TABDP.values()):
            rr = [(v[3], r) for r, v in TABDP.items() if v[:3] == (nm, col, mk) and r in mean.index]
            if rr:
                ax.errorbar([e for e, _ in rr], [mean.loc[r, c] for _, r in rr], yerr=[sd.loc[r, c] if sd.loc[r, c] == sd.loc[r, c] else 0 for _, r in rr],
                            color=col, marker=mk, ms=7, lw=1.5, capsize=3, label=f"{nm} (DP)", markeredgecolor="#fcfcfb")
        for r, (nm, col, st) in TAB.items():
            if r in mean.index:
                ax.axhline(mean.loc[r, c], color=col, ls=st, lw=2, label=f"{nm} (not DP)")
                ax.axhspan(mean.loc[r, c] - sd.loc[r, c], mean.loc[r, c] + sd.loc[r, c], color=col, alpha=0.12, lw=0)
        if c in REAL_REF:
            ax.axhline(s[REAL_REF[c]].mean(), color="#52514e", ls=":", lw=1, label="real data")
        if CHANCE.get(c) is not None:
            ax.axhline(CHANCE[c], color="#9a9994", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels([f"{e:g}" for e in eps]); ax.minorticks_off()
        ax.set_xlabel("ε of LowRankDP  (δ = 1e-5)"); ax.set_ylabel(lab)
        nm = {"BRCA": "TCGA-BRCA", "COMB": "TCGA-COMBINED"}[D]
        ax.set_title(f"{nm}: {lab}  ({better} is better)\nfolds {', '.join(map(str, folds))}", fontsize=9.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.legend(frameon=False, fontsize=7.5)
        fig.tight_layout(); fig.savefig(f"{od}/{c.replace('@', '_at_')}.png"); plt.close(fig)
    print(open(f"{ROOT}/{D}_summary.md").read())
