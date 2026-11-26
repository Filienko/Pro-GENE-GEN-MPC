#!/usr/bin/env python
"""Tables for the DP-MVN stage ablation (dpmvn_ablation.py + dpmvn_ablation_eval.sh).  Mean over the 5 CAMDA folds.

"Other attacks" = per fold the highest of MahalaMIA, DCR, DCR ratio, DOMIAS, marginal (eval_suite.py), MAMA-lite 4 / 10
bins and kNN-LR (clipped), then averaged (the "strongest attack" convention of BRCA_compare.md).

python dpmvn_ablation_table.py [--roots $OUT/ABLATION/BRCA $OUT/ABLATION/BRCA_eps10]  ->  $OUT/ABLATION/BRCA_ablation.md
"""
import argparse, glob, json, os

import numpy as np, pandas as pd

OUT = os.environ["OUT"]
p = argparse.ArgumentParser()
p.add_argument("--roots", nargs="+", default=[os.path.join(OUT, "ABLATION", "BRCA"), os.path.join(OUT, "ABLATION", "BRCA_eps10")])
p.add_argument("--out_md", default=os.path.join(OUT, "ABLATION", "BRCA_ablation.md"))
a = p.parse_args()
STEPS = ["mean", "var", "basis", "anchor", "n", "clip", "ccov"]
ORDER = ["base"] + [f"off_{s}" for s in STEPS] + [f"only_{s}" for s in STEPS] + ["off_all", "base_x10", "off_all_x10"]
LABEL = {"base": "base: every step noisy", "off_all": "no noise at all", "base_x10": "base, 10x synthetic rows",
         "off_all_x10": "no noise, 10x synthetic rows",
         **{f"off_{s}": f"noise off: {s}" for s in STEPS}, **{f"only_{s}": f"noise only on: {s}" for s in STEPS}}
OTHER = ["mahala_auc", "dcr_auc", "dcrr_auc", "domias_auc", "marg_auc", "mama4", "mama10", "knn"]


def cat(pattern):
    fs = [f for f in glob.glob(pattern) if "scores" not in f]
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def delta(v, t, base, quiet):
    """off_* / *_x10 variants against the full-noise base, only_* against the noiseless release"""
    if pd.isna(t) or v in ("base", "off_all"):
        return "--"
    ref = quiet if v.startswith("only_") else base
    return "--" if ref is None or pd.isna(ref) else f"{t - ref:+.3f}"


def f3(x):
    return "--" if x is None or pd.isna(x) else f"{x:.3f}"


md = ["# DP-MVN stage ablation (BRCA, 5 CAMDA folds)", "",
      "Each variant switches the Gaussian noise of some DP-MVN steps off (dpmvn_ablation.py); the noisy steps keep the noise "
      "SD and realisation they have in the release at the nominal ε.  ε(noisy) = the privacy cost of the steps that still "
      "carry noise.  Tracing = dpmvn_trace_attack.py with the release's own noise SDs; its two parts are the class-mean "
      "channel and the per-gene spread (MAD) channel.  Membership AUC: 0.5 = chance, bar < 0.6.", ""]
for root in a.roots:
    vs = sorted(glob.glob(os.path.join(root, "eps*_*")))
    vs = [v for v in vs if os.path.isdir(v)]
    if not vs:
        continue
    eps = os.path.basename(vs[0]).split("_")[0]
    tr = cat(os.path.join(root, "attack", "trace_*.csv"))
    ev = cat(os.path.join(root, "eval", "*.csv")); kl = cat(os.path.join(root, "kl", "*.csv"))
    mm = cat(os.path.join(root, "mama", "*.csv")); kn = cat(os.path.join(root, "knn", "knn_*.csv"))
    rows = []
    for v in ORDER:
        run = f"{eps}_{v}"; d = os.path.join(root, run)
        if not os.path.isdir(d):
            continue
        info = [x["info"] for x in json.load(open(os.path.join(d, "config.json")))["folds"].values()]
        r = {"variant": v, "eps_noisy": np.mean([i["eps_noisy_part"] for i in info]),
             "k_eff": np.median([i["k_eff"] for i in info])}
        t = tr[tr.run == run] if len(tr) else tr
        if len(t):
            for s in ["trace", "trace_mean", "trace_var"]:
                r[s] = t[t.score == s].auc.mean()
            r["trace_worst"] = t[t.score == "trace"].auc.max()
        e = ev[ev.run == run] if len(ev) else ev
        if len(e):
            for c in ["tstr_acc", "tstr_f1", "w1", "de_tpr", "de_fpr", "wcorr_r"]:
                r[c] = e[c].mean()
            per = e.set_index("fold")[OTHER[:5]].copy()
            if len(mm):
                m_ = mm[mm.run == run].pivot_table(index="fold", columns="bins", values="auc")
                if len(m_):
                    per["mama4"], per["mama10"] = m_.get(4), m_.get(10)
            if len(kn):
                k_ = kn[(kn.run == run) & (kn.score == "knn_lr_clip")].set_index("fold").auc
                if len(k_):
                    per["knn"] = k_
            r["other"] = per.max(1).mean(); r["other_which"] = per.mean().idxmax()
        k = kl[kl.run == run] if len(kl) else kl
        if len(k):
            r["kl"] = k["kl_real||syn_test"].mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    ix = R.set_index("variant").trace; base, quiet = ix.get("base"), ix.get("off_all")
    md += [f"## ε = {eps[3:]}", ""]
    hdr = ["Variant", "ε(noisy)", "k_eff", "Tracing (worst)", "Δ (off: vs base; only: vs no noise)","mean ch.", "spread ch.", "Other attacks (which)",
           "TSTR acc", "TSTR F1", "W1", "KL(real‖syn)", "DE TPR / FPR", "within-subtype corr"]
    md += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, r in R.iterrows():
        g = r.get
        md.append("| " + " | ".join([
            LABEL[r.variant], f"{r.eps_noisy:.1f}", f"{r.k_eff:.0f}",
            f"{f3(g('trace'))} ({f3(g('trace_worst'))})", delta(r.variant, g("trace"), base, quiet),
            f3(g("trace_mean")), f3(g("trace_var")),
            "--" if pd.isna(g("other")) else f"{g('other'):.3f} ({g('other_which')})",
            f3(g("tstr_acc")), f3(g("tstr_f1")), f3(g("w1")), f3(g("kl")),
            "--" if pd.isna(g("de_tpr")) else f"{g('de_tpr'):.3f} / {g('de_fpr'):.3f}", f3(g("wcorr_r"))]) + " |")
    md.append("")
    R.to_csv(os.path.join(root, "ablation_summary.csv"), index=False)
open(a.out_md, "w").write("\n".join(md) + "\n")
print("\n".join(md))
