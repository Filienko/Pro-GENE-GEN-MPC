#!/usr/bin/env python
"""BRCA: utility / fidelity of every DP-MVN epsilon next to MVN, Private-PGM and TabPFN-3.5, with the DP-MVN tracing
attack (dpmvn_trace_attack.py) as the privacy column.  Mean over the 5 CAMDA folds.

Sources (nothing is re-evaluated here):
  $OUT/TABPFN/BRCA_all_folds.csv       eval_suite.py + camda_kl_overlap.py per fold (tabpfn_report.py)
  $OUT/EPSWEEP/BRCA/eval.csv           eval_suite.py for the Private-PGM sweep and DP-MVN eps 0.25
  $OUT/TRACE/kl/<run>_f<k>.csv         camda_kl_overlap.py for those runs (same evaluator, PPML-Huskies-Red-Team/src)
  $OUT/TRACE/BRCA_trace_all.csv        tracing attack, score "trace"
  $OUT/TABPFN/BRCA_compare.md          "Strongest attack" of the earlier suite (MeLoMIA-ND, MAMA-MIA v2, MahalaMIA, kNN-LR, fused)

python trace_utility_table.py   ->  $OUT/TRACE/BRCA_trace_utility.md
"""
import glob, os, re

import pandas as pd

OUT = os.environ["OUT"]
# (run in the eval files, run of the release dir / tracing attack, label, formal DP, label in BRCA_compare.md)
ROWS = [("mvn", "mvn", "MVN in the clear (CAMDA, noise 0.7)", "no", "MVN (CAMDA baseline, noise 0.7)"),
        ("tabpfn_35", "tabpfn_35", "TabPFN-3.5, real prompt", "no", "TabPFN-3.5"),
        ("tabpfn_35_ma5dpcovlrdp470", "tabpfn_35_ma5dpcovlrdp470", "TabPFN-3.5, hybrid prompt (real cluster means 65% + DP-MVN ε30 35%)",
         "no", "TabPFN hybrid prompt, dpcov 65%"),
        ("tabpfn_35_dp30", "tabpfn_35_dp30", "TabPFN-3.5, prompt = DP-MVN ε30 only", "ε=30", "TabPFN-3.5 prompted with DP-MVN ε=30")]
for e in ["10000", "1000", "300", "200", "100", "70", "50", "30", "20", "10", "8", "5", "1", "0.5", "0.25"]:
    ROWS.append((f"lrdp_eps{e}", "v2p_priv_eps30" if e == "30" else f"lrdp_eps{e}", f"DP-MVN ε={e}", f"ε={e}", f"DP-MVN ε={e}"))
for e in ["10", "5", "1", "0.5", "0.25"]:
    ROWS.append((f"pgm_eps{e}", f"pgm_eps{e}", f"Private-PGM ε={e}", f"ε={e}", f"Private-PGM ε={e}"))
MET = [("tstr_acc", "TSTR acc ↑", 3), ("tstr_f1", "TSTR macro-F1 ↑", 3), ("w1", "W1 ↓", 3),
       ("kl_real||syn_test", "KL(real‖syn) ↓", 3), ("kl_syn||real_test", "Inv. KL(syn‖real) ↓", 3),
       ("de_tpr", "DE TPR ↑", 3), ("de_fpr", "DE FPR ↓", 3)]

ev = pd.read_csv(os.path.join(OUT, "TABPFN", "BRCA_all_folds.csv"))
sw = pd.read_csv(os.path.join(OUT, "EPSWEEP", "BRCA", "eval.csv"))
kl = [pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "TRACE", "kl", "*.csv"))]
if kl:
    sw = sw.merge(pd.concat(kl)[["run", "fold", "kl_real||syn_test", "kl_syn||real_test"]], on=["run", "fold"], how="left")
ev = pd.concat([ev, sw[~sw.run.isin(ev.run)]])
tr = pd.read_csv(os.path.join(OUT, "TRACE", "BRCA_trace_all.csv")); tr = tr[tr.score == "trace"]
prev = {}
for line in open(os.path.join(OUT, "TABPFN", "BRCA_compare.md")):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 10 and re.match(r"[\d.]+ \(", c[8]):
        prev[c[0]] = c[8]


def fmt(x, d):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


hdr = ["Method", "Formal DP"] + [m[1] for m in MET] + ["Tracing MIA (worst)", "Previous strongest MIA (worst)"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
trtr = ev[["trtr_acc", "trtr_f1"]].mean()
lines.append("| *Real training data (TRTR reference)* | -- | " + f"{trtr.trtr_acc:.3f} | {trtr.trtr_f1:.3f} | " + " | ".join(["--"] * 5) + " | -- | -- |")
for er, rr, lab, dp, plab in ROWS:
    g = ev[ev.run == er]
    if g.empty:
        continue
    m = g.mean(numeric_only=True); t = tr[tr.run == rr]
    tcol = f"{t.auc.mean():.3f} ({t.auc.max():.3f})" if len(t) else "--"
    lines.append("| " + " | ".join([lab, dp] + [fmt(m.get(k), d) for k, _, d in MET] + [tcol, prev.get(plab, "--")]) + " |")
md = ("# BRCA: utility, fidelity and the DP-MVN tracing attack (5 CAMDA folds, means)\n\n"
      "Membership-inference AUC: 0.5 = chance, bar < 0.6.  Single-seed fold noise on TSTR macro-F1 is about ±0.05 per fold.\n\n"
      + "\n".join(lines) + "\n")
open(os.path.join(OUT, "TRACE", "BRCA_trace_utility.md"), "w").write(md)
print(md)
