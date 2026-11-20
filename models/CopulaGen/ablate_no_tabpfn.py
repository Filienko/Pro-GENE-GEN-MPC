#!/usr/bin/env python
"""Ablation: release the dpcov cluster rows themselves (the TabPFN prompt of tabpfn_35_dpcov_p100) without TabPFN,
to see what TabPFN adds on top of its prompt.  Same code path as run_tabpfn.py (mdav / microagg_prompt, dpcov
spread from the DP-MVN eps30 release), same seeds per fold.

python ablate_no_tabpfn.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS \
    --jitter_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30 --folds 3 --out_dir $OUT/TABPFN/BRCA/dpcov_prompt_only
"""
import argparse, ast, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402

src = open(os.path.join(HERE, "run_tabpfn.py")).read(); tree = ast.parse(src); ns = {"np": np}
exec("\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("mdav", "microagg_prompt")), ns)

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--jitter_dir", required=True); p.add_argument("--k", type=int, default=5)
p.add_argument("--seed", type=int, default=42); p.add_argument("--out_dir", required=True)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
expr = load_expression(a.counts); labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]; genes = list(expr.columns); classes = sorted(labels.unique())
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
for f in a.folds or sorted(splits):
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    X = expr.loc[tr, genes].values.astype(np.float64); y = np.array([classes.index(c) for c in labels.loc[tr]])
    dp = (pd.read_csv(os.path.join(a.jitter_dir, f"synthetic_data_split_{f}.csv"))[genes].values.astype(np.float64),
          np.array([classes.index(c) for c in pd.read_csv(os.path.join(a.jitter_dir, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]]))
    Xp, yp, _ = ns["microagg_prompt"](X, y, a.k, "dpcov", np.random.default_rng(a.seed + 1000 + f), dp)   # rng_p of run_tabpfn.py
    perm = np.random.default_rng(f).permutation(len(yp))
    save_synthetic(a.out_dir, f, Xp[perm], np.array(classes, dtype=object)[yp[perm]], genes, a.label_col)
    json.dump({"train": tr, "test": te}, open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w"))
    pd.DataFrame({"sample_id": tr + te, "membership_label": [1] * len(tr) + [0] * len(te)}).to_csv(
        os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
    print(f"fold {f}: {len(yp)} rows", flush=True)
json.dump({"args": vars(a)}, open(os.path.join(a.out_dir, "config.json"), "w"), indent=1)
