#!/usr/bin/env python
"""Non-private MVN baseline: the CAMDA challenge generator of MIA_on_bulkRNAseq_CAMDA2026 (mia/generators/mvn.py,
used unchanged; per class N(mu_c, Sigma_c) + N(0, noise_level * Sigma_c), noise_level 0.7 as in the abstract),
fold by fold on the CAMDA splits, in the same output layout as run_lrdp.py / run_tabpfn.py.

python run_mvn.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --out_dir $OUT/TABPFN/BRCA/mvn
"""
import argparse, json, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--noise_level", type=float, default=0.7); p.add_argument("--seed", type=int, default=42)
p.add_argument("--mia_repo", default=os.path.expanduser("~/privacy/MIA_on_bulkRNAseq_CAMDA2026"))
p.add_argument("--out_dir", required=True)
a = p.parse_args()

sys.path.insert(0, a.mia_repo)
import mia  # noqa: E402,F401
pkg = types.ModuleType("mia.generators"); pkg.__path__ = [os.path.join(a.mia_repo, "mia", "generators")]
sys.modules["mia.generators"] = pkg
from mia.generators.mvn import MVNGenerator  # noqa: E402

os.makedirs(a.out_dir, exist_ok=True)
expr = load_expression(a.counts); labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]; genes = list(expr.columns); classes = sorted(labels.unique())
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
log = {"args": vars(a), "classes": classes, "folds": {}}
for f in a.folds or sorted(splits):
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    t0 = time.time()
    g = MVNGenerator(seed=a.seed + f, noise_level=a.noise_level)
    g.fit(expr.loc[tr, genes].values, np.array([classes.index(c) for c in labels.loc[tr]]), len(classes))
    Xs, ys = g.sample(len(tr))
    save_synthetic(a.out_dir, f, Xs, np.array(classes, dtype=object)[ys], genes, a.label_col)
    json.dump({"train": tr, "test": te}, open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w"))
    with open(os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), "w") as fh:
        fh.write("sample_id,membership_label\n" + "".join(f"{s},1\n" for s in tr) + "".join(f"{s},0\n" for s in te))
    log["folds"][str(f)] = {"n_synth": int(len(Xs)), "seconds": round(time.time() - t0, 1)}
    json.dump(log, open(os.path.join(a.out_dir, "config.json"), "w"), indent=1)
    print(f"[MVN] fold {f}: {len(Xs)} rows, {time.time() - t0:.0f}s", flush=True)
