#!/usr/bin/env python
"""Generate with MIAV (Chaibub Neto, ICLR 2026; ~/TabSDS-genomic/src/MIAV, miav_tabpfn_generator used unchanged),
fold by fold, in the same layout as run_lrdp.py.  With tabpfn 9.x the plain TabPFNRegressor() / TabPFNClassifier()
inside MIAV are TabPFN-3.5 (default weights); --model_version v2 swaps in the open v2 weights (smoke tests only).

MIAV, per column j: m = sorted Uniform(0,1) draws re-ordered by the ranks of column j; TabPFN is fitted on one half
(m -> x_j) and predicts x_j for the other half from that half's own m; halves are swapped and stacked.  The subtype
column is generated the same way (TabPFN classifier).  Training rows are shuffled before the halving.

Runs in the camda_sota env.  Resumable per fold.
"""
import argparse, functools, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--out_dir", required=True); p.add_argument("--seed", type=int, default=42)
p.add_argument("--miav_dir", default=os.path.expanduser("~/TabSDS-genomic/src/MIAV"))
p.add_argument("--model_version", default="default", help='"default" = TabPFN-3.5; "v2" = open weights')
p.add_argument("--n_genes", type=int, default=None, help="smoke test: first n genes only")
a = p.parse_args()

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
sys.path.insert(0, a.miav_dir)
import utility_functions_implementing_tabpfn_generators_iclr as M  # noqa: E402
# compatibility shim: newer scipy's rankdata returns float ranks, which MIAV then uses as indices
_rank = M.ranking_with_random_tie_breaking
M.ranking_with_random_tie_breaking = lambda x: _rank(x).astype(np.int64)
if a.model_version != "default":
    from tabpfn.constants import ModelVersion
    v = ModelVersion(a.model_version)
    M.TabPFNRegressor = functools.partial(M.TabPFNRegressor.create_default_for_version, v)
    M.TabPFNClassifier = functools.partial(M.TabPFNClassifier.create_default_for_version, v)

os.makedirs(a.out_dir, exist_ok=True)
expr = load_expression(a.counts)
labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]
genes = list(expr.columns)[: a.n_genes] if a.n_genes else list(expr.columns)
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
cfg_path = os.path.join(a.out_dir, "config.json")
log = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"args": vars(a), "folds": {}}
for f in a.folds or sorted(splits):
    if os.path.exists(os.path.join(a.out_dir, f"synthetic_data_split_{f}.csv")):
        print(f"[MIAV] fold {f}: exists, skipped", flush=True); continue
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    np.random.seed(a.seed + f)
    order = np.random.permutation(len(tr))
    df = expr.loc[tr, genes].iloc[order].astype(np.float64).reset_index(drop=True)
    df[a.label_col] = labels.loc[tr].iloc[order].astype(str).values          # categorical -> TabPFN classifier
    t0 = time.time()
    Z = M.miav_tabpfn_generator(df, show_progress=True)
    Xs = Z[genes].astype(np.float32).values; ys = Z[a.label_col].astype(str).values
    save_synthetic(a.out_dir, f, Xs, ys, genes, a.label_col)
    json.dump({"train": tr, "test": te}, open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w"))
    pd.DataFrame({"sample_id": tr + te, "membership_label": [1] * len(tr) + [0] * len(te)}).to_csv(
        os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
    log["folds"][str(f)] = {"seconds": round(time.time() - t0, 1)}
    json.dump(log, open(cfg_path, "w"), indent=2)
    print(f"[MIAV] fold {f}: {Xs.shape} in {time.time() - t0:.0f}s", flush=True)
