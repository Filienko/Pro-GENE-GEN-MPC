#!/usr/bin/env python
"""Generate with LowRankDP, fold by fold (same output layout as run_copulagen.py).

python run_lrdp.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS \
    --dp_epsilon 10 --k 30 [--external_counts REF.tsv] [--vote 20] --out_dir $OUT/BRCA/lrdp_eps10
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits  # noqa: E402
from run_copulagen import write_fold  # noqa: E402
from lowrank_dp import LowRankDP  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--out_dir", required=True)
p.add_argument("--dp_epsilon", type=float, default=10.0); p.add_argument("--delta", type=float, default=1e-5)
p.add_argument("--k", type=int, default=30); p.add_argument("--shrink", type=float, default=0.3)
p.add_argument("--external_counts", default=None, help="public non-member cohort (anchor + basis)")
p.add_argument("--vote", type=int, default=0, help="cells per class for the vote step (0 = off)")
p.add_argument("--noise_off", nargs="*", default=[], help="ablation only: measurements released without noise")
p.add_argument("--between", action="store_true", help="add the between-class directions to the basis (private-only mode)")
p.add_argument("--v1", action="store_true", help="original settings: zCDP bound, clip quantile 0.9, shares mean 0.40 / basis 0.15 / ccov 0.20")
p.add_argument("--v2", action="store_true", help="no-op: the v2 settings (exact GDP accounting, clip quantile 0.75, private-only shares mean 0.45 / basis 0.20 / ccov 0.10) are the default")
p.add_argument("--accounting", default=None, choices=["zcdp", "gdp"], help="gdp = exact Gaussian-DP conversion (tighter)")
p.add_argument("--clip_q", type=float, default=None)
p.add_argument("--shares", default=None, help='JSON overrides of budget shares, e.g. {"mean":0.5,"ccov":0.1}')
p.add_argument("--basis_share", type=float, default=None, help="override the budget share of the DP-PCA step")
p.add_argument("--seed", type=int, default=0, help="fixed seed for reproducible experiments; -1 = OS entropy (use this for a real release: DP needs secret noise)")
a = p.parse_args()
# defaults = v2 settings; --v1 restores the original ones; explicit flags always win
a.accounting = a.accounting or ("zcdp" if a.v1 else "gdp")
a.clip_q = a.clip_q or (0.9 if a.v1 else 0.75)
if not a.v1 and not a.external_counts and a.basis_share is None:
    a.shares = a.shares or '{"mean":0.45,"basis":0.20,"ccov":0.10}'
os.makedirs(a.out_dir, exist_ok=True)
expr = load_expression(a.counts)
labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]
genes = list(expr.columns)
ext = None
if a.external_counts:
    e = load_expression(a.external_counts).reindex(columns=genes)
    pats = {s[:12] for s in expr.index}                      # drop every patient that is in the private pool
    e = e[[s[:12] not in pats for s in e.index]]
    ext = e.values
    print(f"[LowRankDP] public cohort: {ext.shape}", flush=True)
SH = None
if a.basis_share is not None:
    from lowrank_dp import SHARES_PRIVATE
    SH = dict(SHARES_PRIVATE, basis=a.basis_share)      # renormalised inside LowRankDP
if a.shares:
    from lowrank_dp import SHARES_PRIVATE, SHARES_PUBLIC
    SH = dict(SH or (SHARES_PUBLIC if a.external_counts else SHARES_PRIVATE), **json.loads(a.shares))
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
log = {"args": vars(a), "folds": {}}
for f in a.folds or sorted(splits):
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    t0 = time.time()
    g = LowRankDP(epsilon=a.dp_epsilon, delta=a.delta, k=a.k, external=ext, shrink=a.shrink, seed=None if a.seed < 0 else a.seed + f,
                  vote_cells=a.vote, noise_off=a.noise_off, between=a.between, shares=SH, accounting=a.accounting, clip_q=a.clip_q).fit(expr.loc[tr].values, labels.loc[tr].values)
    Xs, ys = g.sample()          # row count from the noisy class counts, not the exact training size
    write_fold(a.out_dir, f, Xs, ys, genes, a.label_col, tr, te)
    log["folds"][f] = {"seconds": round(time.time() - t0, 1), "info": g.info, "shares": g.shares}
    print(f"[LowRankDP] fold {f}: {Xs.shape} in {time.time() - t0:.1f}s  {({k: round(v, 3) if isinstance(v, float) else v for k, v in g.info.items()})}", flush=True)
json.dump(log, open(os.path.join(a.out_dir, "config.json"), "w"), indent=2, default=str)
