#!/usr/bin/env python
"""Stage ablation of DP-MVN (dp_mvn.py, used unchanged): which step's noise buys the privacy?

DP-MVN spends its budget rho on 7 Gaussian releases (SHARES in dp_mvn.py):
    n (class counts), clip (4 clipping-threshold histograms), anchor (per-gene centre/scale histogram),
    mean (class sums), var (class MAD sums), basis (DP-PCA Gram matrix), ccov (class factor covariances).
A variant switches the noise of some steps off by giving them an (effectively) infinite budget share after the
generator is built; the other steps keep exactly the noise SD they have at the nominal epsilon, and, because every
random draw still happens in the same order, exactly the same noise realisation (the variants are paired).

Variants (any number, comma-free names):
    base             every step noisy (reproduces the dp_mvn.py release at that epsilon, seed = fold)
    off_<step>       that step noiseless, the rest noisy        -> how much privacy that step's noise provides
    only_<step>      only that step noisy                       -> how much privacy that step alone provides
    off_all          no noise at all (structure only: low rank, clipping, shrinkage, resampling)
    <variant>_x<m>   release m times as many synthetic rows (the resampling step)
Steps: n clip anchor mean var basis ccov.

python dpmvn_ablation.py --counts $BRCA_TSV --annot $BRCA_ANN --splits_yaml $BRCA_SPLITS --eps 30 \
    --variants base off_mean only_mean off_all base_x10 --out_root $OUT/ABLATION/BRCA --dpmvn_dir <dir of dp_mvn.py>
"""
import argparse, json, os, re, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["n", "clip", "anchor", "mean", "var", "basis", "ccov"]
HUGE = 1e12                     # budget share that makes a step's noise SD ~1e-6 of its sensitivity


def parse_variant(v):
    m = re.fullmatch(r"(base|off_all|off_[a-z]+|only_[a-z]+)(?:_x(\d+))?", v)
    if not m:
        raise SystemExit(f"bad variant {v}")
    core, mult = m.group(1), int(m.group(2) or 1)
    if core == "base":
        quiet = []
    elif core == "off_all":
        quiet = list(STEPS)
    elif core.startswith("off_"):
        quiet = [core[4:]]
    else:
        quiet = [s for s in STEPS if s != core[5:]]
    if not set(quiet) <= set(STEPS):
        raise SystemExit(f"unknown step in {v}")
    return quiet, mult


def main():
    p = argparse.ArgumentParser(description="DP-MVN stage ablation")
    p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
    p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
    p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--eps", type=float, default=30.0); p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--variants", nargs="+", required=True); p.add_argument("--out_root", required=True)
    p.add_argument("--seed", type=int, default=0, help="fold seed = seed + fold (as dp_mvn.py)")
    p.add_argument("--dpmvn_dir", default=HERE, help="directory holding dp_mvn.py")
    a = p.parse_args()
    sys.path.insert(0, a.dpmvn_dir); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
    from dp_mvn import DPMVN, eps_spent
    from factor_core import load_expression, load_labels, load_splits
    from run_copulagen import write_fold
    expr = load_expression(a.counts); labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[labels.index]; genes = list(expr.columns)
    splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
    for v in a.variants:
        quiet, mult = parse_variant(v)
        out = os.path.join(a.out_root, f"eps{a.eps:g}_{v}"); os.makedirs(out, exist_ok=True)
        cfg = os.path.join(out, "config.json")
        log = json.load(open(cfg)) if os.path.exists(cfg) else {
            "args": {**vars(a), "variants": None, "variant": v, "dp_epsilon": a.eps, "noiseless_steps": quiet, "mult": mult},
            "folds": {}}
        for f in a.folds or sorted(splits):
            if str(f) in log["folds"] and os.path.exists(os.path.join(out, f"synthetic_data_split_{f}.csv")):
                continue
            tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
            t0 = time.time()
            g = DPMVN(epsilon=a.eps, delta=a.delta, seed=a.seed + f)
            for s in quiet:
                g.shares[s] = HUGE
            g.fit(expr.loc[tr].values, labels.loc[tr].values)
            n0 = int(round(sum(g.n_hat.values())))
            Xs, ys = g.sample(n0 * mult)
            write_fold(out, f, Xs, ys, genes, a.label_col, tr, te)
            noisy = [(se, sg) for se, sg in g.ledger if se / sg < 1e4]          # releases that still carry noise
            info = {**g.info, "eps_noisy_part": eps_spent(noisy, a.delta) if noisy else 0.0, "n_rows": int(len(ys))}
            log["folds"][str(f)] = {"seconds": round(time.time() - t0, 1), "info": info}
            json.dump(log, open(cfg, "w"), indent=2, default=str)
            print(f"{v} fold {f}: {Xs.shape}  k_eff {g.info.get('k_eff')}  eps of the noisy steps {info['eps_noisy_part']:.2f}  "
                  f"sigma_mean {g.info['sigma_mean']:.3g}  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
