#!/usr/bin/env python
"""Run the Huskies red-team MAMA-MIA scorer (SyntheticData_MIA_PGM/competition_mia.py) on run dirs.

Protocol (same as eval_suite.py): a random half of the held-out split is the attacker's
reference population; scores are computed for the other half (non-members) and a random
subset of the training split (members, 4x as many).  Reports AUC and TPR@FPR=0.1 for
n_bins in --bins.

python mamamia_eval.py --counts $BRCA_TSV --mia_repo ~/privacy/SyntheticData_MIA_PGM --syn_dir run1 run2 ...
"""
import argparse, glob, json, os, re, sys, contextlib, io

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FactorGen"))
from factor_core import load_expression  # noqa: E402
from eval_suite import auc, tpr_at  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True)
p.add_argument("--mia_repo", default=os.path.expanduser("~/privacy/SyntheticData_MIA_PGM"))
p.add_argument("--syn_dir", nargs="+", required=True)
p.add_argument("--bins", type=int, nargs="+", default=[4, 10])
p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--out_csv", default=None)
a = p.parse_args()
sys.path.insert(0, a.mia_repo)
from competition_mia import build_pgm_focal_points, encode_dataframes, mama_mia_score  # noqa: E402

expr = load_expression(a.counts)
rows = []
for sd in a.syn_dir:
    for fx in sorted(glob.glob(os.path.join(sd, "synthetic_data_split_*.csv"))):
        f = int(re.findall(r"split_(\d+)", fx)[-1])
        if a.folds and f not in a.folds:
            continue
        ids = json.load(open(os.path.join(sd, f"split_{f}_ids.json")))
        S = pd.read_csv(fx)
        cols = list(S.columns)
        rng = np.random.default_rng(f)
        te = np.array(ids["test"])[rng.permutation(len(ids["test"]))]
        ref, out = te[: len(te) // 2], te[len(te) // 2:]
        inn = np.array(ids["train"])[rng.choice(len(ids["train"]), min(len(ids["train"]), 4 * len(out)), replace=False)]
        R = expr.loc[ref, cols].reset_index(drop=True)
        T = expr.loc[np.r_[inn, out], cols].reset_index(drop=True)
        y = np.r_[np.ones(len(inn)), np.zeros(len(out))]
        r = {"run": os.path.basename(os.path.normpath(sd)), "fold": f}
        for nb in a.bins:
            with contextlib.redirect_stdout(io.StringIO()):
                Se, Re, Te = encode_dataframes(S.copy(), R.copy(), T.copy(), cols, nb, standardize=True, name=f"cg_{os.getpid()}")  # unique name: the encoder caches thresholds in a shared file
                sc = mama_mia_score(Se, Re, Te, build_pgm_focal_points(cols, None))
            r[f"mamamia{nb}_auc"] = auc(sc[y == 1], sc[y == 0])
            r[f"mamamia{nb}_tpr@0.1"] = tpr_at(sc[y == 1], sc[y == 0], 0.1)
        rows.append(r)
        print(r, flush=True)
df = pd.DataFrame(rows)
print(df.drop(columns="fold").groupby("run").mean().round(3).to_string())
if a.out_csv:
    df.to_csv(a.out_csv, index=False)
