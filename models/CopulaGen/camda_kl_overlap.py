#!/usr/bin/env python
"""KL divergence and top-feature overlap, computed with the official CAMDA evaluator code
(pgm_attack_CAMDA/src/evaluation: Statistics.compute_kl_divergences,
Statistics.count_feature_overlap, ModelEvaluator.train_and_evaluate / get_top_n_feature_importance).

kl_real||syn_{train,test} : what CAMDA reports as kl_mean_train / kl_mean_test
                            (compute_kl_divergences(X_syn, X_real) = sum p_real log(p_real / p_syn)).
kl_syn||real_{train,test} : the inverted direction (arguments swapped), sum p_syn log(p_syn / p_real).
overlap_{count,prop}       : top-10 positive OvR-logreg coefficients per class, union over classes,
                            |syn ∩ real| / |syn|, classifier trained on synthetic vs on real train.

Pseudo-runs: REAL_TRAIN (synthetic := real train; KL floor for the test columns, overlap = 1 by
construction) and REAL_TEST (synthetic := held-out real; sampling-noise floor for the overlap).

python camda_kl_overlap.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --oracle --syn_dir run1 ...
"""
import argparse, glob, json, os, re, sys, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FactorGen"))
from factor_core import load_expression  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True)
p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID")
p.add_argument("--label_col", default="Subtype")
p.add_argument("--camda_src", default=os.path.expanduser("~/privacy/pgm_attack_CAMDA/src"))
p.add_argument("--syn_dir", nargs="+", required=True)
p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--oracle", action="store_true")
p.add_argument("--out_csv", default=None)
a = p.parse_args()

sys.path.insert(0, a.camda_src)
from evaluation.evaluate import ModelEvaluator  # noqa: E402
from evaluation.utils.stats import Statistics  # noqa: E402

ev = object.__new__(ModelEvaluator)  # skip config/IO __init__, keep the official methods
ev.random_seed = 777

expr = load_expression(a.counts)
ann = pd.read_csv(a.annot)
labels = ann.set_index(ann[a.sample_col].astype(str))[a.label_col].astype(str)


def score(S, ys, Xtr, ytr, Xte, yte):
    from sklearn.preprocessing import LabelEncoder
    yte_enc = LabelEncoder().fit_transform(yte)
    acc_s, _, f_syn, *_ = ev.train_and_evaluate(S, ys, Xte, yte, yte_enc, Xtr, reset_model=True)
    acc_r, _, f_real, *_ = ev.train_and_evaluate(Xtr, ytr, Xte, yte, yte_enc, Xtr, reset_model=True)
    cnt, prop = Statistics.count_feature_overlap(f_syn, f_real)
    return {
        "acc_syn": acc_s, "acc_real": acc_r,
        "overlap_count": cnt, "overlap_prop": prop,
        "kl_real||syn_train": Statistics.compute_kl_divergences(S, Xtr)[0],
        "kl_real||syn_test": Statistics.compute_kl_divergences(S, Xte)[0],
        "kl_syn||real_train": Statistics.compute_kl_divergences(Xtr, S)[0],
        "kl_syn||real_test": Statistics.compute_kl_divergences(Xte, S)[0],
    }


rows = []
done_oracle = set()
for sd in a.syn_dir:
    for fx in sorted(glob.glob(os.path.join(sd, "synthetic_data_split_*.csv"))):
        f = int(re.findall(r"split_(\d+)", fx)[-1])
        if a.folds and f not in a.folds:
            continue
        ids = json.load(open(os.path.join(sd, f"split_{f}_ids.json")))
        S_df = pd.read_csv(fx)
        cols = list(S_df.columns)
        ys = pd.read_csv(os.path.join(sd, f"synthetic_labels_split_{f}.csv")).iloc[:, 0].astype(str).values
        Xtr = expr.loc[ids["train"], cols].values
        Xte = expr.loc[ids["test"], cols].values
        ytr = labels.loc[ids["train"]].values
        yte = labels.loc[ids["test"]].values
        jobs = [(os.path.basename(os.path.normpath(sd)), S_df.values, ys)]
        if a.oracle and f not in done_oracle:
            done_oracle.add(f)
            jobs += [("REAL_TRAIN", Xtr, ytr), ("REAL_TEST", Xte, yte)]
        for name, S, y in jobs:
            r = {"run": name, "fold": f, **score(S, y, Xtr, ytr, Xte, yte)}
            rows.append(r)
            print(r, flush=True)

df = pd.DataFrame(rows)
print(df.drop(columns="fold").groupby("run", sort=False).mean().round(3).to_string())
if a.out_csv:
    df.to_csv(a.out_csv, index=False)
