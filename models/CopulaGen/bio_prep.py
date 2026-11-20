#!/usr/bin/env python
"""Bioresponse (HF inria-soda/tabular-benchmark, clf_num/Bioresponse.csv) in the pipeline's format.
BIO     = balanced 1089-row subset (BRCA's size, splits 871/218);  BIOFULL = all 3434 rows.
Values: public bounds [0, 1] mapped to [2, 22] (the scale DP-MVN's private anchor assumes; no private data used).

python bio_prep.py --raw ~/privacy/hf_data/bioresponse/clf_num/Bioresponse.csv --out ~/privacy/hf_data/BIO [--full]
"""
import argparse, os
import numpy as np, pandas as pd, yaml

p = argparse.ArgumentParser(); p.add_argument("--raw", required=True); p.add_argument("--out", required=True)
p.add_argument("--full", action="store_true"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(a.raw); rng = np.random.default_rng(0); tag = "BIOFULL" if a.full else "BIO"
if not a.full:
    idx = np.concatenate([rng.choice(np.where(d.target == c)[0], n, replace=False) for c, n in [(0, 545), (1, 544)]])
    d = d.iloc[np.sort(idx)].reset_index(drop=True)
ids = [f"{'BIOF' if a.full else 'BIO'}_{i:04d}" for i in range(len(d))]
X = 2.0 + 20.0 * d.drop(columns=["target"]).values
pd.DataFrame(X.T, index=d.columns[:-1], columns=ids).rename_axis("feature").to_csv(os.path.join(a.out, f"{tag}_features.tsv"), sep="\t")
lab = np.where(d.target.values == 1, "active", "inactive")
pd.DataFrame({"samplesID": ids, "Subtype": lab}).to_csv(os.path.join(a.out, f"{tag}_labels.csv"), index=False)
fold = np.empty(len(ids), int)
for c in np.unique(lab):
    w = np.where(lab == c)[0]; w = w[rng.permutation(len(w))]; fold[w] = np.arange(len(w)) % 5
sp = {"splits": {f"split_{k+1}": {"train_index": [ids[i] for i in np.where(fold != k)[0]],
                                   "test_index": [ids[i] for i in np.where(fold == k)[0]]} for k in range(5)}}
yaml.safe_dump(sp, open(os.path.join(a.out, f"{tag}_splits.yaml"), "w"))
print(X.shape, {k: (len(v["train_index"]), len(v["test_index"])) for k, v in sp["splits"].items()})
