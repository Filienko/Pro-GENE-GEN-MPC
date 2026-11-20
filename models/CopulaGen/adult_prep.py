#!/usr/bin/env python
"""Adult (HF scikit-learn/adult-census-income) in the pipeline's format: a 1089-row subset stratified by income
(same size and 871/218 member splits as BRCA), raw values (no rescaling), categorical columns as integer codes.

Public schema (meta.json), taken from the full 32,561-row table as public metadata, like any DP benchmark's domain:
  numeric columns: [lo, hi] bounds;  categorical columns: list of levels (code = index, ordered by frequency).
  education is dropped (it is education.num one-to-one).

python adult_prep.py --raw ~/privacy/hf_data/adult_raw/adult.csv --out ~/privacy/hf_data/ADULT
"""
import argparse, json, os
import numpy as np, pandas as pd, yaml

p = argparse.ArgumentParser(); p.add_argument("--raw", required=True); p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=1089); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(a.raw).drop(columns=["education"])
y = d.pop("income")
cat = [c for c in d.columns if d[c].dtype == object]; num = [c for c in d.columns if c not in cat]
meta = {"numeric": {c: [float(d[c].min()), float(d[c].max())] for c in num},
        "categorical": {c: d[c].value_counts().index.tolist() for c in cat}, "columns": list(d.columns)}
for c in cat:
    d[c] = d[c].map({v: i for i, v in enumerate(meta["categorical"][c])}).astype(float)
rng = np.random.default_rng(a.seed)
frac = a.n / len(d)
idx = np.concatenate([rng.choice(np.where(y == v)[0], int(round(frac * (y == v).sum())), replace=False) for v in sorted(y.unique())])
idx = np.sort(idx)[: a.n]
d, y = d.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
ids = [f"ADULT_{i:04d}" for i in range(len(d))]
pd.DataFrame(d.values.T, index=d.columns, columns=ids).rename_axis("feature").to_csv(os.path.join(a.out, "ADULT_features.tsv"), sep="\t")
lab = np.where(y.str.contains(">"), "gt50K", "le50K")
pd.DataFrame({"samplesID": ids, "Subtype": lab}).to_csv(os.path.join(a.out, "ADULT_labels.csv"), index=False)
fold = np.empty(len(ids), int)
for c in np.unique(lab):
    w = np.where(lab == c)[0]; w = w[rng.permutation(len(w))]; fold[w] = np.arange(len(w)) % 5
sp = {"splits": {f"split_{k+1}": {"train_index": [ids[i] for i in np.where(fold != k)[0]],
                                   "test_index": [ids[i] for i in np.where(fold == k)[0]]} for k in range(5)}}
yaml.safe_dump(sp, open(os.path.join(a.out, "ADULT_splits.yaml"), "w"))
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(d.shape, pd.Series(lab).value_counts().to_dict(), {k: (len(v["train_index"]), len(v["test_index"])) for k, v in sp["splits"].items()})
