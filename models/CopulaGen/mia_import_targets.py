#!/usr/bin/env python
"""Import our synthetic releases as targets of MIA_on_bulkRNAseq_CAMDA2026, the way that repo imports the blue
team's published NoisyDiffusion data (mia/targets.py: _import_published_nd), so its attacks run on them unchanged.

Target name: ours@run=<run>  ->  $CAMDA_ARTIFACTS/targets/<DS>/ours/run=<run>/split_k/{synthetic_data.csv,
synthetic_labels.csv, meta.json}.  Refuses a fold whose training ids differ from the repo's member ids.

python mia_import_targets.py --dataset BRCA --run_dir DIR [DIR ...]      (needs ~/privacy/mia_verify/env.sh sourced)
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="BRCA"); p.add_argument("--run_dir", nargs="+", required=True)
p.add_argument("--mia_repo", default=os.path.expanduser("~/privacy/MIA_on_bulkRNAseq_CAMDA2026"))
a = p.parse_args()
sys.path.insert(0, a.mia_repo)
from mia import datasets as D, paths  # noqa: E402

genes = list(D.load_expression(a.dataset).columns)
splits = D.load_target_splits(a.dataset)
for rd in a.run_dir:
    run = os.path.basename(os.path.normpath(rd))
    for k in sorted(splits):
        fx = Path(rd) / f"synthetic_data_split_{k}.csv"
        if not fx.exists():
            continue
        ids = json.load(open(Path(rd) / f"split_{k}_ids.json"))
        if set(ids["train"]) != set(splits[k]["member_ids"]):
            raise SystemExit(f"{run} split {k}: training ids differ from the repo's member ids")
        out = paths.target_dir(a.dataset, f"ours@run={run}", k); out.mkdir(parents=True, exist_ok=True)
        X = pd.read_csv(fx)[genes]
        y = pd.read_csv(Path(rd) / f"synthetic_labels_split_{k}.csv").iloc[:, 0].astype(str)
        X.to_csv(out / "synthetic_data.csv", index=False)
        pd.DataFrame({"label": y.values}).to_csv(out / "synthetic_labels.csv", index=False)
        (out / "meta.json").write_text(json.dumps({
            "dataset": a.dataset, "generator": f"ours@run={run}", "split": k, "params": {"run": run}, "seed": None,
            "source": "external-ours", "origin": str(fx), "n_synthetic": int(len(X)),
            "sha1": hashlib.sha1(fx.read_bytes()).hexdigest()[:16]}, indent=2))
        print(f"imported {run} split {k}: {X.shape}", flush=True)
