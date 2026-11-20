#!/usr/bin/env python
"""Two-stage ablation with NoisyDiffusion (the CAMDA ND generator of MIA_on_bulkRNAseq_CAMDA2026,
mia/generators/nd.py, preset "target", used unchanged) as the stage-2 model, fold by fold on the CAMDA splits.

What ND is trained on (--train):
  real      the training members themselves (stage-1 ND = the CAMDA ND baseline)
  release   rows of a synthetic release only (--release_dir; e.g. DP-MVN eps30 -> formal eps30 by post-processing)
  mix       the hybrid prompt of run_tabpfn.py: 5-member cluster averages with dpcov spread (shape from --jitter_dir)
            + --n_release rows of --release_dir (871 + 470 = the "65%" mix)
  cluster   the dpcov cluster rows only (100%)

Run in the camda-mia env:
~/miniconda3/envs/camda-mia/bin/python run_nd_stage.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype \
    --splits_yaml $BRCA_SPLITS --train mix --release_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30 \
    --jitter_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30 --n_release 470 --folds 3 --out_dir $OUT/TABPFN/BRCA/nd_mix65_lrdp
"""
import argparse, ast, json, os, sys, time, types
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--train", required=True, choices=["real", "release", "mix", "cluster"])
p.add_argument("--release_dir", default=None); p.add_argument("--jitter_dir", default=None)
p.add_argument("--n_release", type=int, default=0, help="mix: rows of the release added (0 = all)")
p.add_argument("--k", type=int, default=5); p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=None, help="override the preset (smoke tests only)")
p.add_argument("--device", default=None)
p.add_argument("--mia_repo", default=os.path.expanduser("~/privacy/MIA_on_bulkRNAseq_CAMDA2026"))
p.add_argument("--out_dir", required=True)
a = p.parse_args()

sys.path.insert(0, a.mia_repo)
import mia  # noqa: E402,F401
pkg = types.ModuleType("mia.generators"); pkg.__path__ = [os.path.join(a.mia_repo, "mia", "generators")]
sys.modules["mia.generators"] = pkg
from mia.generators.nd import NDGenerator  # noqa: E402
import torch  # noqa: E402

src = open(os.path.join(HERE, "run_tabpfn.py")).read(); tree = ast.parse(src); ns = {"np": np}
exec("\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("mdav", "microagg_prompt")), ns)

os.makedirs(a.out_dir, exist_ok=True)
expr = load_expression(a.counts); labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]; genes = list(expr.columns); classes = sorted(labels.unique())
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")


def release(d, f):
    X = pd.read_csv(os.path.join(d, f"synthetic_data_split_{f}.csv"))[genes].values.astype(np.float64)
    y = np.array([classes.index(c) for c in pd.read_csv(os.path.join(d, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]])
    return X, y


cfg = os.path.join(a.out_dir, "config.json")
log = json.load(open(cfg)) if os.path.exists(cfg) else {"args": vars(a), "folds": {}}
for f in a.folds or sorted(splits):
    if os.path.exists(os.path.join(a.out_dir, f"synthetic_data_split_{f}.csv")):
        print(f"[ND] fold {f}: exists, skipped", flush=True); continue
    t0 = time.time(); rng_p = np.random.default_rng(a.seed + 1000 + f)
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    Xr = expr.loc[tr, genes].values.astype(np.float64); yr = np.array([classes.index(c) for c in labels.loc[tr]])
    if a.train == "real":
        X, y = Xr, yr
    elif a.train == "release":
        X, y = release(a.release_dir, f)
    else:
        X, y, _ = ns["microagg_prompt"](Xr, yr, a.k, "dpcov", rng_p, release(a.jitter_dir, f))
        if a.train == "mix":
            Xa, ya = release(a.release_dir, f)
            if 0 < a.n_release < len(Xa):
                keep = rng_p.choice(len(Xa), a.n_release, replace=False); Xa, ya = Xa[keep], ya[keep]
            X, y = np.vstack([X, Xa]), np.r_[y, ya]
    over = {"seed": a.seed + f, "device": dev, **({"epochs": a.epochs} if a.epochs else {})}
    g = NDGenerator.preset("target", **over).fit(X.astype(np.float32), y, len(classes))
    Xs, ys = g.sample(len(tr))
    save_synthetic(a.out_dir, f, Xs, np.array(classes, dtype=object)[ys], genes, a.label_col)
    json.dump({"train": tr, "test": te}, open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w"))
    pd.DataFrame({"sample_id": tr + te, "membership_label": [1] * len(tr) + [0] * len(te)}).to_csv(
        os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
    log["folds"][str(f)] = {"seconds": round(time.time() - t0, 1), "n_train_rows": int(len(y)), "n_synth": int(len(Xs)), "device": dev}
    json.dump(log, open(cfg, "w"), indent=1)
    print(f"[ND] fold {f}: trained on {len(y)} rows ({a.train}), released {len(Xs)} in {time.time() - t0:.0f}s", flush=True)
