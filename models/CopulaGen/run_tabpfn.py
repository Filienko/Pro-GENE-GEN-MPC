#!/usr/bin/env python
"""Generate with the TabPFN generator of MIA_on_bulkRNAseq_CAMDA2026 (mia/generators/tabpfn.py, used unchanged),
fold by fold, in the same output layout as run_lrdp.py so eval_suite.py / redteam_attacks.py read it directly.

Runs in the camda_sota environment (tabpfn 9.1.0 + tabpfn-extensions 0.6.3).  Resumable: a fold whose
synthetic_data_split_k.csv exists is skipped; config.json is rewritten after every fold.

~/miniconda3/envs/camda_sota/bin/python run_tabpfn.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype \
    --splits_yaml $BRCA_SPLITS --model_version v2 --out_dir $OUT/TABPFN/BRCA/tabpfn_v2

Differentially private variants (the real rows never enter the prompt):
  --context_dir D   the context is D's synthetic fold k (e.g. a LowRankDP eps run) instead of the real training rows;
                    the output is post-processing of D, so it carries exactly D's (eps, delta).
  --public_counts/--public_annot   add rows of a public auxiliary pool (here: non-breast TCGA tumours, patients
                    disjoint from BRCA) to the context under an extra class "__public__"; rows generated with that
                    class are discarded, so the release is n rows of the private classes.  No privacy cost.
"""
import argparse, importlib.util, json, os, sys, time, types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--out_dir", required=True)
p.add_argument("--mia_repo", default=os.path.expanduser("~/privacy/MIA_on_bulkRNAseq_CAMDA2026"))
p.add_argument("--model_version", default="default", help='"default" = TabPFN-3.5 (licence token needed), "v2" = open Nature-paper weights')
p.add_argument("--preprocess", default="none", help='"none" = gene by gene (repo default); "standard+pca:64" = 64 PC scores')
p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--n_permutations", type=int, default=3)
p.add_argument("--context_dir", default=None, help="DP synthetic run to use as the context (fold-matched)")
p.add_argument("--public_counts", default=None); p.add_argument("--public_annot", default=None)
p.add_argument("--public_label_col", default="cancer_type"); p.add_argument("--public_exclude", nargs="*", default=["Breast"])
p.add_argument("--n_public", type=int, default=600)
p.add_argument("--n_genes", type=int, default=None, help="smoke test: first n genes only")
# empirical (not formally DP) defences: weaken each member's pull on the prompt
p.add_argument("--microagg", type=int, default=0, help="k > 1: prompt = MDAV cluster means of k members per subtype")
p.add_argument("--restore", default="cov", choices=["none", "cov", "swap"],
               help="with --microagg: none = the means only; cov = k jittered copies per mean, jitter ~ N(0, pooled "
                    "within-cluster covariance, Ledoit-Wolf), so the prompt keeps the members' spread; swap = each mean plus the "
                    "real within-cluster deviations of a different random cluster of the same class (non-Gaussian shape)")
p.add_argument("--add_context_dir", default=None, help="append this (DP) release's fold rows to the prompt")
p.add_argument("--add_context_n", type=int, default=0, help="> 0: append only this many of those rows (random)")
p.add_argument("--shards", type=int, default=1, help="K > 1: K disjoint member shards, each prompts n/K rows")
p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

# import mia.generators.tabpfn without running mia/generators/__init__.py (which pulls in every other generator)
sys.path.insert(0, a.mia_repo)
import mia  # noqa: E402,F401
pkg = types.ModuleType("mia.generators"); pkg.__path__ = [os.path.join(a.mia_repo, "mia", "generators")]
sys.modules["mia.generators"] = pkg
from mia.generators.tabpfn import TabPFNGenerator  # noqa: E402

os.makedirs(a.out_dir, exist_ok=True)
expr = load_expression(a.counts)
labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]
genes = list(expr.columns)[: a.n_genes] if a.n_genes else list(expr.columns)
classes = sorted(labels.unique())
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
public = None
if a.public_counts:
    pe = load_expression(a.public_counts)
    pl = load_labels(a.public_annot, pe.index, a.sample_col, a.public_label_col)
    public = pe.loc[pl.index[~pl.isin(a.public_exclude)], genes]
    assert not set(public.index) & set(expr.index), "public pool overlaps the private data"


def mdav(X, k):
    """Maximum-distance-to-average-vector microaggregation: groups of k (last one up to 2k-1)."""
    idx, groups = np.arange(len(X)), []
    def take(r, idx):
        d = ((X[idx] - X[r]) ** 2).sum(1); g = idx[np.argsort(d)[:k]]; return g, np.setdiff1d(idx, g)
    while len(idx) >= 3 * k:
        c = X[idx].mean(0); r = idx[((X[idx] - c) ** 2).sum(1).argmax()]
        g, idx = take(r, idx); groups.append(g)
        s_ = idx[((X[idx] - X[r]) ** 2).sum(1).argmax()]
        g, idx = take(s_, idx); groups.append(g)
    if len(idx) >= 2 * k:
        c = X[idx].mean(0); r = idx[((X[idx] - c) ** 2).sum(1).argmax()]
        g, idx = take(r, idx); groups.append(g)
    if len(idx):
        groups.append(idx)
    return groups


def microagg_prompt(X, y, k, restore, rng):
    """Within each class: MDAV groups -> means; optionally k jittered copies per mean with the pooled within-group
    covariance (Ledoit-Wolf), so no prompt row is any single member."""
    from sklearn.covariance import LedoitWolf
    sd = X.std(0) + 1e-8; Z = X / sd
    means, ym, devs, sizes = [], [], [], []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        for g in mdav(Z[ic], k) if len(ic) >= 2 * k else [np.arange(len(ic))]:
            rows = X[ic[g]]; means.append(rows.mean(0)); ym.append(c); devs.append(rows - rows.mean(0)); sizes.append(len(g))
    M, ym = np.array(means), np.array(ym)
    if restore == "none":
        return M, ym, sizes
    if restore == "swap":                       # deviations moved to another cluster's mean: real shape, no real row
        out, oy, cls = [], [], np.array(ym)
        for i in range(len(M)):
            same = np.where((cls == cls[i]) & (np.arange(len(M)) != i))[0]
            j = rng.choice(same) if len(same) else i
            dv = devs[j] * np.sqrt(sizes[j] / max(sizes[j] - 1, 1))
            out.append(M[i] + dv[rng.choice(len(dv), sizes[i], replace=len(dv) < sizes[i])]); oy += [cls[i]] * sizes[i]
        return np.vstack(out), np.array(oy), sizes
    D = np.vstack(devs); D *= np.sqrt(np.repeat(np.array(sizes) / np.maximum(np.array(sizes) - 1, 1), sizes))[:, None]
    L = np.linalg.cholesky(LedoitWolf().fit(D).covariance_ + 1e-6 * np.eye(X.shape[1]))
    reps = np.array(sizes)
    Xp = np.repeat(M, reps, 0) + rng.standard_normal((reps.sum(), X.shape[1])) @ L.T
    return Xp, np.repeat(ym, reps), sizes


cfg_path = os.path.join(a.out_dir, "config.json")
log = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"args": vars(a), "classes": classes, "folds": {}}
for f in a.folds or sorted(splits):
    if os.path.exists(os.path.join(a.out_dir, f"synthetic_data_split_{f}.csv")):
        print(f"[TabPFN] fold {f}: exists, skipped", flush=True); continue
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    if a.context_dir:                                           # DP: a private synthetic release is the prompt
        Xc = pd.read_csv(os.path.join(a.context_dir, f"synthetic_data_split_{f}.csv"))[genes].values
        y = np.array([classes.index(c) for c in pd.read_csv(os.path.join(a.context_dir, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]])
    else:
        Xc = expr.loc[tr, genes].values; y = np.array([classes.index(c) for c in labels.loc[tr]])
    nc, n_pub = len(classes), 0
    if public is not None:                                      # public pool as an extra class
        rng = np.random.default_rng(a.seed + f)
        P = public.values[rng.choice(len(public), min(a.n_public, len(public)), replace=False)]
        Xc = np.vstack([Xc, P]); y = np.r_[y, np.full(len(P), nc)]; nc += 1; n_pub = len(P)
    t0 = time.time(); rng_p = np.random.default_rng(a.seed + 1000 + f); extra = {}
    if a.microagg > 1:
        Xc, y, sizes = microagg_prompt(Xc.astype(np.float64), y, a.microagg, a.restore, rng_p)
        extra = {"microagg_k": a.microagg, "restore": a.restore, "n_groups": len(sizes), "max_group": int(max(sizes))}
    if a.add_context_dir:                                       # hybrid prompt: + a DP release's rows
        Xa = pd.read_csv(os.path.join(a.add_context_dir, f"synthetic_data_split_{f}.csv"))[genes].values
        ya = np.array([classes.index(c) for c in pd.read_csv(os.path.join(a.add_context_dir, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]])
        if 0 < a.add_context_n < len(Xa):
            keep = rng_p.choice(len(Xa), a.add_context_n, replace=False); Xa, ya = Xa[keep], ya[keep]
        Xc, y = np.vstack([Xc, Xa]), np.r_[y, ya]; extra.update(added_context=a.add_context_dir, n_added=int(len(Xa)))
    n = len(tr)
    if a.shards > 1:                                            # disjoint, class-stratified shards
        sh = np.empty(len(y), int)
        for c in np.unique(y):
            ic = rng_p.permutation(np.where(y == c)[0]); sh[ic] = np.arange(len(ic)) % a.shards
        jobs = [(Xc[sh == j], y[sh == j], n // a.shards + (j < n % a.shards)) for j in range(a.shards)]
        extra["shards"] = a.shards
    else:
        jobs = [(Xc, y, n)]
    keep_X, keep_y, n_gen = [], [], 0
    for jn, (Xj, yj, nj) in enumerate(jobs):
        g = TabPFNGenerator(seed=a.seed + f + 7919 * jn, device=a.device, preprocess=a.preprocess, temperature=a.temperature,
                            n_permutations=a.n_permutations, model_version=a.model_version)
        g.fit(Xj, yj, nc)
        got = 0
        while got < nj:
            m = int(np.ceil((nj - got) * len(yj) / max(1, (yj < len(classes)).sum()) * 1.1))
            Xg, yg = g.sample(m); n_gen += m; ok = yg < len(classes)
            take = min(int(ok.sum()), nj - got)
            keep_X.append(Xg[ok][:take]); keep_y.append(yg[ok][:take]); got += take
    Xs, ys = np.vstack(keep_X), np.concatenate(keep_y)
    save_synthetic(a.out_dir, f, Xs, np.array(classes, dtype=object)[ys], genes, a.label_col)
    json.dump({"train": tr, "test": te}, open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w"))
    pd.DataFrame({"sample_id": tr + te, "membership_label": [1] * len(tr) + [0] * len(te)}).to_csv(
        os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
    log["folds"][str(f)] = {"seconds": round(time.time() - t0, 1), "params": g.resolved_params(),
                            "context": a.context_dir or "real training rows", "n_context": int(len(y)),
                            "n_public": n_pub, "n_generated": n_gen, **extra}
    json.dump(log, open(cfg_path, "w"), indent=2, default=str)
    print(f"[TabPFN] fold {f}: {Xs.shape} in {time.time() - t0:.0f}s", flush=True)
