#!/usr/bin/env python
"""TabPFN-PE: Private Evolution (Lin et al. 2024; tabular variant Tab-PE as in ~/dptab) with TabPFN-3.5 as the
variation operator.  No public data.  The private rows never enter a TabPFN prompt; they are touched only by
nearest-neighbour vote histograms, which are released with Gaussian noise.

  0. Start population: a LowRankDP release at eps_init (n rows), expanded to N = pop_mult * n rows by TabPFN
     variations (post-processing).  Coordinates: the top-K principal directions of that DP release
     (post-processing too); a row = centre + coords @ P.T + residual, and only coords are evolved
     (the residual rides along with its parent).
  1. For t = 1..T: every private row votes for its nearest same-class candidate (coords space).  Histogram,
     add/remove sensitivity 1, + N(0, sigma^2), minus threshold, clipped at 0.
       t < T : draw N/2 parents with prob. ~ noisy votes; their variations (TabPFN, below) join them.
       t = T : release n rows drawn without replacement with prob. ~ noisy votes.
  Variation: each coordinate of a parent is masked with prob. r_t (decaying r0 -> rT); masked coordinates are
  resampled one at a time (Gibbs) from TabPFN-3.5's predictive distribution given the parent's other coords
  and its class, the prompt being the current (already private) population.

Privacy (Gaussian DP, add/remove): mu_total^2 = mu_init^2 + T / sigma^2; sigma is set so that the total is
eps_total at delta.  --sigma0 runs the noiseless vote (NON-PRIVATE ceiling, never report as DP).
Runs in camda_sota.  Resumable per fold.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen")); sys.path.insert(0, HERE)
from factor_core import load_expression, load_labels, load_splits, save_synthetic  # noqa: E402
from lowrank_dp import rho_from_eps_gdp, eps_spent  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
p.add_argument("--out_dir", required=True)
p.add_argument("--init_dir", required=True, help="LowRankDP run used as the start population")
p.add_argument("--init_eps", type=float, required=True); p.add_argument("--total_eps", type=float, required=True)
p.add_argument("--delta", type=float, default=1e-5)
p.add_argument("--T", type=int, default=10); p.add_argument("--pop_mult", type=int, default=2)
p.add_argument("--K", type=int, default=30); p.add_argument("--r0", type=float, default=0.5); p.add_argument("--rT", type=float, default=0.1)
p.add_argument("--threshold", type=float, default=None, help="vote threshold H (default: sigma)")
p.add_argument("--sigma0", action="store_true", help="noiseless votes: NON-PRIVATE ceiling")
p.add_argument("--n_estimators", type=int, default=4); p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
import torch  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

# ---- accounting -------------------------------------------------------------------------------------------
mu_init, mu_tot = np.sqrt(2 * rho_from_eps_gdp(a.init_eps, a.delta)), np.sqrt(2 * rho_from_eps_gdp(a.total_eps, a.delta))
assert mu_tot > mu_init, "total eps must exceed the init eps"
sigma = 0.0 if a.sigma0 else float(np.sqrt(a.T) / np.sqrt(mu_tot ** 2 - mu_init ** 2))
thr = sigma if a.threshold is None else a.threshold
eps_check = None if a.sigma0 else eps_spent([(1.0, 1 / mu_init)] + [(1.0, sigma)] * a.T, a.delta)
print(f"[PE] mu_init {mu_init:.3f}  mu_total {mu_tot:.3f}  vote sigma {sigma:.3f}  threshold {thr:.3f}  "
      f"eps check {eps_check}", flush=True)

expr = load_expression(a.counts)
labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
expr = expr.loc[labels.index]; genes = list(expr.columns); classes = sorted(labels.unique())
splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)


def variation(Cp, yp, C_ctx, y_ctx, r, rng):
    """Gibbs-resample a random subset (rate r) of each parent's coordinates from TabPFN's predictive
    distribution, prompt = (C_ctx, y_ctx)."""
    K = Cp.shape[1]; Cv = Cp.copy()
    M = rng.random(Cp.shape) < r
    M[np.arange(len(M)), rng.integers(0, K, len(M))] = True          # at least one coordinate per row
    for j in rng.permutation(K):
        rows = np.where(M[:, j])[0]
        if not len(rows):
            continue
        oth = [i for i in range(K) if i != j]
        Xf = np.column_stack([y_ctx, C_ctx[:, oth]]).astype(np.float32)
        reg = TabPFNRegressor(n_estimators=a.n_estimators, device=a.device, random_state=int(rng.integers(1 << 30)),
                              ignore_pretraining_limits=True, categorical_features_indices=[0])
        reg.fit(Xf, C_ctx[:, j].astype(np.float32))
        pred = reg.predict(np.column_stack([yp[rows], Cv[rows][:, oth]]).astype(np.float32), output_type="full")
        Cv[rows, j] = pred["criterion"].sample(torch.as_tensor(pred["logits"]).float().cpu(), t=1.0).cpu().numpy()
    return Cv


def votes(Cpriv, ypriv, Cpop, ypop):
    """Each private row votes for its nearest candidate of the same class."""
    H = np.zeros(len(Cpop))
    for c in np.unique(ypriv):
        cand = np.where(ypop == c)[0]
        if not len(cand):
            continue                                                 # a class the population lost: vote dropped
        pr = Cpriv[ypriv == c]
        d = (pr ** 2).sum(1)[:, None] - 2 * pr @ Cpop[cand].T + (Cpop[cand] ** 2).sum(1)[None]
        np.add.at(H, cand[d.argmin(1)], 1)
    return H


os.makedirs(a.out_dir, exist_ok=True)
cfg_path = os.path.join(a.out_dir, "config.json")
log = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {
    "args": vars(a), "mu_init": mu_init, "mu_total": mu_tot, "sigma": sigma, "eps_total_check": eps_check, "folds": {}}
for f in a.folds or sorted(splits):
    if os.path.exists(os.path.join(a.out_dir, f"synthetic_data_split_{f}.csv")):
        print(f"[PE] fold {f}: exists, skipped", flush=True); continue
    t0 = time.time(); rng = np.random.default_rng(a.seed + f); torch.manual_seed(a.seed + f)
    tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
    Xpriv = expr.loc[tr, genes].values.astype(np.float64); ypriv = np.array([classes.index(c) for c in labels.loc[tr]])
    Xi = pd.read_csv(os.path.join(a.init_dir, f"synthetic_data_split_{f}.csv"))[genes].values.astype(np.float64)
    yi = np.array([classes.index(c) for c in pd.read_csv(os.path.join(a.init_dir, f"synthetic_labels_split_{f}.csv")).iloc[:, 0]])
    m0 = Xi.mean(0); P = np.linalg.svd(Xi - m0, full_matrices=False)[2][: a.K].T     # basis of the DP release
    Ci = (Xi - m0) @ P; Ri = Xi - m0 - Ci @ P.T
    Cpriv = (Xpriv - m0) @ P
    n = len(tr); N = a.pop_mult * n
    # start population: the DP release plus TabPFN variations of it (no private data involved)
    par = rng.integers(0, len(Ci), N - len(Ci))
    C = np.vstack([Ci, variation(Ci[par], yi[par], Ci, yi, a.r0, rng)]); y = np.r_[yi, yi[par]]; R = np.vstack([Ri, Ri[par]])
    hist = []
    for t in range(1, a.T + 1):
        H = votes(Cpriv, ypriv, C, y)
        Hn = H + sigma * rng.standard_normal(len(H)); w = np.clip(Hn - thr, 0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        hist.append({"t": t, "noisy_votes_pos": int((w > 0).sum()), "top_share": float(np.sort(w)[-10:].sum() / w.sum()),
                     "nn_dist": float(np.sqrt(((Cpriv[:, None] - C[None]) ** 2).sum(-1).min(1)).mean()) if a.sigma0 else None})
        if t == a.T:
            k = min(n, int((w > 0).sum()))
            idx = rng.choice(len(C), k, replace=False, p=w / w.sum())
            if k < n:
                idx = np.r_[idx, rng.choice(len(C), n - k, replace=True, p=w / w.sum())]
            C, y, R = C[idx], y[idx], R[idx]
            break
        r = a.rT + (a.r0 - a.rT) * (1 - (t - 1) / max(a.T - 2, 1))
        sel = rng.choice(len(C), N // 2, replace=True, p=w / w.sum())
        kids = variation(C[sel], y[sel], C, y, max(r, a.rT), rng)
        C, y, R = np.vstack([C[sel], kids]), np.r_[y[sel], y[sel]], np.vstack([R[sel], R[sel]])
        print(f"[PE] fold {f} t={t}: positive noisy bins {hist[-1]['noisy_votes_pos']}, mask rate {r:.2f}, "
              f"{time.time() - t0:.0f}s", flush=True)
    Xs = (m0 + C @ P.T + R).astype(np.float32)
    save_synthetic(a.out_dir, f, Xs, np.array(classes, dtype=object)[y], genes, a.label_col)
    json.dump({"train": tr, "test": te}, open(os.path.join(a.out_dir, f"split_{f}_ids.json"), "w"))
    pd.DataFrame({"sample_id": tr + te, "membership_label": [1] * len(tr) + [0] * len(te)}).to_csv(
        os.path.join(a.out_dir, f"membership_labels_split_{f}.csv"), index=False)
    log["folds"][str(f)] = {"seconds": round(time.time() - t0, 1), "iterations": hist}
    json.dump(log, open(cfg_path, "w"), indent=2)
    print(f"[PE] fold {f}: {Xs.shape} in {time.time() - t0:.0f}s", flush=True)
