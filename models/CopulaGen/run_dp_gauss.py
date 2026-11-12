#!/usr/bin/env python
"""Naive DP Gaussian baselines for the LowRankDP ablation (same accounting, same output layout).

  textbook : no standardisation; rows centred at the middle of the public range [0, 25] and bounded by the
             worst-case norm 12.5 sqrt(d); noisy class counts, class sums and FULL d x d second moments.
  adaptive : LowRankDP's standardisation (noisy fixed-bin histogram) and clipping threshold (noisy quantile),
             then noisy class sums and FULL d x d second moments.
  --cov class | pooled : one covariance per class, or one within-class covariance shared by all classes.

No shrinkage, no low-rank step.  The noisy second moment is projected to PSD and sampled from directly.

python run_dp_gauss.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS \
    --dp_epsilon 30 --mode adaptive --cov pooled --out_dir $OUT/LRDP/ablation/BRCA/adapt_pooled_eps30
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels, load_splits  # noqa: E402
from run_copulagen import write_fold  # noqa: E402
from lowrank_dp import LowRankDP, rho_from_eps_gdp, eps_spent  # noqa: E402
from dp_copula import sym_noise, psd  # noqa: E402


class DPGauss(LowRankDP):
    """reuses LowRankDP's noise calibration (_sigma), clipping (_clip_threshold, _clip_rows) and ledger"""

    def __init__(self, epsilon, delta=1e-5, mode="adaptive", cov="class", clip_q=0.75, seed=0):
        sh = (dict(n=0.02, mean=0.49, ccov=0.49) if mode == "textbook"
              else dict(n=0.02, clip=0.03, anchor=0.08, mean=0.45, ccov=0.42))
        super().__init__(epsilon=epsilon, delta=delta, clip_q=clip_q, seed=seed, shares=sh, accounting="gdp")
        self.mode, self.cov_mode = mode, cov

    def fit(self, X, y):
        y = y.astype(str)
        self.classes = [str(c) for c in np.unique(y)]
        d = X.shape[1]
        self.rho = rho_from_eps_gdp(self.eps, self.delta)
        rng, self.ledger = self.rng, []
        sn = self._sigma("n", 1.0)                           # all class counts together are one release
        self.n_hat = {c: max(2.0, (y == c).sum() + rng.normal() * sn) for c in self.classes}
        n_tot = sum(self.n_hat.values())
        self.pri = {c: self.n_hat[c] / n_tot for c in self.classes}
        if self.mode == "textbook":
            a, b = np.full(d, 12.5), np.ones(d)
            Z = np.clip(X, 0, 25) - a
            Cz = Cr = 12.5 * np.sqrt(d)                       # public worst-case bound
        else:                                                # same anchor as LowRankDP (private-only)
            idx = np.clip(X.astype(int), 0, 24)
            H = np.stack([(idx == j).sum(0) for j in range(25)]).astype(float)
            H = np.maximum(H + rng.standard_normal(H.shape) * self._sigma("anchor", np.sqrt(d)), 0)
            cen = np.arange(25) + 0.5
            w = H * (np.abs(np.arange(25)[:, None] - H.argmax(0)[None]) <= 3)
            sw = w.sum(0) + 1e-9
            a = (w * cen[:, None]).sum(0) / sw
            b = np.clip(np.sqrt(np.maximum((w * (cen[:, None] - a) ** 2).sum(0) / sw - 1 / 12, 0.0)), 0.4, 2.5)
            Z = (X - a) / b
            Cz = self._clip_threshold(np.linalg.norm(Z, axis=1), 2)
        self.a, self.b = a, b
        s_mean = self._sigma("mean", Cz)
        Zcl = self._clip_rows(Z, Cz)
        self.mean = {c: (Zcl[y == c].sum(0) + rng.standard_normal(d) * s_mean) / self.n_hat[c] for c in self.classes}
        Zw = Z - np.stack([self.mean[c] for c in self.classes])[np.searchsorted(self.classes, y)]
        if self.mode != "textbook":
            Cr = self._clip_threshold(np.linalg.norm(Zw, axis=1), 2)
        Rcl = self._clip_rows(Zw, Cr)
        sc = self._sigma("ccov", Cr ** 2)
        if self.cov_mode == "pooled":                        # one release
            P = psd((Rcl.T @ Rcl + sym_noise(rng, d, sc)) / n_tot, 1e-6)
            self.cov = {c: P for c in self.classes}
        else:                                                # one per class: parallel composition
            self.cov = {c: psd((Rcl[y == c].T @ Rcl[y == c] + sym_noise(rng, d, sc)) / self.n_hat[c], 1e-6)
                        for c in self.classes}
        self.info.update(Cz=float(Cz), Cr=float(Cr), rho=float(self.rho), eps_spent_autodp=eps_spent(self.ledger, self.delta))
        return self

    def sample(self):
        n_total = int(round(sum(self.n_hat.values())))
        Xs, ys, roots = [], [], {}
        for c in self.classes:
            nc = int(round(n_total * self.pri[c]))
            if nc <= 0:
                continue
            key = id(self.cov[c])
            if key not in roots:
                w, V = np.linalg.eigh(self.cov[c]); roots[key] = V * np.sqrt(np.maximum(w, 0))
            Zs = self.mean[c] + self.rng.standard_normal((nc, len(self.a))) @ roots[key].T
            Xs.append(np.clip(self.a + self.b * Zs, 0.0, 25.0)); ys += [c] * nc
        X, yv = np.vstack(Xs), np.array(ys)
        p = self.rng.permutation(len(yv))
        return X[p], yv[p]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
    p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
    p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dp_epsilon", type=float, default=10.0); p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--mode", default="adaptive", choices=["textbook", "adaptive"])
    p.add_argument("--cov", default="class", choices=["class", "pooled"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    expr = load_expression(a.counts)
    labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[labels.index]; genes = list(expr.columns)
    splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
    log = {"args": vars(a), "folds": {}}
    for f in a.folds or sorted(splits):
        tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
        t0 = time.time()
        g = DPGauss(a.dp_epsilon, a.delta, a.mode, a.cov, seed=a.seed + f).fit(expr.loc[tr].values, labels.loc[tr].values)
        Xs, ys = g.sample()
        write_fold(a.out_dir, f, Xs, ys, genes, a.label_col, tr, te)
        log["folds"][f] = {"seconds": round(time.time() - t0, 1), "info": g.info, "shares": g.shares}
        print(f"[DPGauss {a.mode}/{a.cov}] fold {f}: {Xs.shape} in {time.time() - t0:.1f}s  {({k: round(v, 3) for k, v in g.info.items()})}", flush=True)
    json.dump(log, open(os.path.join(a.out_dir, "config.json"), "w"), indent=2, default=str)
