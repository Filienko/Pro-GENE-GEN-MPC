#!/usr/bin/env python
"""DP-MVN: differentially private multivariate-normal synthetic data, per class, with a low-rank covariance.

This is the generator behind every "DP-MVN" result (it is lowrank_dp.LowRankDP with the settings that were actually
used -- private-only, run_lrdp.py --v2 -- and nothing else).  It reproduces those releases bit for bit.

Model, per class c, on standardized values z = (x - a) / b:

    z | c  =  mean_c  +  W^T s  +  e,      s ~ N(0, S_c)  (k factors),   e ~ N(0, diag(resid_c))

Privacy: (epsilon, delta)-DP for add/remove-one neighbours.  Every release below is a Gaussian mechanism on a
clipped sum; a record belongs to one class, so the per-class sums of one step cost the same as one pooled sum.
The budget rho (exact Gaussian-DP conversion, autodp) is split over the steps by SHARES.

  1. n      class counts                                           sensitivity 1
  2. anchor per-gene histogram, 25 unit bins on [0, 25]  -> centre a and scale b of each gene      sensitivity sqrt(d)
  3. clip   L2 clipping thresholds = noisy 75 % quantile of the row norms, on a fixed log grid (one histogram
            per clipped quantity, 4 of them, sharing this budget)                                    sensitivity 1
  4. mean   per-class sums of z, rows clipped to C_z;  class shifts then shrunk per gene (empirical Bayes)
                                                                                                    sensitivity C_z
  5. var    per-class sums of |z - class mean| (mean absolute deviation -> per-gene spread)          sensitivity C_u
  6. basis  W (k x d): DP-PCA of the pooled within-class Gram matrix; keep the components whose noisy
            eigenvalue clears 2.1 x the noise edge sigma sqrt(d)                                     sensitivity C_g^2
  7. ccov   per class: sum of s s^T with s = W (z - class mean), rows clipped to C_s; shrunk toward the pooled
            factor covariance                                                                        sensitivity C_s^2

Everything else (shrinkage, eigen-decompositions, residual variances, sampling) is post-processing.
Sampling: s ~ N(0, S_c), e ~ N(0, diag(resid_c)), x = clip(a + b (mean_c + W^T s + e), 0, 25); the number of rows
comes from the noisy class counts.

Input values are expected on [0, 25] (VST expression is).  For other data pass --bounds_json: each column is
mapped from its public [lo, hi] to [0, 24] and back at the output.

python dp_mvn.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS \
    --dp_epsilon 30 --out_dir $OUT/EPSWEEP/BRCA/dpmvn_eps30
"""
import argparse, json, os, sys, time, warnings

import numpy as np

SHARES = dict(n=0.02, clip=0.03, anchor=0.08, mean=0.45, var=0.12, basis=0.20, ccov=0.10)
GRID = np.geomspace(1e-2, 1e5, 141)          # data-independent grid for the clipping quantiles
N_CLIP = 4                                   # clipping histograms sharing the "clip" budget (C_z, C_u, C_g, C_s)


# ------------------------------------------------------------------------------------------------ accounting
def rho_from_eps(eps, delta):
    """Largest budget rho = mu^2 / 2 whose composed Gaussian mechanism (sensitivity/sigma = mu) has eps(delta) <= eps,
    with autodp's exact Gaussian mechanism (Balle & Wang 2018), by bisection on mu."""
    from autodp.mechanism_zoo import ExactGaussianMechanism
    if not np.isfinite(eps):
        return np.inf
    lo, hi = 1e-4, (60.0 if eps <= 2000 else 400.0)      # mu 60 = eps ~2055
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ExactGaussianMechanism(sigma=1 / mid).get_approxDP(delta) <= eps else (lo, mid)
    return lo ** 2 / 2


def eps_spent(ledger, delta):
    """eps at delta of a list of (sensitivity, sigma) Gaussian releases, composed by autodp (independent check)."""
    from autodp.mechanism_zoo import ExactGaussianMechanism
    from autodp.transformer_zoo import ComposeGaussian
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mechs = [ExactGaussianMechanism(sigma=sg / se) for se, sg in ledger]
        return float(ComposeGaussian()(mechs, [1] * len(mechs)).get_approxDP(delta))


# ------------------------------------------------------------------------------------------------ helpers
def sym_noise(rng, k, sigma):
    """symmetric k x k Gaussian noise (upper triangle drawn, mirrored)"""
    A = rng.standard_normal((k, k)) * sigma
    return np.triu(A) + np.triu(A, 1).T


def psd(C, floor=0.0):
    w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.maximum(w, floor)) @ V.T


def clip_rows(A, C):
    return A * np.minimum(1.0, C / (np.linalg.norm(A, axis=1) + 1e-12))[:, None]


class DPMVN:
    def __init__(self, epsilon=30.0, delta=1e-5, k=30, shrink=0.3, clip_q=0.75, kthr=2.1, seed=0, rho=None):
        """rho: give the budget directly instead of (epsilon, delta) (used by dp_mvn_copula.py to split one budget)."""
        self.eps, self.delta, self.k, self.shrink, self.clip_q, self.kthr = epsilon, delta, k, shrink, clip_q, kthr
        self.rho_given = rho
        self.rng = np.random.default_rng(seed)
        self.shares = {k_: v / sum(SHARES.values()) for k_, v in SHARES.items()}
        self.info = {}

    # one Gaussian release: sigma for sensitivity sens under this step's share of the budget
    def _sigma(self, name, sens):
        s = sens / np.sqrt(2 * self.rho * self.shares[name])
        self.info[f"sigma_{name}"] = float(s); self.ledger.append((float(sens), float(s)))
        return s

    def _clip_threshold(self, norms):
        """noisy clip_q-quantile of the row norms on the fixed grid (one histogram, sensitivity 1)"""
        sg = 1.0 / np.sqrt(2 * self.rho * self.shares["clip"] / N_CLIP); self.ledger.append((1.0, float(sg)))
        h = np.histogram(norms, GRID)[0] + self.rng.standard_normal(len(GRID) - 1) * sg
        h = np.where(h > 4 * sg, h, 0.0)                     # drop noise-only bins before accumulating
        if h.sum() <= 0:
            return float(GRID[len(GRID) // 2])
        cum = np.cumsum(h)
        return float(GRID[min(int(np.searchsorted(cum, self.clip_q * cum[-1])) + 1, len(GRID) - 1)])

    def fit(self, X, y):
        rng = self.rng
        y = np.asarray(y).astype(str); self.classes = [str(c) for c in np.unique(y)]
        n, d = X.shape
        self.rho = self.rho_given if self.rho_given is not None else rho_from_eps(self.eps, self.delta)
        self.ledger = []
        # 1. class counts
        sn = self._sigma("n", 1.0)
        self.n_hat = {c: max(2.0, (y == c).sum() + rng.normal() * sn) for c in self.classes}
        n_tot = sum(self.n_hat.values()); pri = {c: self.n_hat[c] / n_tot for c in self.classes}
        # 2. anchor: centre a and scale b of each gene from a noisy 25-bin histogram (window of +-3 bins around the mode)
        edges = np.arange(0, 26.0); idx = np.clip(X.astype(int), 0, 24)
        H = np.stack([(idx == j).sum(0) for j in range(25)]).astype(float)
        H = np.maximum(H + rng.standard_normal(H.shape) * self._sigma("anchor", np.sqrt(d)), 0)
        jm = H.argmax(0); cen = edges[:-1] + 0.5
        w = H * (np.abs(np.arange(25)[:, None] - jm[None]) <= 3); sw = w.sum(0) + 1e-9
        a = (w * cen[:, None]).sum(0) / sw
        b = np.clip(np.sqrt(np.maximum((w * (cen[:, None] - a) ** 2).sum(0) / sw - 1 / 12, 0.0)), 0.4, 2.5)
        self.a, self.b, self.lo, self.hi = a, b, np.zeros(d), np.full(d, 25.0)
        Z = (X - a) / b
        # 4. class means: clipped class sums of z, then per-gene empirical-Bayes shrinkage of the class shifts
        Cz = self._clip_threshold(np.linalg.norm(Z, axis=1)); Zcl = clip_rows(Z, Cz)
        s_mean = self._sigma("mean", Cz)
        sums = {c: Zcl[y == c].sum(0) + rng.standard_normal(d) * s_mean for c in self.classes}
        mu = sum(sums.values()) / n_tot
        D = np.stack([sums[c] / self.n_hat[c] - mu for c in self.classes])        # noisy class shifts
        e = np.array([s_mean / self.n_hat[c] for c in self.classes])               # their noise SD per gene
        wgt = (1 / e ** 2)[:, None]
        tau2 = np.maximum((wgt * (D ** 2 - e[:, None] ** 2)).sum(0) / wgt.sum(), 0.0)   # between-class variance per gene
        t2 = np.maximum(tau2[None], D ** 2 - 2 * e[:, None] ** 2)                       # a strong own shift is kept
        D = D * t2 / (t2 + e[:, None] ** 2)
        self.mean = {c: mu + D[i] for i, c in enumerate(self.classes)}
        Zw = Z - np.stack([self.mean[c] for c in self.classes])[np.searchsorted(self.classes, y)]   # within-class residuals
        # 5. per-gene within-class spread from mean absolute deviations, shrunk to the across-gene median and to pooled
        U = np.abs(Zw); Cu = self._clip_threshold(np.linalg.norm(U, axis=1)); Ucl = clip_rows(U, Cu)
        s_var = self._sigma("var", Cu)
        vs = {c: Ucl[y == c].sum(0) + rng.standard_normal(d) * s_var for c in self.classes}
        q = sum(vs.values()) / n_tot
        med, nz = np.median(q), (s_var / n_tot) ** 2
        tq = max(np.var(q) - nz, 1e-6); q = np.maximum(med + (q - med) * tq / (tq + nz), 0.05)
        tau = 0.15 * np.median(q); self.var = {}
        for c in self.classes:
            wc = tau ** 2 / (tau ** 2 + (s_var / self.n_hat[c]) ** 2) * self.n_hat[c] / (self.n_hat[c] + 30.0)
            qc = np.maximum(wc * vs[c] / self.n_hat[c] + (1 - wc) * q, 0.05)
            self.var[c] = (qc * np.sqrt(np.pi / 2)) ** 2                          # MAD -> variance (Gaussian)
        # 6. basis: DP-PCA of the pooled within-class Gram matrix
        Cg = self._clip_threshold(np.linalg.norm(Zw, axis=1)); Zg = clip_rows(Zw, Cg)
        sg = self._sigma("basis", Cg ** 2)
        ev, V = np.linalg.eigh(Zg.T @ Zg + sym_noise(rng, d, sg)); ev, V = ev[::-1], V[:, ::-1]
        k_eff = int(np.clip((ev > self.kthr * sg * np.sqrt(d)).sum(), 1, self.k))
        W = np.linalg.qr(V[:, :k_eff])[0].T; k = W.shape[0]
        self.info.update(k_eff=k_eff)
        # 7. class factor covariances, shrunk toward the pooled one (more for small classes / more noise)
        S = Zw @ W.T; Cs = self._clip_threshold(np.linalg.norm(S, axis=1)); Scl = clip_rows(S, Cs)
        sc = self._sigma("ccov", Cs ** 2)
        cov = {c: (Scl[y == c].T @ Scl[y == c] + sym_noise(rng, k, sc)) / self.n_hat[c] for c in self.classes}
        pooled = psd(sum(pri[c] * cov[c] for c in self.classes), 1e-4)
        self.S, self.resid = {}, {}
        for c in self.classes:
            noise_rel = sc * np.sqrt(k) / self.n_hat[c] / max(np.trace(pooled) / k, 1e-9)
            lam = float(np.clip(max(self.shrink, k / self.n_hat[c], noise_rel), 0, 1))
            self.S[c] = psd((1 - lam) * cov[c] + lam * pooled, 1e-4)
            expl = np.einsum("ji,jk,ki->i", W, self.S[c], W)                         # variance the factors explain
            self.resid[c] = np.maximum(self.var[c] - expl, 0.05 * self.var[c])
        self.W, self.pri = W, pri
        self.info.update(Cz=Cz, Cu=Cu, Cg=Cg, Cs=Cs, rho=float(self.rho), eps_spent_autodp=eps_spent(self.ledger, self.delta))
        return self

    def sample(self, n_total=None):
        """n_total=None: rows = rounded sum of the noisy class counts (no exact count is used)"""
        if n_total is None:
            n_total = int(round(sum(self.n_hat.values())))
        counts = {c: int(round(n_total * self.pri[c])) for c in self.classes}
        big = max(counts, key=counts.get); counts[big] += n_total - sum(counts.values())
        Xs, ys = [], []
        for c, nc in counts.items():
            if nc <= 0:
                continue
            k = self.S[c].shape[0]
            s = self.rng.standard_normal((nc, k)) @ np.linalg.cholesky(self.S[c] + 1e-9 * np.eye(k)).T
            Zs = self.mean[c] + s @ self.W + self.rng.standard_normal((nc, len(self.a))) * np.sqrt(self.resid[c])
            Xs.append(np.clip(self.a + self.b * Zs, self.lo, self.hi)); ys += [c] * nc
        X, yv = np.vstack(Xs), np.array(ys)
        p = self.rng.permutation(len(yv))
        return X[p], yv[p]


# ------------------------------------------------------------------------------------------------ CLI (fold by fold)
def public_bounds(path, cols):
    """per-column public [lo, hi] (adult_prep.py meta.json: numeric bounds; categorical = [0, n_levels - 1])"""
    mj = json.load(open(path))
    bd = {**mj["numeric"], **{c: [0.0, len(v) - 1.0] for c, v in mj["categorical"].items()}}
    return np.array([bd[g][0] for g in cols]), np.array([bd[g][1] for g in cols])


def cli(make_generator, description):
    """shared fold-by-fold driver (also used by dp_mvn_copula.py): CAMDA splits in, run_lrdp.py output layout out"""
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
    from factor_core import load_expression, load_labels, load_splits
    from run_copulagen import write_fold
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
    p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
    p.add_argument("--splits_yaml", required=True); p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dp_epsilon", type=float, default=30.0); p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--k", type=int, default=30, help="max number of factors")
    p.add_argument("--seed", type=int, default=0, help="fold seed = seed + fold; -1 = OS entropy (use this for a real release)")
    p.add_argument("--bounds_json", default=None, help="public per-column domain for data not on [0, 25] (adult_prep.py meta.json)")
    p.add_argument("--copula_bins", type=int, default=100, help="(dp_mvn_copula.py) bins of the private marginal histogram")
    p.add_argument("--marg_share", type=float, default=0.08, help="(dp_mvn_copula.py) budget share of the marginals")
    p.add_argument("--copula_thr", type=float, default=3.0, help="(dp_mvn_copula.py) per-feature selection threshold")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    expr = load_expression(a.counts); labels = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[labels.index]; genes = list(expr.columns)
    LO = HI = None
    if a.bounds_json:
        LO, HI = public_bounds(a.bounds_json, genes); expr = 24.0 * (expr - LO) / (HI - LO)
    splits = load_splits(a.splits_yaml, expr.index, labels, 5, 42)
    log = {"args": vars(a), "folds": {}}
    for f in a.folds or sorted(splits):
        tr = [s for s in splits[f][0] if s in expr.index]; te = [s for s in splits[f][1] if s in expr.index]
        t0 = time.time()
        g = make_generator(a, None if a.seed < 0 else a.seed + f).fit(expr.loc[tr].values, labels.loc[tr].values)
        Xs, ys = g.sample()
        if LO is not None:
            Xs = np.clip(LO + Xs * (HI - LO) / 24.0, LO, HI)
        write_fold(a.out_dir, f, Xs, ys, genes, a.label_col, tr, te)
        log["folds"][f] = {"seconds": round(time.time() - t0, 1), "info": g.info}
        print(f"fold {f}: {Xs.shape} in {time.time() - t0:.1f}s  eps spent {g.info['eps_spent_autodp']:.3f}  "
              f"k_eff {g.info.get('k_eff')}", flush=True)
        json.dump(log, open(os.path.join(a.out_dir, "config.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    cli(lambda a, seed: DPMVN(epsilon=a.dp_epsilon, delta=a.delta, k=a.k, seed=seed), "DP-MVN (low-rank DP multivariate normal)")
