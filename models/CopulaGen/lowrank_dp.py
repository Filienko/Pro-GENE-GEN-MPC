"""LowRankDP: DP generator that measures class structure only in a k-dimensional factor space.

DP-CopulaGen and Private-PGM measure a histogram per (class, gene) and reconstruct the class
marginals from the noisy bin counts.  Here the class mean and spread are measured directly as
L2-clipped sums, which is a more noise-efficient statistic, and the covariance is measured only in
k << d dimensions (zCDP, Gaussian mechanism, add/remove neighbours).  As in those methods, a record
is in one class, so class-wise measurements compose in parallel:

  n      class counts                                              sensitivity 1
  anchor per-gene centre / scale (a, b) used only to standardise:  public cohort, or a coarse pooled
         histogram (fixed bins of width 1 on [0, 25])              sensitivity sqrt(d)
  clip   L2 clipping thresholds = noisy quantiles of record norms on a fixed log grid   sensitivity 1
  mean   per-class sums of z = (x - a) / b in d dimensions, rows L2-clipped to C_z.  A record
         belongs to one class, so this costs the same as one pooled sum          sensitivity C_z
         The class shifts are then shrunk gene by gene (empirical Bayes, post-processing).
  var    per-class sums of |z - class mean|, rows L2-clipped to C_u              sensitivity C_u
  basis  W (k x d) for within-class covariance: PCA of the public cohort, or DP-PCA of the
         within-class Gram matrix                                                sensitivity C_g^2
  ccov   per class: sum of s s^T,  s = W (z - class mean), rows clipped to C_s   sensitivity C_s^2

Model: z | c = mean_c + W^T s + e,  s ~ N(0, S_c),  e ~ N(0, diag(v_c)),  x = a + b z.
Optional vote step (Private-Evolution style): candidates drawn from the DP model are grouped into
cells per class, private records vote for their nearest cell (sensitivity 1), and cells are
resampled by noisy votes.  This lets the factor distribution depart from a Gaussian.
"""
from typing import Optional

import numpy as np

from dp_copula import rho_from_eps, sym_noise, psd

SHARES_PUBLIC = dict(n=0.02, clip=0.03, mean=0.45, var=0.15, ccov=0.35)
SHARES_PRIVATE = dict(n=0.02, clip=0.03, anchor=0.08, mean=0.40, var=0.12, basis=0.15, ccov=0.20)
GRID = np.geomspace(1e-2, 1e5, 141)          # data-independent grid for the clipping quantiles


def rho_from_eps_gdp(eps, delta):
    """Budget for accounting="gdp", computed with the autodp library (Wang et al.), not by hand.
    Every release is a Gaussian mechanism whose sensitivity / sigma ratio is fixed in advance, so the
    composition is itself a Gaussian mechanism with (sens / sigma)^2 = sum_j (sens_j / sigma_j)^2 = 2 rho.
    autodp's ExactGaussianMechanism gives its exact eps at the given delta (analytic Gaussian mechanism,
    Balle & Wang 2018); the largest rho whose eps does not exceed the target is found by bisection.
    eps_spent() below re-derives the spent eps from the noise actually used, with autodp's own composition."""
    import warnings
    from autodp.mechanism_zoo import ExactGaussianMechanism
    if not np.isfinite(eps):
        return np.inf
    lo, hi = 1e-4, 60.0                          # mu = sens / sigma of the composed mechanism
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(60):
            mid = (lo + hi) / 2
            ok = ExactGaussianMechanism(sigma=1 / mid).get_approxDP(delta) <= eps
            lo, hi = (mid, hi) if ok else (lo, mid)
    return lo ** 2 / 2


def eps_spent(ledger, delta):
    """eps at the given delta for a list of (sensitivity, sigma) Gaussian releases, composed by autodp."""
    import warnings
    from autodp.mechanism_zoo import ExactGaussianMechanism
    from autodp.transformer_zoo import ComposeGaussian
    mechs = [ExactGaussianMechanism(sigma=sg / se) for se, sg in ledger if sg > 0]
    if len(mechs) < len(ledger) or not mechs:
        return float("inf")                      # a release without noise (ablation) has no guarantee
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(ComposeGaussian()(mechs, [1] * len(mechs)).get_approxDP(delta))


class LowRankDP:
    def __init__(self, epsilon=10.0, delta=1e-5, k=30, external: Optional[np.ndarray] = None,
                 external_y=None, shrink=0.3, clip_q=0.9, seed=0, vote_cells=0, vote_share=0.10,
                 noise_off=(), shares=None, between=False, accounting="zcdp"):
        self.eps, self.delta, self.k = epsilon, delta, k
        self.external, self.external_y = external, external_y
        self.shrink, self.clip_q = shrink, clip_q
        self.rng = np.random.default_rng(seed)
        self.vote_cells, self.noise_off = vote_cells, set(noise_off)
        self.between, self.accounting = between, accounting
        sh = dict(shares or (SHARES_PUBLIC if external is not None else SHARES_PRIVATE))
        if vote_cells:
            sh = {k_: v * (1 - vote_share) for k_, v in sh.items()}
            sh["vote"] = vote_share
        tot = sum(sh.values())
        self.shares = {k_: v / tot for k_, v in sh.items()}
        self.info = {}

    # ------------------------------------------------------------------ helpers
    def _sigma(self, name, sens):
        if name in self.noise_off or not np.isfinite(self.rho):
            self.ledger.append((float(sens), 0.0))
            return 0.0
        s = sens / np.sqrt(2 * self.rho * self.shares[name])
        self.info[f"sigma_{name}"] = float(s)
        self.ledger.append((float(sens), float(s)))
        return s

    def _clip_threshold(self, norms, parts):
        """noisy clip_q-quantile of the record norms on the fixed grid (one histogram, sensitivity 1)"""
        sg = self._sigma_clip(parts)
        h = np.histogram(norms, GRID)[0] + self.rng.standard_normal(len(GRID) - 1) * sg
        h = np.where(h > 4 * sg, h, 0.0)          # drop noise-only bins before accumulating
        if h.sum() <= 0:
            return float(np.sqrt(norms.shape[0]) * 0 + GRID[len(GRID) // 2])
        cum = np.cumsum(h)
        j = int(np.searchsorted(cum, self.clip_q * cum[-1]))
        return float(GRID[min(j + 1, len(GRID) - 1)])

    def _sigma_clip(self, parts):
        if "clip" in self.noise_off or not np.isfinite(self.rho):
            self.ledger.append((1.0, 0.0))
            return 0.0
        s = 1.0 / np.sqrt(2 * self.rho * self.shares["clip"] / parts)
        self.ledger.append((1.0, float(s)))
        return s

    @staticmethod
    def _clip_rows(A, C):
        return A * np.minimum(1.0, C / (np.linalg.norm(A, axis=1) + 1e-12))[:, None]

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        y = y.astype(str)
        self.classes = [str(c) for c in np.unique(y)]
        n, d = X.shape
        self.rho = (rho_from_eps_gdp if self.accounting == "gdp" else rho_from_eps)(self.eps, self.delta)
        rng = self.rng
        self.ledger = []                         # (sensitivity, sigma) of every noisy release
        public = self.external is not None
        # 1. class counts
        sn = self._sigma("n", 1.0)
        self.n_hat = {c: max(2.0, (y == c).sum() + rng.normal() * sn) for c in self.classes}
        n_tot = sum(self.n_hat.values())
        # 2. anchor
        if public:
            a, b = self.external.mean(0), self.external.std(0) + 1e-6
            self.lo, self.hi = self.external.min(0) - 1.0, self.external.max(0) + 1.0
        else:
            edges = np.arange(0, 26.0)
            idx = np.clip(X.astype(int), 0, 24)
            H = np.stack([(idx == j).sum(0) for j in range(25)]).astype(float)
            H += rng.standard_normal(H.shape) * self._sigma("anchor", np.sqrt(d))
            H = np.maximum(H, 0)
            jm = H.argmax(0)
            cen = edges[:-1] + 0.5
            w = H * (np.abs(np.arange(25)[:, None] - jm[None]) <= 3)
            sw = w.sum(0) + 1e-9
            a = (w * cen[:, None]).sum(0) / sw
            b = np.sqrt(np.maximum((w * (cen[:, None] - a) ** 2).sum(0) / sw - 1 / 12, 0.0))
            b = np.clip(b, 0.4, 2.5)
            self.lo, self.hi = np.zeros(d), np.full(d, 25.0)
        self.a, self.b = a, b
        Z = (X - a) / b
        n_clip = 3 if public else 4      # number of clipping histograms sharing the 'clip' budget
        C = len(self.classes)
        pri = {c: self.n_hat[c] / n_tot for c in self.classes}
        # 3. class sums in d dimensions.  A record is in one class, so all class sums together cost
        #    exactly what the pooled sum costs (sensitivity C_z).
        Cz = self._clip_threshold(np.linalg.norm(Z, axis=1), n_clip)
        Zcl = self._clip_rows(Z, Cz)
        s_mean = self._sigma("mean", Cz)
        sums = {c: Zcl[y == c].sum(0) + rng.standard_normal(d) * s_mean for c in self.classes}
        mu = sum(sums.values()) / n_tot
        D = np.stack([sums[c] / self.n_hat[c] - mu for c in self.classes])        # noisy class shifts
        e = np.array([s_mean / self.n_hat[c] for c in self.classes])               # their noise SD per gene
        # empirical-Bayes shrinkage per gene (post-processing): the between-class variance tau_g^2 of
        # gene g is estimated from all classes, and each noisy shift is multiplied by
        # tau_g^2 / (tau_g^2 + e_c^2).  Small classes fall back to the pooled mean where the gene
        # shows no class effect.
        self.mean = {}
        if s_mean > 0:
            wgt = (1 / e ** 2)[:, None]
            tau2 = np.maximum((wgt * (D ** 2 - e[:, None] ** 2)).sum(0) / wgt.sum(), 0.0)
            # a class with strong evidence of its own keeps its shift (non-negative garrote part)
            t2 = np.maximum(tau2[None], D ** 2 - 2 * e[:, None] ** 2)
            D = D * t2 / (t2 + e[:, None] ** 2)
        for i, c in enumerate(self.classes):
            self.mean[c] = mu + D[i]
        r = 0
        Mrow = np.stack([self.mean[c] for c in self.classes])[np.searchsorted(self.classes, y)]
        Zw = Z - Mrow                                                             # within-class residuals
        # 4. per-gene within-class spread from mean absolute deviations (row norm ~ ||z||, far smaller
        #    sensitivity than squared deviations), per class at the cost of one pooled sum
        U = np.abs(Zw)
        Cu = self._clip_threshold(np.linalg.norm(U, axis=1), n_clip)
        Ucl = self._clip_rows(U, Cu)
        s_var = self._sigma("var", Cu)
        vs = {c: Ucl[y == c].sum(0) + rng.standard_normal(d) * s_var for c in self.classes}
        q = sum(vs.values()) / n_tot
        if s_var > 0:  # shrink toward the across-gene median by the estimated signal / noise ratio
            med, nz = np.median(q), (s_var / n_tot) ** 2
            tq = max(np.var(q) - nz, 1e-6)
            q = med + (q - med) * tq / (tq + nz)
        q = np.maximum(q, 0.05)
        tau = 0.15 * np.median(q)
        self.var = {}
        for c in self.classes:
            w = tau ** 2 / (tau ** 2 + (s_var / self.n_hat[c]) ** 2) * self.n_hat[c] / (self.n_hat[c] + 30.0)
            qc = np.maximum(w * vs[c] / self.n_hat[c] + (1 - w) * q, 0.05)
            self.var[c] = (qc * np.sqrt(np.pi / 2)) ** 2
        # 5. basis for the within-class covariance
        if public:
            E = (self.external - a) / b
            g = self.external_y if self.external_y is not None else np.zeros(len(E))
            for name in np.unique(g):
                E[g == name] -= E[g == name].mean(0)
            _, _, Vt = np.linalg.svd(E, full_matrices=False)
            W = Vt[: self.k]
        else:
            # between-class directions: the row space of the released class means (post-processing, no
            # budget).  Samples also vary along these axes inside a class, so they are used as the first
            # basis vectors and removed from the residuals before DP-PCA, which then spends its budget
            # on what is left.
            Qb = np.zeros((0, d))
            if self.between and C > 1:
                Dm = np.stack([self.mean[c] for c in self.classes])
                Dm = Dm - Dm.mean(0)
                Qb = np.linalg.svd(Dm, full_matrices=False)[2][: min(C - 1, self.k - 1)]
            Zd = Zw - Zw @ Qb.T @ Qb
            Cg = self._clip_threshold(np.linalg.norm(Zd, axis=1), n_clip)
            Zg = self._clip_rows(Zd, Cg)
            sg = self._sigma("basis", Cg ** 2)
            ev, V = np.linalg.eigh(Zg.T @ Zg + sym_noise(rng, d, sg))
            ev, V = ev[::-1], V[:, ::-1]
            # keep the components whose eigenvalue clears the noise edge 2 sigma sqrt(d) (post-processing)
            k_max = self.k - len(Qb)
            k_eff = int(np.clip((ev > 2.1 * sg * np.sqrt(d)).sum(), 0 if len(Qb) else 1, k_max)) if sg > 0 else k_max
            W = np.linalg.qr(np.vstack([Qb, V[:, :k_eff].T]).T)[0].T
            self.info["k_eff"] = k_eff
            self.info["k_between"] = len(Qb)
        k = W.shape[0]
        # 6. class covariances in the factor space
        S = Zw @ W.T
        Cs = self._clip_threshold(np.linalg.norm(S, axis=1), n_clip)
        Scl = self._clip_rows(S, Cs)
        sc = self._sigma("ccov", Cs ** 2)
        cov = {c: (Scl[y == c].T @ Scl[y == c] + sym_noise(rng, k, sc)) / self.n_hat[c] for c in self.classes}
        pooled = psd(sum(pri[c] * cov[c] for c in self.classes), 1e-4)
        self.S, self.lam, self.resid = {}, {}, {}
        for c in self.classes:
            noise_rel = sc * np.sqrt(k) / self.n_hat[c] / max(np.trace(pooled) / k, 1e-9)
            lam = float(np.clip(max(self.shrink, k / self.n_hat[c], noise_rel), 0, 1))
            self.S[c] = psd((1 - lam) * cov[c] + lam * pooled, 1e-4)
            self.lam[c] = lam
            expl = np.einsum("ji,jk,ki->i", W, self.S[c], W)
            self.resid[c] = np.maximum(self.var[c] - expl, 0.05 * self.var[c])
        self.W, self.pri = W, pri
        self.info.update(Cz=Cz, Cu=Cu, Cs=Cs, k=k, r_between=r, rho=float(self.rho))
        # 7. optional vote step
        self.cells = None
        if self.vote_cells:
            self._vote(S, y)
        try:      # independent check of the guarantee from the noise actually used (autodp)
            self.info["eps_spent_autodp"] = eps_spent(self.ledger, self.delta)
        except ImportError:
            self.info["eps_spent_autodp"] = None
        return self

    def _vote(self, S_priv, y):
        from sklearn.cluster import KMeans
        sv = self._sigma("vote", 1.0)
        self.cells = {}
        for c in self.classes:
            J = int(max(2, min(self.vote_cells, self.n_hat[c] // 15)))
            cand = self._draw(c, 4000)
            km = KMeans(J, n_init=2, random_state=0).fit(cand)
            votes = np.bincount(km.predict(S_priv[y == c]), minlength=J) + self.rng.standard_normal(J) * sv
            w = np.maximum(votes, 0) + 1e-9
            self.cells[c] = (km, w / w.sum())

    def _draw(self, c, n):
        k = self.S[c].shape[0]
        return self.rng.standard_normal((n, k)) @ np.linalg.cholesky(self.S[c] + 1e-9 * np.eye(k)).T

    def _scores(self, c, n):
        if self.cells is None:
            return self._draw(c, n)
        km, w = self.cells[c]
        want = self.rng.multinomial(n, w)
        out = []
        for _ in range(60):
            cand = self._draw(c, 4 * n + 2000)
            lab = km.predict(cand)
            for j in np.where(want > 0)[0]:
                take = cand[lab == j][: want[j]]
                out.append(take)
                want[j] -= len(take)
            if want.sum() == 0:
                break
        out = np.vstack(out)
        if len(out) < n:
            out = np.vstack([out, self._draw(c, n - len(out))])
        return out[self.rng.permutation(len(out))[:n]]

    def sample(self, n_total=None):
        """n_total=None: number of rows = rounded sum of the noisy class counts (no exact count is used)"""
        cs = self.classes
        if n_total is None:
            n_total = int(round(sum(self.n_hat.values())))
        counts = {c: int(round(n_total * self.pri[c])) for c in cs}
        big = max(counts, key=counts.get)
        counts[big] += n_total - sum(counts.values())
        Xs, ys = [], []
        for c, nc in counts.items():
            if nc <= 0:
                continue
            Zs = self.mean[c] + self._scores(c, nc) @ self.W + self.rng.standard_normal((nc, len(self.a))) * np.sqrt(self.resid[c])
            Xs.append(np.clip(self.a + self.b * Zs, self.lo, self.hi))
            ys += [c] * nc
        X, yv = np.vstack(Xs), np.array(ys)
        p = self.rng.permutation(len(yv))
        return X[p], yv[p]
