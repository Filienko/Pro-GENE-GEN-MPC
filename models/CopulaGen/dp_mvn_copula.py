#!/usr/bin/env python
"""DP-MVN with Gaussian-copula marginals: a wrapper around dp_mvn.DPMVN (which it calls unchanged).

DP-MVN models each feature as Gaussian.  For skewed, zero-inflated, bounded or categorical features this wrapper
moves the data to normal scores first and maps the synthetic data back afterwards:

    fit:     x  --F (private marginal)-->  u = F(x)  --Phi^-1-->  z  --(12.5 + 2 z)-->  x'  -->  DPMVN.fit(x')
    sample:  DPMVN.sample()  -->  x'  -->  z = (x' - 12.5) / 2  --Phi-->  u  --F^-1-->  x

(12.5 + 2 z is a fixed public map that puts normal scores on the [0, 25] scale DP-MVN works on.)

Budget: one (epsilon, delta) is split.  The marginals get --marg_share of rho, DPMVN the rest; the total is checked
with autodp (composition of every Gaussian release of both parts).

  marginals  per-feature histogram, B equal bins on [0, 25] (one count per feature per record)    sensitivity sqrt(d)

Post-processing of that histogram (no further access to the data):
  * bins below 2 sigma of the noise are set to 0 (otherwise every empty bin carries noise mass);
  * the histogram is blended with the Gaussian of its own mean / SD, weight lam = sigma sqrt(B) / n (noise-to-signal);
  * per feature, the copula is used only if the released histogram departs from that Gaussian by more than the noise
    explains:  sum_b (H_b - n G_b)^2 / (n G_b + sigma^2 + 1)  >  B + thr sqrt(2 B);  other features go to DPMVN as is;
  * F is piecewise linear (uniform within a bin).  Values sitting exactly on a bin edge (category codes, zeros) get a
    random position inside their bin (randomized probability-integral transform), so ties do not collapse to one z.

python dp_mvn_copula.py --counts $ADULT_TSV --annot $ADULT_ANN --label_col Subtype --splits_yaml $ADULT_SPLITS \
    --bounds_json $ADULT_META --dp_epsilon 30 --copula_bins 100 --out_dir $OUT/ADULT/lrdpc_eps30
"""
import numpy as np
from scipy import special

from dp_mvn import DPMVN, rho_from_eps, eps_spent, cli

LOC, SCALE = 12.5, 2.0          # public map of normal scores onto DP-MVN's [0, 25] scale (+-6 SD stays inside)


class DPMVNCopula:
    def __init__(self, epsilon=30.0, delta=1e-5, bins=100, marg_share=0.08, thr=3.0, seed=0, **dpmvn_kw):
        self.eps, self.delta, self.B, self.share, self.thr = epsilon, delta, int(bins), marg_share, thr
        self.seed, self.kw = seed, dpmvn_kw
        self.rng = np.random.default_rng(None if seed is None else seed + 5555)    # own stream: DPMVN's draws unchanged
        self.info = {}

    def fit(self, X, y):
        n, d = X.shape; B = self.B
        rho = rho_from_eps(self.eps, self.delta)
        # private marginals: one B-bin histogram per feature on [0, 25]
        e = np.linspace(0.0, 25.0, B + 1)
        ib = np.clip((X / 25.0 * B).astype(int), 0, B - 1)
        H = np.stack([(ib == j).sum(0) for j in range(B)]).astype(float)
        sm = np.sqrt(d) / np.sqrt(2 * rho * self.share)
        H += self.rng.standard_normal(H.shape) * sm
        # post-processing: drop noise-only bins, blend with the histogram's own Gaussian by noise-to-signal ratio
        Hh = np.where(H > 2 * sm, H, 0.0); tot = Hh.sum(0)
        P = np.where(tot > 0, Hh / np.maximum(tot, 1e-12), 1.0 / B)
        c = (e[:-1] + e[1:]) / 2
        m = (P * c[:, None]).sum(0); s = np.sqrt(np.maximum((P * (c[:, None] - m) ** 2).sum(0), 1e-4))
        G = special.ndtr((e[1:, None] - m) / s) - special.ndtr((e[:-1, None] - m) / s)
        G = np.maximum(G, 1e-12); G /= G.sum(0)
        n_est = float(max(np.median(tot), 1.0))
        lam = float(np.clip(sm * np.sqrt(B) / n_est, 0.02, 1.0))
        P = np.maximum((1 - lam) * P + lam * G, 1e-9); P /= P.sum(0)
        stat = (((H - n_est * G) ** 2) / (n_est * G + sm ** 2 + 1.0)).sum(0)
        self.use = stat > B + self.thr * np.sqrt(2.0 * B)
        self.e, self.P, self.F = e, P, np.vstack([np.zeros(d), np.cumsum(P, 0)])
        # forward map, then DP-MVN on the rest of the budget
        self.inner = DPMVN(rho=rho * (1 - self.share), delta=self.delta, seed=self.seed, **self.kw).fit(self._to_dpmvn(X), y)
        self.info = {**self.inner.info, "copula_features": int(self.use.sum()), "copula_lam": lam, "sigma_marg": float(sm),
                     "eps_spent_autodp": eps_spent(self.inner.ledger + [(float(np.sqrt(d)), float(sm))], self.delta)}
        return self

    def _to_dpmvn(self, X):
        e, P, F, B = self.e, self.P, self.F, self.B
        j = np.clip(np.searchsorted(e, X, side="right") - 1, 0, B - 1)
        w = np.clip((X - e[j]) / (e[j + 1] - e[j]), 0.0, 1.0)
        w = np.where(w <= 1e-9, self.rng.uniform(size=X.shape), w)               # ties on a bin edge: spread over the bin
        cols = np.arange(X.shape[1])[None, :]
        z = special.ndtri(np.clip(F[j, cols] + w * P[j, cols], 1e-5, 1 - 1e-5))
        return np.where(self.use[None, :], LOC + SCALE * z, X)

    def sample(self, n_total=None):
        Xp, ys = self.inner.sample(n_total)
        X = Xp.copy(); e, P, F, B = self.e, self.P, self.F, self.B
        u = np.clip(special.ndtr((Xp - LOC) / SCALE), 1e-9, 1 - 1e-9)
        for g in np.where(self.use)[0]:
            j = np.clip(np.searchsorted(F[:, g], u[:, g], side="right") - 1, 0, B - 1)
            X[:, g] = e[j] + np.clip((u[:, g] - F[j, g]) / P[j, g], 0.0, 1.0) * (e[j + 1] - e[j])
        return X, ys


if __name__ == "__main__":
    cli(lambda a, seed: DPMVNCopula(epsilon=a.dp_epsilon, delta=a.delta, bins=a.copula_bins, marg_share=a.marg_share,
                                    thr=a.copula_thr, seed=seed, k=a.k),
        "DP-MVN with Gaussian-copula marginals (wrapper around dp_mvn.py)")
