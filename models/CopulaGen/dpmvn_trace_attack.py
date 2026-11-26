#!/usr/bin/env python
"""DP-MVN tracing attack: a membership-inference attack tailored to dp_mvn.py.

DP-MVN releases (through its synthetic rows) per-class means, per-gene spreads and a low-rank covariance, each a
Gaussian mechanism on a clipped sum.  A member x of class c moves the released class mean by about w x / n_c (w = its
clipping factor), i.e. by 1/n_c of its own deviation, in every one of the d = 978 genes at once.  That is the
"trace amount" setting of Homer et al. (2008) / Dwork et al. (2015): a single gene says nothing, but the inner product
of the candidate's deviation with the release's deviation, summed over d genes, has a member-vs-non-member shift of
order sqrt(d / n_c) standard deviations.  Distance-based attacks (MahalaMIA, DCR) and low-dimensional ones
(DOMIAS in PCA-20, 1-/2-way marginals) do not use this.

Threat model (same information as kNN-LR / MAMA-MIA here): the synthetic release, the candidate pool (members +
non-members, with subtype labels) and the DP-MVN algorithm with its epsilon (read from the release's config.json;
--eps overrides).  No shadow models, no non-member data beyond the pool.

Per class c (z = per-gene standardised scale; every estimate below comes from the release or the pool):
  m      synthetic class mean;  r_i = leave-one-out mean of the pool's class-c candidates (after clipping)
  u_i    = clip(x_i) - r_i      (clip = the mechanism's row clipping, centre/scale/threshold recomputed on the pool)
  v_i    = m - r_i
  Sigma  = noise covariance of v:  release sampling (factor model of the synthetic rows, k from the spectrum gap,
           i.e. the release's own k_eff) x (1 + reference share) + DP noise of the mean step (sigma from epsilon)
  coef_g = weight of a member in the released class mean of gene g: 1/n + s_cg (1/n_c - 1/n), s_cg = the mechanism's
           empirical-Bayes shrinkage of class shifts, recomputed on the synthetic class shifts
  t_i    = u_i' A v_i / sqrt(u_i' A u_i)  with A = Sigma^-1 diag(coef n_c)     (per-record calibrated, N(0,1) if out)
  delta_i= u_i' A diag(coef) u_i / sqrt(...)                                     (its expected shift if in)
  score  = delta_i t_i - delta_i^2 / 2       (Gaussian log-likelihood ratio, comparable across classes and records)
Second moment (trace_var): the same with |z - class mean| (the mechanism's MAD step) against the synthetic per-gene
spread, diagonal noise model, shrinkage weight of the mechanism.  trace = trace_mean + trace_var.

python dpmvn_trace_attack.py --counts $BRCA_TSV --annot $BRCA_ANN --syn_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30 \
    --folds 1 2 3 4 5 --out_csv $OUT/TRACE/trace.csv
"""
import argparse, json, os, sys, warnings

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression, load_labels  # noqa: E402

try:
    from dp_mvn import rho_from_eps, SHARES  # the generator's own accounting
except ImportError:  # same as dp_mvn.py (exact Gaussian-DP conversion, autodp; budget shares)
    SHARES = dict(n=0.02, clip=0.03, anchor=0.08, mean=0.45, var=0.12, basis=0.20, ccov=0.10)

    def rho_from_eps(eps, delta):
        from autodp.mechanism_zoo import ExactGaussianMechanism
        if not np.isfinite(eps):
            return np.inf
        lo, hi = 1e-4, (60.0 if eps <= 2000 else 400.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if ExactGaussianMechanism(sigma=1 / mid).get_approxDP(delta) <= eps else (lo, mid)
        return lo ** 2 / 2

SH = {k: v / sum(SHARES.values()) for k, v in SHARES.items()}


# ------------------------------------------------------------------------------------------------ pieces
def anchor(X):
    """dp_mvn's centre a / scale b of each gene (+-3 bins around the mode of a 25-bin histogram), noiseless, on X"""
    idx = np.clip(X.astype(int), 0, 24); H = np.stack([(idx == j).sum(0) for j in range(25)]).astype(float)
    jm = H.argmax(0); cen = np.arange(25) + 0.5
    w = H * (np.abs(np.arange(25)[:, None] - jm[None]) <= 3); sw = w.sum(0) + 1e-9
    a = (w * cen[:, None]).sum(0) / sw
    b = np.clip(np.sqrt(np.maximum((w * (cen[:, None] - a) ** 2).sum(0) / sw - 1 / 12, 0.0)), 0.4, 2.5)
    return a, b


def gap_rank(s, kmax, win=8, flat=1.04):
    """number of spikes (the release's k_eff): the factor eigenvalues sit above a dense, nearly flat noise bulk, so
    k = first index from which `win` consecutive eigenvalue ratios s_j / s_j+1 all stay below `flat`.
    Fallback: position of the largest ratio among the first kmax."""
    r = s[:kmax + win] / np.maximum(s[1:kmax + win + 1], 1e-12)
    for j in range(1, kmax + 1):
        if (r[j:j + win] < flat).all():
            return j
    return int(np.argmax(r[:kmax])) + 1


def woodbury(L, D):
    """V -> V (L L' + diag D)^-1"""
    Di = 1.0 / D
    if L.shape[1] == 0:
        return lambda V: V * Di
    M = np.linalg.inv(np.eye(L.shape[1]) + (L.T * Di) @ L)
    return lambda V: V * Di - ((V * Di) @ L) @ M @ (L.T * Di)


def loo_mean(A):
    return (A.sum(0)[None] - A) / (len(A) - 1)


def tstat(T, nv, sig):
    """per-record calibrated statistic t = T / sqrt(nv) (N(0,1) for a non-member under the noise model) and its
    predicted shift for a member, delta = sig / sqrt(nv)"""
    nv = np.maximum(nv, 1e-12)
    return T / np.sqrt(nv), sig / np.sqrt(nv)


def self_calibrate(t, dl, pi):
    """Fix a mis-specified noise model with the pool alone: t ~ N(g delta, s^2) for members (share pi of the pool,
    known from the release size), N(0, s^2) otherwise.  Moments over the pool give
    g = mean(t) / (pi mean(delta)),  s^2 = var(t) - pi g^2 E[delta^2] + (pi g E[delta])^2.  Returns (t / s, g delta / s)."""
    g = max(t.mean() / max(pi * dl.mean(), 1e-12), 0.0)
    s2 = t.var() - pi * g ** 2 * (dl ** 2).mean() + (pi * g * dl.mean()) ** 2
    s = np.sqrt(max(s2, 0.25))
    return t / s, g * dl / s


def log_lr(t, dl):
    return dl * t - dl ** 2 / 2


def trace_scores(X, y, S, yS, eps, delta=1e-5, q=None, kmax=60, use_dp=True, use_clip=True, use_shrink=True,
                 sigma_mean=None, sigma_var=None, n_member=None):
    """per-candidate scores (dict of arrays) for candidates X (labels y) against the release S (labels yS).
    sigma_mean / sigma_var: the mechanism's noise SD of the mean / spread step (default: from eps and dp_mvn's shares).
    n_member: training-set size when the release has a different number of rows (default: the release size)."""
    N, d = X.shape; n_syn = len(S)
    n_mem = float(n_member or n_syn); scale = n_syn / n_mem          # synthetic rows per training row
    classes = [c for c in np.unique(y) if (yS == c).sum() >= 3]
    rho = rho_from_eps(eps, delta) if (use_dp and eps is not None and np.isfinite(eps)) else np.inf
    # the mechanism's standardisation and row clipping, recomputed on the pool (80 % of it are the members)
    a, b = anchor(X); Z = (X - a) / b
    nz = np.linalg.norm(Z, axis=1); Cz = float(np.quantile(nz, 0.75))
    w = np.minimum(1.0, Cz / nz) if use_clip else np.ones(N)
    Xw = a + b * Z * w[:, None]
    s_mean = Cz / np.sqrt(2 * rho * SH["mean"]) if np.isfinite(rho) else 0.0           # z units, per class sum
    if sigma_mean is not None:
        s_mean = float(sigma_mean) if use_dp else 0.0
    ns = {c: int((yS == c).sum()) for c in classes}                                       # synthetic rows per class
    nc = {c: ns[c] / scale for c in classes}                                              # ~ the noisy class counts
    Sm = {c: S[yS == c].mean(0) for c in classes}
    # shrinkage of the class shifts (dp_mvn step 4), on the synthetic shifts minus their sampling variance
    mu = S.mean(0)
    Dz = np.stack([(Sm[c] - mu) / b for c in classes])
    samp = np.stack([S[yS == c].var(0) / ns[c] for c in classes]) / b ** 2
    e2 = np.array([(s_mean / nc[c]) ** 2 for c in classes])[:, None]
    if use_shrink and s_mean > 0:
        wg = 1 / e2[:, 0]
        tau2 = np.maximum((wg[:, None] * (Dz ** 2 - samp - e2)).sum(0) / wg.sum(), 0.0)
        t2 = np.maximum(tau2[None], Dz ** 2 - samp - 2 * e2)
        shr = t2 / (t2 + e2)
    else:
        shr = np.ones((len(classes), d))
    # factor model of the release: shared basis from the pooled within-class synthetic residuals (dp_mvn's W),
    # per-class factor covariance and residual variances
    R = S.copy()
    for c in classes:
        R[yS == c] -= Sm[c]
    keep = np.isin(yS, classes); R = R[keep]; yR = yS[keep]
    sd = R.std(0) + 1e-6; Rt = R / sd
    U_, s_, Vt = np.linalg.svd(Rt, full_matrices=False)
    k = gap_rank(s_ ** 2 / len(Rt), min(kmax, len(Rt) // 4)) if q is None else int(q)
    V = Vt[:k].T
    Fp = Rt @ V; Sp = Fp.T @ Fp / len(Rt)
    kap = max(N - n_mem, 0) / N                         # pool reference: its non-member share adds noise to r
    out = {"trace_T": np.zeros(N)}
    t1, d1, t2, d2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    # second-moment step constants (dp_mvn step 5)
    zw_all = Z - np.stack([(Sm[c] - a) / b if c in Sm else np.zeros(d) for c in y])
    Uabs = np.abs(zw_all); nu = np.linalg.norm(Uabs, axis=1); Cu = float(np.quantile(nu, 0.75))
    wu = np.minimum(1.0, Cu / nu) if use_clip else np.ones(N)
    s_var = Cu / np.sqrt(2 * rho * SH["var"]) if np.isfinite(rho) else 0.0
    if sigma_var is not None:
        s_var = float(sigma_var) if use_dp else 0.0
    q_syn = {c: np.sqrt(2 / np.pi) * S[yS == c].std(0) / b for c in classes}            # released MAD, z units
    tau = 0.15 * np.median(np.mean([q_syn[c] for c in classes], 0))
    for ci, c in enumerate(classes):
        i = np.where(y == c)[0]
        if len(i) < 3:
            continue
        n_c, m_c = nc[c], ns[c]; Rc = Rt[yR == c]
        # ---- first moment (sampling noise of the synthetic mean: m_c rows; reference noise: kap / n_c)
        Fc = Rc @ V; lam = float(np.clip(max(0.3, k / m_c), 0, 1))
        Sc = (1 - lam) * Fc.T @ Fc / m_c + lam * Sp
        res = np.maximum(((Rc - Fc @ V.T) ** 2).mean(0), 1e-4)
        ev, P = np.linalg.eigh(Sc); L = V @ (P * np.sqrt(np.maximum(ev, 0)))             # Sigma_c = L L' + diag(res)
        Lx = L * np.sqrt(1 / m_c + kap / n_c)
        Dx = res * (1 / m_c + kap / n_c) + (s_mean * b / (n_c * sd)) ** 2
        A = woodbury(Lx, Dx)
        coef = 1 / n_mem + shr[ci] * (1 / n_c - 1 / n_mem)
        Xc = Xw[i] / sd; r = loo_mean(Xc); u = Xc - r; v = Sm[c][None] / sd - r
        uc = u * (coef * n_c)[None]; Au = A(uc)
        T = (Au * v).sum(1)
        t1[i], d1[i] = tstat(T, (Au * uc).sum(1), (Au * u * coef[None]).sum(1))
        # ---- second moment (MAD per gene), diagonal noise model
        wc = tau ** 2 / (tau ** 2 + (s_var / n_c) ** 2) * n_c / (n_c + 30.0)
        Uc = Uabs[i] * wu[i][:, None]; ru = loo_mean(Uc); ru_pool = loo_mean(Uabs * wu[:, None])[i]
        u2 = Uc - ru; v2 = q_syn[c][None] - (wc * ru + (1 - wc) * ru_pool)
        coef2 = wc / n_c + (1 - wc) / n_mem
        # Var of v2: synthetic SD -> MAD estimate (q^2 / 2m) + pool reference (MAD of |z|, kap share) + DP noise
        V2 = q_syn[c] ** 2 / (2 * m_c) + kap * q_syn[c] ** 2 * (np.pi / 2 - 1) / n_c + (wc * s_var / n_c) ** 2
        V2 = np.maximum(V2, 1e-8)
        T2 = (u2 * v2 / V2).sum(1)
        e2_ = (u2 ** 2 / V2).sum(1)
        t2[i], d2[i] = T2 / np.sqrt(e2_), np.sqrt(e2_) * coef2
        out["trace_T"][i] = (T - np.median(T)) / (T.std() + 1e-12)
    pi = min(n_mem / N, 0.99)
    out["trace_mean_raw"] = log_lr(t1, d1); out["trace_var_raw"] = log_lr(t2, d2)
    out["trace_mean"] = log_lr(*self_calibrate(t1, d1, pi)); out["trace_var"] = log_lr(*self_calibrate(t2, d2, pi))
    out["trace"] = out["trace_mean"] + out["trace_var"]
    out["k"] = k
    return out


# ------------------------------------------------------------------------------------------------ CLI
def tpr_at(y, s, f):
    fp, tp, _ = roc_curve(y, s); return float(np.interp(f, fp, tp))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--counts", required=True); p.add_argument("--annot", required=True)
    p.add_argument("--sample_col", default="samplesID"); p.add_argument("--label_col", default="Subtype")
    p.add_argument("--syn_dir", nargs="+", required=True); p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--eps", type=float, default=None, help="override the release's epsilon (default: its config.json)")
    p.add_argument("--q", type=int, default=None, help="number of factors (default: spectrum gap of the release)")
    p.add_argument("--no_dp", action="store_true", help="ablation: ignore the DP noise")
    p.add_argument("--no_clip", action="store_true", help="ablation: ignore the row clipping")
    p.add_argument("--no_shrink", action="store_true", help="ablation: ignore the class-shift shrinkage")
    p.add_argument("--sigma_from_config", action="store_true",
                   help="take the mean / spread noise SD from the release's config.json (sigma_mean, sigma_var) instead of "
                        "from epsilon (for releases whose budget split is not dp_mvn's, e.g. dpmvn_ablation.py)")
    p.add_argument("--n_member_from_ids", action="store_true",
                   help="training-set size = len(train ids) (for releases with more or fewer rows than the training set)")
    p.add_argument("--out_csv", required=True)
    a = p.parse_args()
    expr = load_expression(a.counts); lab = load_labels(a.annot, expr.index, a.sample_col, a.label_col)
    expr = expr.loc[lab.index]; genes = list(expr.columns)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    rows = []
    for sd_ in a.syn_dir:
        run = os.path.basename(os.path.normpath(sd_))
        eps = a.eps
        if eps is None:
            cfg = os.path.join(sd_, "config.json")
            eps = float(json.load(open(cfg))["args"].get("dp_epsilon", np.inf)) if os.path.exists(cfg) else np.inf
        for f in a.folds:
            fs = os.path.join(sd_, f"synthetic_data_split_{f}.csv")
            if not os.path.exists(fs):
                continue
            ids = json.load(open(os.path.join(sd_, f"split_{f}_ids.json")))
            cand = ids["train"] + ids["test"]; mem = np.r_[np.ones(len(ids["train"])), np.zeros(len(ids["test"]))]
            X = expr.loc[cand, genes].values; y = lab.loc[cand].values.astype(str)
            S = pd.read_csv(fs)[genes].values
            yS = pd.read_csv(os.path.join(sd_, f"synthetic_labels_split_{f}.csv")).iloc[:, 0].values.astype(str)
            kw = {}
            if a.sigma_from_config:
                info = json.load(open(os.path.join(sd_, "config.json")))["folds"][str(f)]["info"]
                kw.update(sigma_mean=info["sigma_mean"], sigma_var=info["sigma_var"])
            if a.n_member_from_ids:
                kw["n_member"] = len(ids["train"])
            o = trace_scores(X, y, S, yS, eps, q=a.q, use_dp=not a.no_dp, use_clip=not a.no_clip, use_shrink=not a.no_shrink, **kw)
            names = ["trace", "trace_mean", "trace_var", "trace_mean_raw", "trace_var_raw", "trace_T"]
            for nm in names:
                rows.append({"run": run, "fold": f, "eps": eps, "k": o["k"], "score": nm, "auc": roc_auc_score(mem, o[nm]),
                             "tpr@0.01": tpr_at(mem, o[nm], 0.01), "tpr@0.1": tpr_at(mem, o[nm], 0.1)})
            pd.DataFrame({"sample_id": cand, **{nm: o[nm] for nm in names}, "member": mem}).to_csv(
                a.out_csv.replace(".csv", f"_{run}_f{f}_scores.csv"), index=False)
            print(run, f, f"k={o['k']}", " ".join(f"{r['score']}={r['auc']:.3f}/{r['tpr@0.01']:.3f}" for r in rows[-len(names):]),
                  flush=True)
    R = pd.DataFrame(rows); R.to_csv(a.out_csv, index=False)
    if len(R):
        g = R.groupby(["run", "score"]).agg(auc=("auc", "mean"), worst=("auc", "max"), tpr1=("tpr@0.01", "mean"),
                                            n=("fold", "nunique"))
        print(g.round(3).to_string())


if __name__ == "__main__":
    main()
