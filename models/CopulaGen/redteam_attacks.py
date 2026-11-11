#!/usr/bin/env python
"""PPML-Huskies red-team attacks against a generator in the CAMDA run-dir layout.

Membership test set = the whole real pool (as in the red team's internal evaluation):
members are the fold's training ids, non-members its held-out ids.

  simple   MahalaMIA (MahalanobisNoAux / MahalanobisAux, red-team code) and MAMA-MIA with one-way and
           two-way gene x label marginals, reference = whole labelled pool (as in the paper).
  base     K base shadows: the generator under attack is trained on K random 80/20 splits of the pool
           and its synthetic output is cached (shared by `melomia` and `aware`).
  melomia  MeLoMIA with synth-shadows: the red team's NoisyDiffusion shadow model
           (synth_shadow_assets) is trained on every base-shadow synthetic set, T x N loss features
           are extracted for every pool sample, a meta-classifier ensemble is trained, and a proxy
           trained on the target's synthetic data scores the candidates.  No Optuna: classifiers use
           the red team's default hyper-parameters, all timesteps, summary features.
  aware    the same synth-shadow recipe with a CopulaGen proxy: per-sample likelihood terms of a
           CopulaGen refitted on the synthetic data are the features.

--gen copula : target = existing run dir (--target_dir), base shadows = CopulaGen with --gen_args
--gen nd     : positive control; target and base shadows are the red team's diffusion model trained
               on real data (targets are generated into --target_dir if missing)
"""
import argparse, glob, json, os, pickle, sys, time, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "FactorGen"))
from factor_core import load_expression  # noqa: E402
from eval_suite import auc, tpr_at, nn_dist  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("stage", choices=["simple", "base", "melomia", "aware"])
p.add_argument("--counts", required=True)
p.add_argument("--annot", required=True)
p.add_argument("--label_col", default="Subtype")
p.add_argument("--target_dir", required=True)
p.add_argument("--work", required=True, help="cache dir for shadows / features")
p.add_argument("--gen", default="copula", choices=["copula", "nd", "lrdp", "mvn"])
p.add_argument("--mvn_noise", type=float, default=0.7)
p.add_argument("--gen_eps", type=float, default=30.0)
p.add_argument("--gen_ext", default=None, help="public cohort TSV for --gen lrdp")
p.add_argument("--mama_bins", type=int, nargs="+", default=[4, 10])
p.add_argument("--K", type=int, default=15)
p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--n_noise", type=int, default=300)
p.add_argument("--device", default="cuda:0")
p.add_argument("--reference", default=None, help="disjoint reference TSV (genes x samples)")
p.add_argument("--redteam", default=os.path.expanduser("~/privacy/PPML-Huskies-Red-Team/src"))
p.add_argument("--mama_repo", default=os.path.expanduser("~/privacy/SyntheticData_MIA_PGM"))
p.add_argument("--out_csv", default=None)
a = p.parse_args()
os.makedirs(a.work, exist_ok=True)

expr = load_expression(a.counts)
ann = pd.read_csv(a.annot)
labels = ann.set_index(ann["samplesID"].astype(str))[a.label_col].astype(str).loc[expr.index]
genes = list(expr.columns)
X_pool, y_pool, ids_pool = expr.values.astype(np.float64), labels.values, np.array(expr.index)
ref = None
if a.reference:
    ref = pd.read_csv(a.reference, sep="\t", index_col=0).T[genes]

COPULA_KW = dict(k=30, knots=41, tail=0.02, gauss_mix=0.7)


def fold_ids(f):
    return json.load(open(os.path.join(a.target_dir, f"split_{f}_ids.json")))


def member_mask(f):
    return np.isin(ids_pool, fold_ids(f)["train"]).astype(int)


def metrics(score, y):
    s_in, s_out = score[y == 1], score[y == 0]
    return {"auc": auc(s_in, s_out), "tpr@0.1": tpr_at(s_in, s_out, 0.1), "tpr@0.01": tpr_at(s_in, s_out, 0.01)}


def report(rows):
    df = pd.DataFrame(rows)
    print(df.groupby("attack", sort=False)[["auc", "tpr@0.1", "tpr@0.01"]].agg(["mean", "std"]).round(3).to_string())
    if a.out_csv:
        df.to_csv(a.out_csv, index=False)


# ----------------------------------------------------------------------------- generators
def nd_modules():
    sys.path.insert(0, a.redteam)
    sys.path.insert(0, os.path.join(a.redteam, "mia"))
    from models.synth_shadow_assets import config as C
    from models.synth_shadow_assets import shadow_model as SM
    from models.synth_shadow_assets import loss_features as LF
    from models.synth_shadow_assets.model import generate_samples
    C.DEVICE = a.device
    C.N_NOISE = a.n_noise
    return C, SM, LF, generate_samples


def gen_copula(Xtr, ytr, seed):
    from copula_gen import CopulaGen
    g = CopulaGen(factor="diffusion", device=a.device, seed=seed, diff_kw={"sigma_stop": 0.1}, **COPULA_KW).fit(Xtr, ytr)
    return g.sample(len(Xtr))


_EXT = []


def gen_lrdp(Xtr, ytr, seed):
    from lowrank_dp import LowRankDP
    if a.gen_ext and not _EXT:
        e = load_expression(a.gen_ext).reindex(columns=genes)
        pats = {s[:12] for s in expr.index}
        _EXT.append(e[[s[:12] not in pats for s in e.index]].values)
    g = LowRankDP(epsilon=a.gen_eps, external=_EXT[0] if _EXT else None, seed=seed).fit(Xtr, ytr)
    return g.sample(len(Xtr))


def gen_mvn(Xtr, ytr, seed):
    """CAMDA MVN baseline: per-class full covariance, plus noise drawn from the same covariance x noise level"""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in np.unique(ytr):
        Z = Xtr[ytr == c]
        Cc = np.cov(Z, rowvar=False)
        w, V = np.linalg.eigh((Cc + Cc.T) / 2)
        L = V * np.sqrt(np.maximum(w, 0))
        Xs.append(Z.mean(0) + rng.standard_normal((len(Z), Xtr.shape[1])) @ L.T * np.sqrt(1 + a.mvn_noise))
        ys += [c] * len(Z)
    return np.vstack(Xs), np.array(ys)


def gen_nd(Xtr, ytr, seed, tag):
    import io, contextlib
    C, SM, LF, generate_samples = nd_modules()
    with contextlib.redirect_stdout(io.StringIO()):
        model, tr, scaler = SM.train_shadow_model(Xtr, os.path.join(a.work, f"ndgen_{tag}.pt"), seed, device=a.device)
        Xs = generate_samples(model, tr, len(Xtr), Xtr.shape[1], label=C.DUMMY_LABEL, device=a.device, scaler=scaler)
    return Xs, np.array(["na"] * len(Xs))


def shadow_split(k):
    rng = np.random.default_rng(1000 + k)
    m = np.zeros(len(X_pool), int)
    m[rng.permutation(len(X_pool))[: int(0.8 * len(X_pool))]] = 1
    return m


def load_target(f):
    fx = os.path.join(a.target_dir, f"synthetic_data_split_{f}.csv")
    S = pd.read_csv(fx)[genes].values.astype(np.float64)
    ly = fx.replace("_data_", "_labels_")
    ys = pd.read_csv(ly).iloc[:, 0].astype(str).values if os.path.exists(ly) else np.array(["na"] * len(S))
    return S, ys


def stage_base():
    if a.gen == "mvn":  # build the targets too
        os.makedirs(a.target_dir, exist_ok=True)
        for f in a.folds:
            fx = os.path.join(a.target_dir, f"synthetic_data_split_{f}.csv")
            if os.path.exists(fx):
                continue
            ids = fold_ids(f)
            Xs, ys = gen_mvn(expr.loc[ids["train"]].values, labels.loc[ids["train"]].values, 500 + f)
            pd.DataFrame(Xs, columns=genes).to_csv(fx, index=False)
            pd.DataFrame({a.label_col: ys}).to_csv(fx.replace("_data_", "_labels_"), index=False)
    if a.gen == "nd":  # positive control: build the targets too
        os.makedirs(a.target_dir, exist_ok=True)
        for f in a.folds:
            fx = os.path.join(a.target_dir, f"synthetic_data_split_{f}.csv")
            if os.path.exists(fx):
                continue
            ids = fold_ids(f)
            Xs, _ = gen_nd(expr.loc[ids["train"]].values, None, 500 + f, f"target{f}")
            pd.DataFrame(Xs, columns=genes).to_csv(fx, index=False)
            print(f"[base] ND target fold {f} written", flush=True)
    for k in range(a.K):
        out = os.path.join(a.work, f"base_{k}.npz")
        if os.path.exists(out):
            continue
        m = shadow_split(k)
        t0 = time.time()
        if a.gen == "copula":
            Xs, ys = gen_copula(X_pool[m == 1], y_pool[m == 1], k)
        elif a.gen == "lrdp":
            Xs, ys = gen_lrdp(X_pool[m == 1], y_pool[m == 1], k)
        elif a.gen == "mvn":
            Xs, ys = gen_mvn(X_pool[m == 1], y_pool[m == 1], k)
        else:
            Xs, ys = gen_nd(X_pool[m == 1], None, k, f"base{k}")
        np.savez_compressed(out, X=Xs.astype(np.float32), y=ys, member=m)
        print(f"[base] shadow {k}: {Xs.shape} in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- simple attacks
def stage_simple():
    sys.path.insert(0, a.redteam)
    sys.path.insert(0, os.path.join(a.redteam, "mia"))
    from mia.models.mahalanobis import MahalanobisNoAux, MahalanobisAux, fix_covariance_matrix
    sys.path.insert(0, a.mama_repo)
    import io, contextlib
    from competition_mia import build_pgm_focal_points, encode_dataframes, mama_mia_score
    rows = []
    T = pd.DataFrame(X_pool, columns=genes)
    for f in a.folds:
        S, ys = load_target(f)
        y = member_mask(f)
        Sd = pd.DataFrame(S, columns=genes)
        mu, cov = Sd.mean().values, fix_covariance_matrix(Sd.cov().values)
        sc = {"MahalaMIA (no aux)": MahalanobisNoAux._attack(T, genes, mu, cov)}
        if ref is not None:
            sc["MahalaMIA (aux)"] = MahalanobisAux._attack(T, genes, mu, cov, ref.mean().values,
                                                           fix_covariance_matrix(ref.cov().values))
        for nb in a.mama_bins:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    Se, Re, Te = encode_dataframes(Sd.copy(), T.copy(), T.copy(), genes, nb, standardize=True, name=f"cg_{os.getpid()}")
                    sc[f"MAMA-MIA 1-way ({nb} bins)"] = mama_mia_score(Se, Re, Te, build_pgm_focal_points(genes, None))
                    if len(np.unique(ys)) > 1:
                        Se["label"], Re["label"], Te["label"] = ys, y_pool, y_pool
                        sc[f"MAMA-MIA 1+2-way ({nb} bins)"] = mama_mia_score(
                            Se, Re, Te, build_pgm_focal_points(genes + ["label"], "label"))
            except Exception as e:  # encoder failure on some generators
                print(f"  MAMA-MIA {nb} bins failed on fold {f}: {type(e).__name__}", flush=True)
        for k, v in sc.items():
            rows.append({"attack": k, "fold": f, **metrics(np.asarray(v, float), y)})
        print(f"[simple] fold {f}: " + "  ".join(f"{r['attack']} {r['auc']:.3f}" for r in rows if r["fold"] == f), flush=True)
    report(rows)


# ----------------------------------------------------------------------------- meta-classifier
def fit_meta(F, y, groups):
    """Red-team default classifiers (no Optuna); group-held-out early stopping and AUC weights."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    sys.path.insert(0, os.path.join(a.redteam, "mia"))
    from models.synth_shadow_assets import config as C

    def compute_softmax_weights(aucs, temperature, min_auc_gate):  # copy of the red-team function (their module needs lightgbm)
        names = list(aucs)
        v = np.array([aucs[n] for n in names], float)
        gate = (v >= min_auc_gate).astype(float)
        if gate.sum() == 0:
            gate = np.ones(len(names))
        e = np.exp(v / temperature - (v / temperature)[gate > 0].max()) * gate
        return {n: float(x) for n, x in zip(names, e / e.sum())}
    ug = np.unique(groups)
    val = np.isin(groups, ug[-max(1, len(ug) // 5):])
    tr = ~val
    xp = dict(C.XGB_PARAMS, device="cpu", early_stopping_rounds=C.XGB_EARLY_STOPPING_ROUNDS, n_jobs=8)
    cp = dict(C.CAT_PARAMS, thread_count=8)
    models, aucs = {}, {}
    m = XGBClassifier(**xp).fit(F[tr], y[tr], eval_set=[(F[val], y[val])], verbose=False)
    models["xgb"] = lambda Z, m=m: m.predict_proba(Z)[:, 1]
    m2 = RandomForestClassifier(**C.RF_PARAMS, n_jobs=8, random_state=0).fit(F[tr], y[tr])
    models["rf"] = lambda Z, m=m2: m.predict_proba(Z)[:, 1]
    m3 = CatBoostClassifier(**cp).fit(F[tr], y[tr], eval_set=(F[val], y[val]))
    models["cat"] = lambda Z, m=m3: m.predict_proba(Z)[:, 1]
    for n, fn in models.items():
        s = fn(F[val])
        aucs[n] = auc(s[y[val] == 1], s[y[val] == 0])
    w = compute_softmax_weights(aucs, temperature=C.ENSEMBLE_TEMPERATURE, min_auc_gate=C.ENSEMBLE_MIN_AUC_GATE)
    w = {k: v for k, v in w.items() if v > 0} or {k: 1 / len(models) for k in models}
    print(f"[meta] held-out shadow AUCs {({k: round(v, 3) for k, v in aucs.items()})}  weights {({k: round(v, 2) for k, v in w.items()})}", flush=True)
    return models, w, aucs


def score_meta(models, w, F, y, f, prefix, rows):
    ens = np.zeros(len(F))
    for n, fn in models.items():
        s = fn(F)
        rows.append({"attack": f"{prefix} [{n}]", "fold": f, **metrics(s, y)})
        ens += w.get(n, 0) * s
    rows.append({"attack": f"{prefix} [ensemble]", "fold": f, **metrics(ens, y)})
    # single best raw feature direction, as a sanity floor for the meta-classifier
    print(f"[{prefix}] fold {f}: ensemble AUC {rows[-1]['auc']:.3f}", flush=True)


# ----------------------------------------------------------------------------- MeLoMIA (ND synth-shadows)
def stage_melomia():
    import io, contextlib
    C, SM, LF, _ = nd_modules()
    n_t = len(C.T_SUPERSET)
    Xp32 = X_pool.astype(np.float32)

    def features(S, tag, seed):
        cache = os.path.join(a.work, f"melo_{tag}.npy")
        if os.path.exists(cache):
            return np.load(cache)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            model, tr, scaler = SM.train_shadow_model(S.astype(np.float64), os.path.join(a.work, f"melo_{tag}.pt"), seed, device=a.device)
            def raw(X):
                Xs = scaler.transform(X.astype(np.float64)).astype(np.float32)
                return LF.extract_loss_features(model, tr, Xs, np.full(len(Xs), C.DUMMY_LABEL, np.int64),
                                                t_list=C.T_SUPERSET, n_noise=a.n_noise, device=a.device, batch_size=128)
            R = raw(Xp32)
            if ref is not None:
                # red team's calibrate_raw_features (their module needs optuna): z-score per timestep
                # with the reference non-member loss distribution
                rr = raw(ref.values.astype(np.float32)).reshape(len(ref), n_t, a.n_noise)
                r3 = R.reshape(len(R), n_t, a.n_noise)
                R = ((r3 - rr.mean(axis=(0, 2)).reshape(1, n_t, 1)) / (rr.std(axis=(0, 2)).reshape(1, n_t, 1) + 1e-8)
                     ).reshape(len(R), n_t * a.n_noise)
            F = LF.summarize_features(R, a.n_noise, n_t)
        np.save(cache, F)
        os.remove(os.path.join(a.work, f"melo_{tag}.pt"))
        print(f"[melomia] {tag}: features {F.shape} in {time.time() - t0:.0f}s", flush=True)
        return F

    Fs, ys, gs = [], [], []
    for k in range(a.K):
        b = np.load(os.path.join(a.work, f"base_{k}.npz"), allow_pickle=True)
        Fs.append(features(b["X"], f"shadow{k}", k))
        ys.append(b["member"])
        gs.append(np.full(len(X_pool), k))
    models, w, _ = fit_meta(np.vstack(Fs), np.concatenate(ys), np.concatenate(gs))
    rows = []
    for f in a.folds:
        S, _ = load_target(f)
        F = features(S, f"proxy{f}", 100 + f)
        y = member_mask(f)
        score_meta(models, w, F, y, f, "MeLoMIA", rows)
        rows.append({"attack": "MeLoMIA raw: -mean loss, low t", "fold": f, **metrics(-F[:, :5].mean(1), y)})
    report(rows)


# ----------------------------------------------------------------------------- CopulaGen-aware synth-shadow
def stage_aware():
    from copula_gen import CopulaGen

    def features(S, ys):
        g = CopulaGen(factor="gauss", seed=0, **COPULA_KW).fit(S, ys)
        fm, mg = g.fm, g.marg
        F = np.zeros((len(X_pool), 18))
        cls = {c: i for i, c in enumerate(sorted(np.unique(y_pool)))}
        for c in np.unique(y_pool):
            m = y_pool == c
            if c not in fm.stats:
                continue
            st, X = fm.stats[c], X_pool[m]
            T = (X - mg.loc[c]) / mg.scale[c]
            Z = mg.to_normal(X, c)
            R = Z - st.mu
            Sc = R @ fm.W.T
            E = R - Sc @ fm.W
            w_, V = np.linalg.eigh(st.score_cov)
            P = (Sc @ V) ** 2 / np.maximum(w_, 1e-6)
            Sz = (S[ys == c] - mg.loc[c]) / mg.scale[c]
            d = nn_dist(T, Sz, 3) if len(Sz) >= 3 else np.zeros(len(T))
            d1 = nn_dist(T, Sz, 1) if len(Sz) >= 1 else np.zeros(len(T))
            dall = nn_dist(X, S, 1)
            Ssc = (mg.to_normal(S[ys == c], c) - st.mu) @ fm.W.T
            df_ = nn_dist(Sc, Ssc, 1) if len(Ssc) else np.zeros(len(T))
            F[m] = np.c_[P.sum(1), (E ** 2 / st.resid_var).sum(1), (E ** 2).sum(1), (Z ** 2).sum(1), (T ** 2).sum(1),
                         np.abs(Z).max(1), (np.abs(Z) > 3).mean(1), (np.abs(Z) >= 3.99).mean(1), d1, d, dall, df_,
                         np.sort(P, 1)[:, -1], np.sort(P, 1)[:, -3:].sum(1), P[:, -5:].sum(1), P[:, :5].sum(1),
                         np.full(len(T), (ys == c).sum()), np.full(len(T), cls[c])]
        return F

    Fs, yy, gs = [], [], []
    for k in range(a.K):
        b = np.load(os.path.join(a.work, f"base_{k}.npz"), allow_pickle=True)
        Fs.append(features(b["X"].astype(np.float64), b["y"].astype(str)))
        yy.append(b["member"])
        gs.append(np.full(len(X_pool), k))
        print(f"[aware] shadow {k} features done", flush=True)
    models, w, _ = fit_meta(np.vstack(Fs), np.concatenate(yy), np.concatenate(gs))
    rows = []
    for f in a.folds:
        S, ys = load_target(f)
        F = features(S, ys)
        y = member_mask(f)
        score_meta(models, w, F, y, f, "Copula-aware synth-shadow", rows)
        for c in np.unique(y_pool):  # per-class view of the ensemble
            m = y_pool == c
            ens = sum(w.get(n, 0) * fn(F[m]) for n, fn in models.items())
            if y[m].min() != y[m].max():
                rows.append({"attack": f"  per class: {c} (n={int((m & (y == 1)).sum())})", "fold": f, **metrics(ens, y[m])})
    report(rows)


{"simple": stage_simple, "base": stage_base, "melomia": stage_melomia, "aware": stage_aware}[a.stage]()
