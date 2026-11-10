"""Shared pieces for the FactorGen generators (numpy only).

Design
------
Every generator in this package writes a synthetic sample of class c as

    x = mu_c + W^T s + e,     s in R^k  (factor scores),  e ~ N(0, diag(r_c))

* W (k x d) is a fixed low-rank basis, either fitted on *within-class centred*
  training data or loaded from an external (non-member) cohort.
* mu_c, r_c are per-class means and per-gene residual variances, shrunk toward
  the pooled within-class estimate.
* Only the k-dim factor scores s are modelled by the generator
  (Gaussian / CVAE / diffusion).  The 978-dim residual is drawn independently,
  so the synthetic data is full-rank and does not live in the span of the
  training rows (the leak MahalaMIA exploits in the CAMDA MVN baseline).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_expression(count_tsv: str) -> pd.DataFrame:
    """CAMDA count file: genes x samples TSV (first column = gene id).
    Returns samples x genes."""
    df = pd.read_csv(count_tsv, sep="\t", index_col=0).T
    df.index = df.index.astype(str)
    return df.astype(float)


def load_labels(annot_csv: Optional[str], samples: pd.Index,
                sample_col: str = "samplesID",
                label_col: str = "Subtype") -> pd.Series:
    """Annotation CSV with a sample-id column and a label column.
    Without an annotation file every sample gets the label 'all'."""
    if annot_csv is None:
        print("[FactorGen] WARNING: no annotation file given -> unconditional model "
              "(single class 'all').")
        return pd.Series("all", index=samples, name=label_col)
    ann = pd.read_csv(annot_csv)
    if sample_col not in ann.columns:
        raise ValueError(f"'{sample_col}' not in {annot_csv}: {list(ann.columns)}")
    if label_col not in ann.columns:
        raise ValueError(f"'{label_col}' not in {annot_csv}: {list(ann.columns)}")
    lab = ann.set_index(ann[sample_col].astype(str))[label_col].astype(str)
    missing = samples.difference(lab.index)
    if len(missing):
        print(f"[FactorGen] WARNING: {len(missing)} samples have no label and are dropped.")
    return lab.reindex(samples).dropna()


def load_splits(splits_yaml: Optional[str], samples: pd.Index, labels: pd.Series,
                n_splits: int = 5, seed: int = 42) -> Dict[int, Tuple[List[str], List[str]]]:
    """Return {split_no: (train_ids, test_ids)}.

    * With a CAMDA splits YAML ({"splits": {"split_1": {"train_index": [...],
      "test_index": [...]}}}) the exact challenge folds are reproduced.
    * Otherwise a stratified K-fold split is generated (numpy only).
    """
    if splits_yaml is not None:
        with open(splits_yaml) as f:
            sp = yaml.safe_load(f)["splits"]
        out = {}
        for name, v in sp.items():
            k = int(str(name).split("_")[-1])
            out[k] = ([str(s) for s in v["train_index"]], [str(s) for s in v["test_index"]])
        return dict(sorted(out.items()))

    rng = np.random.default_rng(seed)
    fold_of = pd.Series(-1, index=samples)
    for c in np.unique(labels.values):
        ids = labels.index[labels.values == c].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        for j, chunk in enumerate(np.array_split(ids, n_splits)):
            fold_of[chunk] = j
    out = {}
    for j in range(n_splits):
        test = fold_of.index[fold_of.values == j].tolist()
        train = fold_of.index[fold_of.values != j].tolist()
        out[j + 1] = (train, test)
    return out


def save_synthetic(out_dir: str, split_no: int, X_syn: np.ndarray, y_syn: np.ndarray,
                   genes: List[str], label_col: str) -> Tuple[str, str]:
    """CAMDA naming: synthetic_data_split_{k}.csv (genes only) and
    synthetic_labels_split_{k}.csv (one label column).  These are the files
    the red-team code (red_team.py, KLeak) and evaluate.py read."""
    os.makedirs(out_dir, exist_ok=True)
    fx = os.path.join(out_dir, f"synthetic_data_split_{split_no}.csv")
    fy = os.path.join(out_dir, f"synthetic_labels_split_{split_no}.csv")
    pd.DataFrame(X_syn, columns=genes).to_csv(fx, index=False)
    pd.DataFrame({label_col: y_syn}).to_csv(fy, index=False)
    return fx, fy


# --------------------------------------------------------------------------- #
# Factor basis + per-class statistics
# --------------------------------------------------------------------------- #
def within_class_centre(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Z = np.empty_like(X)
    for c in np.unique(y):
        m = y == c
        Z[m] = X[m] - X[m].mean(0)
    return Z


def fit_basis(X: np.ndarray, y: np.ndarray, k: int, external: Optional[np.ndarray] = None,
              external_y: Optional[np.ndarray] = None) -> np.ndarray:
    """Top-k right singular vectors (k x d) of within-class centred data.

    If `external` (samples x genes, *non-member* cohort with the same
    preprocessing) is given, the basis is learned there instead, so it carries
    no information about the training patients.
    """
    if external is not None:
        Z = external - external.mean(0) if external_y is None else within_class_centre(external, external_y)
    else:
        Z = within_class_centre(X, y)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k]


@dataclass
class ClassStats:
    """Per-class parameters of x = mu + W^T s + e."""
    mu: np.ndarray              # (d,)
    resid_var: np.ndarray       # (d,)
    scores: np.ndarray          # (n_c, k) centred factor scores of the training rows
    score_cov: np.ndarray       # (k, k) shrunk factor covariance
    n: int


@dataclass
class FactorModel:
    W: np.ndarray                                   # (k, d)
    classes: List[str]
    stats: Dict[str, ClassStats] = field(default_factory=dict)
    priors: Dict[str, float] = field(default_factory=dict)
    pooled_score_cov: Optional[np.ndarray] = None
    pooled_resid_var: Optional[np.ndarray] = None
    shrink_used: Dict[str, float] = field(default_factory=dict)


def fit_factor_model(X: np.ndarray, y: np.ndarray, k: int = 50, shrink: float = 0.3,
                     external: Optional[np.ndarray] = None,
                     resid_floor: float = 1e-4, adaptive_shrink: bool = True) -> FactorModel:
    """Fit basis, per-class means, factor covariances and residual variances.

    shrink = lambda in [0, 1]: class factor covariance and residual variance are
    (1 - lambda) * class estimate + lambda * pooled *within-class* estimate.
    (Shrinking toward the global covariance would inject between-class
    variation and create spurious co-expression edges.)
    adaptive_shrink: use lambda_c = max(shrink, min(1, k / n_c)) so that small
    classes (e.g. BRCA Normal-like, ~32 training rows) lean on the pooled estimate.
    """
    W = fit_basis(X, y, k, external=external)
    classes = [str(c) for c in np.unique(y)]
    Zw = within_class_centre(X, y)
    S_all = Zw @ W.T
    pooled_cov = np.cov(S_all, rowvar=False) if W.shape[0] > 1 else np.atleast_2d(S_all.var())
    pooled_rv = np.maximum((Zw - S_all @ W).var(0), resid_floor)

    model = FactorModel(W=W, classes=classes, pooled_score_cov=pooled_cov,
                        pooled_resid_var=pooled_rv)
    n_tot = len(y)
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        S = (Xc - mu) @ W.T
        if len(Xc) > 2:
            cov_c = np.cov(S, rowvar=False)
            rv_c = (Xc - mu - S @ W).var(0)
        else:  # tiny class: rely on pooled statistics only
            cov_c, rv_c = pooled_cov, pooled_rv
        # adaptive shrinkage: a class with n_c rows cannot support a k x k covariance
        # when n_c is close to k (rank-deficient -> membership leak and noisy
        # co-expression), so it borrows at least a fraction k / n_c from the pooled estimate
        lam = shrink
        if adaptive_shrink:
            lam = max(shrink, min(1.0, W.shape[0] / max(len(Xc), 1)))
        cov = (1 - lam) * cov_c + lam * pooled_cov
        rv = np.maximum((1 - lam) * rv_c + lam * pooled_rv, resid_floor)
        model.shrink_used[c] = lam
        model.stats[c] = ClassStats(mu=mu, resid_var=rv, scores=S,
                                    score_cov=np.atleast_2d(cov), n=len(Xc))
        model.priors[c] = len(Xc) / n_tot
    return model


def class_counts(model: FactorModel, n_total: int, uniform: bool = False) -> Dict[str, int]:
    """Number of synthetic rows per class (default: same proportions as training)."""
    if uniform:
        p = {c: 1 / len(model.classes) for c in model.classes}
    else:
        p = model.priors
    counts = {c: int(round(n_total * p[c])) for c in model.classes}
    # fix rounding so the total matches exactly
    diff = n_total - sum(counts.values())
    big = max(counts, key=counts.get)
    counts[big] += diff
    return counts


def moment_match(scores: np.ndarray, target_cov: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Post-process generated (class-centred) factor scores.

    mode = "none": unchanged
           "mean": re-centre to zero mean (class means are added in decode)
           "cov" : re-centre and re-colour so the sample covariance equals the
                   shrunk class covariance; the neural model then only
                   contributes the non-Gaussian shape (higher moments).
    """
    if mode == "none" or len(scores) < 2:
        return scores
    S = scores - scores.mean(0)
    if mode == "mean":
        return S
    if mode != "cov":
        raise ValueError(f"unknown moment_match mode {mode}")
    k = S.shape[1]
    C = np.cov(S, rowvar=False) + 1e-8 * np.eye(k)
    w, V = np.linalg.eigh(C)
    whiten = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    wt, Vt = np.linalg.eigh((target_cov + target_cov.T) / 2)
    colour = Vt @ np.diag(np.sqrt(np.maximum(wt, 0))) @ Vt.T
    return S @ whiten @ colour


def decode(model: FactorModel, c: str, scores: np.ndarray,
           rng: np.random.Generator, resid_scale: float = 1.0) -> np.ndarray:
    """Map factor scores back to gene space and add the independent residual."""
    st = model.stats[c]
    n = scores.shape[0]
    noise = rng.standard_normal((n, len(st.mu))) * np.sqrt(st.resid_var * resid_scale)
    return st.mu + scores @ model.W + noise
