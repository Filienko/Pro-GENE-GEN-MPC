"""StructPGM: Private-PGM with richer, still low-treewidth structure.

The CAMDA P-PGM measures only 1-way marginals and (gene, label) marginals, so
genes are independent given the label and co-expression is lost.  Adding
arbitrary gene-gene marginals quickly makes junction-tree inference
intractable (BRCA has ~59k gene pairs with |r| > 0.3).  StructPGM offers three
clique structures, all with treewidth <= 3:

  star : (gene, label)                         -- current P-PGM (baseline)
  tree : (gene_i, gene_j, label) on a Chow-Liu maximum spanning tree of
         within-class |correlation|
  hub  : gene modules with an auxiliary "eigengene" attribute per module
         (discretised PC1 of the module's within-class-centred genes)
         cliques (gene, eigengene_m(gene), label) for every gene, plus a
         Chow-Liu tree over eigengenes (e_i, e_j, label).  With
         second_module=True each gene is also tied to the best tree-adjacent
         second eigengene: (gene, e_a, e_b, label).

Measurements use the same scheme as models/Private_PGM/model.py::_train_standard:
Gaussian mechanism, measurements grouped into rounds (1-way marginals, then
cliques), weights normalised to unit L2 norm per round, sigma calibrated with
RDP for (epsilon, delta).  epsilon = inf disables noise.

Inference: 'direct' (default) samples the decomposable model straight from the
(noisy) clique tables -- exact for epsilon = inf and seconds to run.  'mbi'
runs Private-PGM mirror descent (needs >= 1000 iterations to converge on 978
genes, ~45-50 min per fold on CPU; use --backend torch on a GPU).

Privacy notes (also in the README): bin edges, bin means, the tree and the
gene modules are computed from the training data without noise (as in the
existing P-PGM preprocessing), so the formal DP guarantee covers the
marginals only.  Pass --external_counts to take structure and bin edges from a
non-member cohort instead.
"""

import math
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "Private_PGM"))  # bundled mbi
from mbi import Dataset, Domain, FactoredInference  # noqa: E402

LABEL = "label"


# --------------------------------------------------------------------------- #
# Privacy calibration (RDP of the Gaussian mechanism, no scipy needed)
# --------------------------------------------------------------------------- #
def calibrate_sigma(epsilon: float, delta: float, rounds: int = 2) -> float:
    """Smallest sigma such that `rounds` Gaussian mechanisms with L2
    sensitivity 1 are (epsilon, delta)-DP (RDP accounting, alpha in 1.1..4096)."""
    if not np.isfinite(epsilon):
        return 0.0
    alphas = np.concatenate([np.linspace(1.1, 10, 90), np.arange(11, 4097)])

    def eps_of(sigma):
        rdp = rounds * alphas / (2 * sigma ** 2)
        return float(np.min(rdp + math.log(1 / delta) / (alphas - 1)))

    lo, hi = 1e-3, 1.0
    while eps_of(hi) > epsilon:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if eps_of(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- #
# Discretisation
# --------------------------------------------------------------------------- #
class Discretizer:
    """Per-column quantile bins (default 4, like the existing P-PGM).
    inverse='mean'  : each bin -> training bin mean (existing behaviour)
    inverse='smooth': truncated normal within the bin, matched to the bin's
                      training mean / SD (better gene-wise KL, slightly weaker
                      correlations)."""

    def __init__(self, bins: int = 4, inverse: str = "mean"):
        self.bins, self.inverse = bins, inverse

    def fit(self, X: np.ndarray, X_edges: Optional[np.ndarray] = None):
        Xe = X if X_edges is None else X_edges
        q = np.linspace(0, 1, self.bins + 1)
        self.edges = np.quantile(Xe, q, axis=0)            # (bins+1, d)
        B = self.transform(X)
        d = X.shape[1]
        self.means = np.zeros((self.bins, d))
        self.sds = np.zeros((self.bins, d))
        for b in range(self.bins):
            m = B == b
            cnt = m.sum(0)
            s = np.where(m, X, 0).sum(0)
            ss = np.where(m, X ** 2, 0).sum(0)
            mean = np.where(cnt > 0, s / np.maximum(cnt, 1), (self.edges[b] + self.edges[b + 1]) / 2)
            var = np.where(cnt > 1, ss / np.maximum(cnt, 1) - mean ** 2, 0)
            self.means[b], self.sds[b] = mean, np.sqrt(np.maximum(var, 0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        B = np.zeros(X.shape, dtype=int)
        for b in range(1, self.bins):
            B += (X > self.edges[b]).astype(int)
        return B

    def inverse_transform(self, B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cols = np.arange(B.shape[1])[None, :]
        mean = self.means[B, cols]
        if self.inverse == "mean":
            return mean
        from scipy.stats import truncnorm
        sd = np.maximum(self.sds[B, cols], 1e-6)
        lo = self.edges[B, cols]
        hi = self.edges[B + 1, cols]
        a, b = (lo - mean) / sd, (hi - mean) / sd
        a, b = np.minimum(a, b - 1e-6), b
        return truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng)


# --------------------------------------------------------------------------- #
# Structure helpers
# --------------------------------------------------------------------------- #
def within_class_centre(X, y):
    Z = np.empty_like(X)
    for c in np.unique(y):
        Z[y == c] = X[y == c] - X[y == c].mean(0)
    return Z


def max_spanning_tree(W: np.ndarray) -> List[Tuple[int, int]]:
    """Prim's algorithm on a dense weight matrix (maximise total weight)."""
    k = len(W)
    A = W.copy()
    np.fill_diagonal(A, -np.inf)
    in_tree = np.zeros(k, bool)
    in_tree[0] = True
    best, parent = A[0].copy(), np.zeros(k, int)
    edges = []
    for _ in range(k - 1):
        cand = np.where(in_tree, -np.inf, best)
        j = int(cand.argmax())
        edges.append((int(parent[j]), j))
        in_tree[j] = True
        upd = (A[j] > best) & ~in_tree
        best[upd], parent[upd] = A[j][upd], j
    return edges


def bfs_orient(edges: List[Tuple[int, int]], k: int, root: int = 0) -> List[Tuple[int, int]]:
    """Order and orient tree edges as (parent, child) in BFS order from root, so that
    every edge after the first introduces exactly one new node."""
    adj = {i: [] for i in range(k)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    out, seen, q = [], {root}, deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                out.append((u, v))
                q.append(v)
    return out


def gene_modules(Rw: np.ndarray, M: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    """k-means (cosine) on within-class correlation profiles -> module id per gene."""
    F = Rw / np.linalg.norm(Rw, axis=1, keepdims=True)
    cent = F[rng.choice(len(F), M, replace=False)]
    lab = None
    for _ in range(iters):
        new = (F @ cent.T).argmax(1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        for m in range(M):
            if (lab == m).any():
                c = F[lab == m].mean(0)
                cent[m] = c / np.linalg.norm(c)
    # relabel to consecutive ids (drop empty modules)
    _, lab = np.unique(lab, return_inverse=True)
    return lab


class StructPGM:
    def __init__(self, structure: str = "hub", bins: int = 4, hub_bins: int = 4,
                 modules: int = 40, second_module: bool = False, inverse: str = "mean",
                 epsilon: float = float("inf"), delta: float = 1e-5, num_iters: int = 1000,
                 backend: str = "numpy", inference: str = "direct", seed: int = 0,
                 verbose: bool = True):
        assert structure in ("star", "tree", "hub")
        self.structure, self.bins, self.hub_bins = structure, bins, hub_bins
        self.n_modules, self.second_module = modules, second_module
        self.epsilon, self.delta, self.num_iters = epsilon, delta, num_iters
        self.inverse = inverse
        self.backend = backend
        assert inference in ("direct", "mbi")
        self.inference = inference
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.verbose = verbose

    # ----------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, genes: List[str],
            external: Optional[np.ndarray] = None, external_y: Optional[np.ndarray] = None):
        np.random.seed(self.seed)  # mbi uses the global numpy RNG
        self.genes = list(genes)
        d = X.shape[1]
        self.classes = [str(c) for c in np.unique(y)]
        y_idx = np.searchsorted(self.classes, y.astype(str))

        # structure statistics: from the training data or an external cohort
        Xs = X if external is None else external
        ys = y if external is None else (external_y if external_y is not None else np.zeros(len(external)))
        Zs = within_class_centre(Xs, np.asarray(ys))
        Rw = np.corrcoef(Zs, rowvar=False)
        Rw = np.nan_to_num(Rw)

        self.disc = Discretizer(self.bins, self.inverse).fit(X, None if external is None else external)
        B = self.transform_genes(X)
        cols: Dict[str, np.ndarray] = {f"g{j}": B[:, j] for j in range(d)}
        sizes = {f"g{j}": self.bins for j in range(d)}
        cliques: List[Tuple[str, ...]] = []

        if self.structure == "star":
            cliques = [(f"g{j}", LABEL) for j in range(d)]
        elif self.structure == "tree":
            self.edges = bfs_orient(max_spanning_tree(np.abs(Rw)), d)
            cliques = [(f"g{a}", f"g{b}", LABEL) for a, b in self.edges]
        else:  # hub
            M = min(self.n_modules, d)
            self.module = gene_modules(Rw, M, self.rng)
            M = int(self.module.max()) + 1
            # eigengene loadings: PC1 of z-scored, within-class-centred module genes
            sd = Zs.std(0) + 1e-12
            Zz = Zs / sd
            self.eg_load, self.eg_sd = [], sd
            E_struct = np.zeros((len(Zz), M))
            for m in range(M):
                g = np.where(self.module == m)[0]
                _, _, vt = np.linalg.svd(Zz[:, g], full_matrices=False)
                v = vt[0] * np.sign(vt[0].sum() + 1e-12)
                self.eg_load.append((g, v))
                E_struct[:, m] = Zz[:, g] @ v
            # eigengene values for the training rows (class-centred with training class means)
            self.class_mu = {c: X[y_idx == i].mean(0) for i, c in enumerate(self.classes)}
            E = self.eigengenes(X, y_idx)
            q = np.linspace(0, 1, self.hub_bins + 1)[1:-1]
            self.eg_edges = np.quantile(E, q, axis=0)
            EB = np.zeros(E.shape, int)
            for b in range(len(q)):
                EB += (E > self.eg_edges[b]).astype(int)
            for m in range(M):
                cols[f"e{m}"] = EB[:, m]
                sizes[f"e{m}"] = self.hub_bins
            CE = np.nan_to_num(np.corrcoef(E_struct, rowvar=False))
            self.eg_edges_tree = bfs_orient(max_spanning_tree(np.abs(CE)), M) if M > 1 else []
            nbrs = {m: set() for m in range(M)}
            for a, b in self.eg_edges_tree:
                nbrs[a].add(b)
                nbrs[b].add(a)
                cliques.append((f"e{a}", f"e{b}", LABEL))
            if M == 1:
                cliques.append(("e0", LABEL))
            # gene -> eigengene correlation, used to pick a second module
            Cge = (Zz.T @ (E_struct / (E_struct.std(0) + 1e-12))) / len(Zz)
            for j in range(d):
                m = int(self.module[j])
                if self.second_module and nbrs[m]:
                    m2 = max(nbrs[m], key=lambda t: abs(Cge[j, t]))
                    cliques.append((f"g{j}", f"e{m}", f"e{m2}", LABEL))
                else:
                    cliques.append((f"g{j}", f"e{m}", LABEL))

        cols[LABEL] = y_idx
        sizes[LABEL] = len(self.classes)
        df = pd.DataFrame(cols)
        domain = Domain(list(sizes.keys()), list(sizes.values()))
        data = Dataset(df, domain)
        self.cliques = cliques
        self.n_cells = int(sum(np.prod([sizes[a] for a in cl]) for cl in cliques))
        if self.verbose:
            print(f"  [StructPGM] structure={self.structure} attrs={len(sizes)} cliques={len(cliques)} "
                  f"cells={self.n_cells:,} epsilon={self.epsilon}")

        # ---------------- measurements (Gaussian mechanism, same scheme as
        # Private_PGM._train_standard: rounds normalised to unit L2 norm, RDP calibration)
        sigma = calibrate_sigma(self.epsilon, self.delta, rounds=2)
        self.sigma = sigma
        if self.inference == "mbi":
            rounds = [[(a,) for a in domain.attrs], cliques]
        else:  # direct sampling only needs the label marginal and the cliques
            rounds = [[(LABEL,)], cliques]
        meas, self.tables = [], {}
        for proj_list in rounds:
            w = 1.0 / math.sqrt(len(proj_list))  # unit L2 norm over the round
            for proj in proj_list:
                x = data.project(proj).datavector()
                yv = w * x + sigma * self.rng.standard_normal(x.size)
                meas.append((sparse.eye(x.size), yv / w, 1.0 / w, proj))
                self.tables[proj] = (yv / w).reshape([sizes[a] for a in proj])
        self.sizes = sizes
        if self.inference == "mbi":
            engine = FactoredInference(domain, backend=self.backend, log=False, iters=self.num_iters)
            self.model = engine.estimate(meas, total=len(df), engine="MD")
        return self

    def _direct_sample(self, n: int) -> pd.DataFrame:
        """Exact sampler for these decomposable structures: label first, then each
        clique (in BFS order) adds its unsampled variables from the clique's
        conditional table.  With epsilon = inf this samples the maximum-likelihood
        model (product of clique marginals / separators); with noise it is the
        PrivBayes-style post-processing of the noisy marginals."""
        out = {}
        lab = np.maximum(self.tables[(LABEL,)], 0) + 1e-9
        out[LABEL] = self.rng.choice(len(lab), size=n, p=lab / lab.sum())
        for cl in self.cliques:
            T = np.maximum(self.tables[cl], 0)
            for v in [a for a in cl if a not in out]:
                cond = [a for a in cl if a in out]
                keep = cond + [v]
                axes = tuple(i for i, a in enumerate(cl) if a not in keep)
                Tm = T.sum(axis=axes) if axes else T
                # reorder axes to cond + [v]
                cur = [a for a in cl if a in keep]
                Tm = np.transpose(Tm, [cur.index(a) for a in keep])
                k = self.sizes[v]
                P = Tm.reshape(-1, k)
                marg_v = P.sum(0) + 1e-9
                rows = P.sum(1, keepdims=True)
                P = np.where(rows > 0, P, marg_v)  # empty / negative rows -> marginal of v
                if cond:
                    idx = np.ravel_multi_index(tuple(out[a] for a in cond),
                                               tuple(self.sizes[a] for a in cond))
                    P = P[idx]
                else:
                    P = np.broadcast_to(P, (n, k))
                P = P + 1e-12
                cdf = np.cumsum(P, 1)
                cdf /= cdf[:, -1:]
                out[v] = np.minimum((self.rng.random((n, 1)) > cdf).sum(1), k - 1)
        return pd.DataFrame(out)

    # ------------------------------------------------------------ helpers
    def transform_genes(self, X):
        return self.disc.transform(X)

    def eigengenes(self, X, y_idx):
        E = np.zeros((len(X), len(self.eg_load)))
        for i, c in enumerate(self.classes):
            m = y_idx == i
            if m.any():
                Z = (X[m] - self.class_mu[c]) / self.eg_sd
                for k, (g, v) in enumerate(self.eg_load):
                    E[m, k] = Z[:, g] @ v
        return E

    # ------------------------------------------------------------- sample
    def _ancestral_sample(self, n: int) -> pd.DataFrame:
        """Sample from the fitted graphical model attribute by attribute, following the
        reverse elimination order (same scheme as mbi's synthetic_data, but vectorised,
        with random instead of systematic draws, and independent of pandas groupby
        semantics, which changed in pandas >= 2.2)."""
        gm = self.model
        dom = gm.domain
        cliques = [set(cl) for cl in gm.cliques]
        order = gm.elimination_order[::-1]
        out = {}
        used = []
        for col in order:
            rel = [cl for cl in cliques if col in cl]
            parents = [a for a in used if any(a in cl for cl in rel)]
            marg = gm.project(tuple(parents) + (col,)).datavector(flatten=False)
            if hasattr(marg, "detach"):  # torch backend
                marg = marg.detach().cpu().numpy()
            marg = np.maximum(np.asarray(marg, dtype=float), 0)
            k = dom[col]
            if parents:
                P = marg.reshape(-1, k)
                idx = np.ravel_multi_index(tuple(out[p] for p in parents),
                                           tuple(dom[p] for p in parents))
                P = P[idx]
            else:
                P = np.broadcast_to(marg.reshape(1, k), (n, k))
            P = P + 1e-12
            cdf = np.cumsum(P, 1)
            cdf /= cdf[:, -1:]
            u = self.rng.random((n, 1))
            out[col] = np.minimum((u > cdf).sum(1), k - 1)
            used.append(col)
        return pd.DataFrame(out)

    def sample(self, n: int):
        syn = self._direct_sample(n) if self.inference == "direct" else self._ancestral_sample(n)
        B = syn[[f"g{j}" for j in range(len(self.genes))]].values
        X = self.disc.inverse_transform(B, self.rng)
        y = np.array(self.classes)[syn[LABEL].values]
        return X, y
