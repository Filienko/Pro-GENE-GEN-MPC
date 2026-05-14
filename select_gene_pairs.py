"""
select_gene_pairs.py — find the gene-gene pairs that most improve Private-PGM
fidelity when passed as gene_gene_cliques to model.train().

Four selection methods:

  pearson  Fast. Ranks every gene pair by |Pearson correlation| in the real
           data. No model training required. Use as a quick starting point.

  gap      Trains one baseline Private-PGM model (no gene-gene cliques),
           generates synthetic data, then ranks pairs by how much the baseline
           fails to reproduce each gene-gene correlation (|real_corr - syn_corr|).

  kl       Recommended. Same single baseline training as gap, but ranks pairs
           by KL(P_real(gᵢ,gⱼ) ‖ P_syn(gᵢ,gⱼ)) on the binned joint distribution.
           Captures the full joint mismatch, not just linear correlation, making
           it a more principled criterion for selecting which pairs to measure.

  greedy   Iteratively adds pairs one at a time, keeping whichever candidate
           reduces mean pairwise KL the most at each step. Expensive: trains
           `top_k` × `--greedy-eval-budget` models. Uses a reduced
           `--greedy-iters` count during search to keep runtime manageable.

Reported metrics (all methods that train a model):
  Correlation MAE        — MAE of all pairwise Pearson correlations
  Correlation RMSE       — RMSE of all pairwise Pearson correlations
  Coexpression TPR       — fraction of high-correlation pairs (|r|≥0.7) recovered
  Mean pairwise KL       — mean KL(P_real ‖ P_syn) over all gene-gene pairs

Usage examples:

  # Fast ranking, no training
  python select_gene_pairs.py --data data/working_bcra_sample.csv \\
      --label-col label --method pearson --top-k 30

  # KL-based ranking (recommended, one baseline training)
  python select_gene_pairs.py --data data/working_bcra_sample.csv \\
      --label-col label --method kl --epsilon 1.0 --delta 1e-5 \\
      --top-k 50 --output recommended_pairs.json

  # Greedy KL optimisation (expensive)
  python select_gene_pairs.py --data data/working_bcra_sample.csv \\
      --label-col label --method greedy --epsilon 1.0 --delta 1e-5 \\
      --top-k 10 --greedy-eval-budget 20 --greedy-iters 2000

Output JSON can be loaded directly as gene_gene_cliques:

  import json
  pairs = [tuple(p) for p in json.load(open("recommended_pairs.json"))["pairs"]]
  model.train(binned_df, config, gene_gene_cliques=pairs)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "models", "Private_PGM"))

from run_benchmark_pgm import (
    bin_dataframe,
    bins_to_continuous,
    compute_correlation_diff,
    compute_coexpression_tpr,
)


# ---------------------------------------------------------------------------
# KL divergence helpers (operate on binned integer DataFrames)
# ---------------------------------------------------------------------------

def _pairwise_kl(real_binned: pd.DataFrame, syn_binned: pd.DataFrame,
                 gi: str, gj: str, n_bins: int, eps: float = 1e-10) -> float:
    """
    KL(P_real(gᵢ,gⱼ) ‖ P_syn(gᵢ,gⱼ)) on the n_bins×n_bins joint distribution.

    Epsilon smoothing ensures the divergence is finite even when a bin is
    empty in the synthetic data.  Both distributions are re-normalised after
    smoothing so the result is a proper KL divergence.
    """
    real_joint = np.zeros((n_bins, n_bins), dtype=float)
    syn_joint = np.zeros((n_bins, n_bins), dtype=float)

    ri = real_binned[gi].values.astype(int).clip(0, n_bins - 1)
    rj = real_binned[gj].values.astype(int).clip(0, n_bins - 1)
    si = syn_binned[gi].values.astype(int).clip(0, n_bins - 1)
    sj = syn_binned[gj].values.astype(int).clip(0, n_bins - 1)

    np.add.at(real_joint, (ri, rj), 1)
    np.add.at(syn_joint, (si, sj), 1)

    p = real_joint.flatten() + eps
    q = syn_joint.flatten() + eps
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def _all_pairwise_kl_df(real_binned: pd.DataFrame, syn_binned: pd.DataFrame,
                        gene_cols: list, n_bins: int) -> pd.DataFrame:
    """
    Compute KL divergence for every gene pair and return a DataFrame sorted
    by kl_div descending (pairs the baseline preserves worst come first).
    """
    rows = []
    for i, gi in enumerate(gene_cols):
        for j in range(i + 1, len(gene_cols)):
            gj = gene_cols[j]
            kl = _pairwise_kl(real_binned, syn_binned, gi, gj, n_bins)
            rows.append({"gene_i": gi, "gene_j": gj, "kl_div": kl})
    return (
        pd.DataFrame(rows)
        .sort_values("kl_div", ascending=False)
        .reset_index(drop=True)
    )


def _mean_pairwise_kl(real_binned: pd.DataFrame, syn_binned: pd.DataFrame,
                      gene_cols: list, n_bins: int) -> float:
    """Mean KL(P_real ‖ P_syn) over all gene-gene pairs."""
    total, count = 0.0, 0
    for i, gi in enumerate(gene_cols):
        for j in range(i + 1, len(gene_cols)):
            total += _pairwise_kl(real_binned, syn_binned, gi, gene_cols[j], n_bins)
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _correlation_rmse(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    real_corr = real_df.corr(method="pearson").fillna(0).values
    syn_corr = syn_df.corr(method="pearson").fillna(0).values
    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    return float(np.sqrt(np.mean((real_corr[mask] - syn_corr[mask]) ** 2)))


def _per_pair_gap(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    cols = real_df.columns.tolist()
    real_corr = real_df.corr(method="pearson").fillna(0)
    syn_corr = syn_df.corr(method="pearson").fillna(0)
    rows = []
    for i, gi in enumerate(cols):
        for j in range(i + 1, len(cols)):
            gj = cols[j]
            rc, sc = real_corr.loc[gi, gj], syn_corr.loc[gi, gj]
            rows.append({"gene_i": gi, "gene_j": gj,
                         "real_corr": rc, "syn_corr": sc, "corr_gap": abs(rc - sc)})
    return pd.DataFrame(rows).sort_values("corr_gap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metric reporting
# ---------------------------------------------------------------------------

def _print_metrics(label: str, corr_mae: float, corr_rmse: float,
                   coexp_tpr: float, mean_kl: float) -> None:
    print(f"\n  [{label}]")
    print(f"    Correlation MAE      : {corr_mae:.4f}  (lower is better)")
    print(f"    Correlation RMSE     : {corr_rmse:.4f}  (lower is better)")
    print(f"    Coexpression TPR     : {coexp_tpr:.4f}  (higher is better, |r|≥0.7)")
    print(f"    Mean pairwise KL     : {mean_kl:.4f}  (lower is better)")


# ---------------------------------------------------------------------------
# Data loading + binning
# ---------------------------------------------------------------------------

def load_and_bin(data_path: str, label_col: str, n_bins: int):
    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. "
                         f"Columns: {list(df.columns)[:10]} …")
    # Drop non-numeric columns that are not the label (e.g. sample-ID columns
    # like "Unnamed: 0" or "TCGA-..." that appear when the CSV was written with
    # a pandas index or a separate sample-name column).
    for col in df.select_dtypes(include=["object"]).columns:
        if col != label_col:
            df = df.drop(columns=[col])

    feature_cols = [c for c in df.columns if c != label_col]
    if df[label_col].dtype == object or not pd.api.types.is_integer_dtype(df[label_col]):
        df[label_col] = pd.Categorical(df[label_col]).codes
    binned_df, bin_means = bin_dataframe(df, label_col, n_bins=n_bins)
    config = {col: n_bins for col in feature_cols}
    config[label_col] = int(df[label_col].nunique())
    return df, binned_df, bin_means, feature_cols, config


# ---------------------------------------------------------------------------
# Training helper — returns (continuous_gene_df, binned_gene_df)
# ---------------------------------------------------------------------------

def _train_and_generate(binned_df, config, label_col, feature_cols, bin_means,
                        epsilon, delta, num_iters, gene_gene_cliques=None,
                        enable_privacy=True):
    """
    Returns (continuous_gene_df, binned_syn_df).

    continuous_gene_df: gene columns mapped back to continuous values via bin_means.
    binned_syn_df:      raw bin-index gene columns (integers), needed for KL computation.
    """
    from models.Private_PGM.model import Private_PGM

    pgm = Private_PGM(
        target_variable=label_col,
        enable_privacy=enable_privacy,
        target_epsilon=epsilon,
        target_delta=delta,
    )
    pgm.train(binned_df, config,
              gene_gene_cliques=gene_gene_cliques or [],
              num_iters=num_iters)

    syn_array = pgm.generate(num_rows=len(binned_df))

    # Binned (integer bin indices)
    binned_syn = pd.DataFrame(syn_array[:, :-1], columns=feature_cols)

    # Continuous (unbin via training-set bin means)
    syn_cont = bins_to_continuous(syn_array, feature_cols, label_col, bin_means)

    return syn_cont[feature_cols], binned_syn


# ---------------------------------------------------------------------------
# Method 1 — pearson  (no training)
# ---------------------------------------------------------------------------

def method_pearson(real_gene_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Rank gene pairs by |Pearson correlation| in the real data."""
    cols = real_gene_df.columns.tolist()
    corr = real_gene_df.corr(method="pearson").fillna(0)
    rows = [{"gene_i": cols[i], "gene_j": cols[j],
             "abs_corr": abs(corr.iloc[i, j])}
            for i in range(len(cols)) for j in range(i + 1, len(cols))]
    return (pd.DataFrame(rows)
            .sort_values("abs_corr", ascending=False)
            .reset_index(drop=True)
            .head(top_k))


# ---------------------------------------------------------------------------
# Method 2 — gap  (one training, ranked by correlation gap)
# ---------------------------------------------------------------------------

def method_gap(real_gene_df, real_binned_gene, binned_df, config,
               label_col, feature_cols, bin_means,
               epsilon, delta, num_iters, top_k, n_bins, enable_privacy):
    print("\nTraining baseline Private-PGM (no gene-gene cliques)…")
    t0 = time.time()
    syn_cont, syn_binned = _train_and_generate(
        binned_df, config, label_col, feature_cols, bin_means,
        epsilon, delta, num_iters, enable_privacy=enable_privacy)
    print(f"  Baseline trained in {time.time()-t0:.1f}s")

    corr_mae = compute_correlation_diff(real_gene_df, syn_cont)
    corr_rmse = _correlation_rmse(real_gene_df, syn_cont)
    coexp_tpr = compute_coexpression_tpr(real_gene_df, syn_cont)
    mean_kl = _mean_pairwise_kl(real_binned_gene, syn_binned, feature_cols, n_bins)
    _print_metrics("baseline (no gene-gene cliques)", corr_mae, corr_rmse, coexp_tpr, mean_kl)

    # Build per-pair table with both gap and KL columns
    gap_df = _per_pair_gap(real_gene_df, syn_cont)
    kl_df = _all_pairwise_kl_df(real_binned_gene, syn_binned, feature_cols, n_bins)
    kl_map = {(r.gene_i, r.gene_j): r.kl_div for _, r in kl_df.iterrows()}
    gap_df["kl_div"] = gap_df.apply(
        lambda r: kl_map.get((r.gene_i, r.gene_j),
                             kl_map.get((r.gene_j, r.gene_i), float("nan"))), axis=1)

    return gap_df.head(top_k), dict(
        corr_mae=corr_mae, corr_rmse=corr_rmse,
        coexp_tpr=coexp_tpr, mean_pairwise_kl=mean_kl)


# ---------------------------------------------------------------------------
# Method 3 — kl  (one training, ranked by pairwise KL divergence)
# ---------------------------------------------------------------------------

def method_kl(real_gene_df, real_binned_gene, binned_df, config,
              label_col, feature_cols, bin_means,
              epsilon, delta, num_iters, top_k, n_bins, enable_privacy):
    print("\nTraining baseline Private-PGM (no gene-gene cliques)…")
    t0 = time.time()
    syn_cont, syn_binned = _train_and_generate(
        binned_df, config, label_col, feature_cols, bin_means,
        epsilon, delta, num_iters, enable_privacy=enable_privacy)
    print(f"  Baseline trained in {time.time()-t0:.1f}s")

    corr_mae = compute_correlation_diff(real_gene_df, syn_cont)
    corr_rmse = _correlation_rmse(real_gene_df, syn_cont)
    coexp_tpr = compute_coexpression_tpr(real_gene_df, syn_cont)
    mean_kl = _mean_pairwise_kl(real_binned_gene, syn_binned, feature_cols, n_bins)
    _print_metrics("baseline (no gene-gene cliques)", corr_mae, corr_rmse, coexp_tpr, mean_kl)

    kl_df = _all_pairwise_kl_df(real_binned_gene, syn_binned, feature_cols, n_bins)

    # Annotate with real Pearson correlation for reference
    real_corr = real_gene_df.corr(method="pearson").fillna(0)
    kl_df["real_corr"] = kl_df.apply(
        lambda r: real_corr.loc[r.gene_i, r.gene_j], axis=1)

    return kl_df.head(top_k), dict(
        corr_mae=corr_mae, corr_rmse=corr_rmse,
        coexp_tpr=coexp_tpr, mean_pairwise_kl=mean_kl)


# ---------------------------------------------------------------------------
# Method 4 — greedy  (iterative, minimises mean pairwise KL)
# ---------------------------------------------------------------------------

def method_greedy(real_gene_df, real_binned_gene, binned_df, config,
                  label_col, feature_cols, bin_means,
                  epsilon, delta, num_iters, top_k, eval_budget,
                  greedy_iters, n_bins, enable_privacy):
    """
    At each step add whichever candidate gene pair reduces mean pairwise KL
    the most.  Candidates are seeded in KL-descending order from a single
    baseline run.  Each step evaluates at most `eval_budget` candidates using
    `greedy_iters` inference iterations (fewer than the final training budget).
    """
    print("\nTraining baseline Private-PGM for greedy seed…")
    t0 = time.time()
    syn_cont, syn_binned = _train_and_generate(
        binned_df, config, label_col, feature_cols, bin_means,
        epsilon, delta, greedy_iters, enable_privacy=enable_privacy)
    print(f"  Baseline trained in {time.time()-t0:.1f}s")

    baseline_mae = compute_correlation_diff(real_gene_df, syn_cont)
    baseline_rmse = _correlation_rmse(real_gene_df, syn_cont)
    baseline_tpr = compute_coexpression_tpr(real_gene_df, syn_cont)
    baseline_kl = _mean_pairwise_kl(real_binned_gene, syn_binned, feature_cols, n_bins)
    _print_metrics("baseline", baseline_mae, baseline_rmse, baseline_tpr, baseline_kl)

    # Seed candidate order: pairs with worst KL first
    kl_df = _all_pairwise_kl_df(real_binned_gene, syn_binned, feature_cols, n_bins)
    candidates = [(r.gene_i, r.gene_j) for _, r in kl_df.iterrows()]

    chosen = []
    current_kl = baseline_kl

    print(f"\nGreedy selection (top_k={top_k}, eval_budget={eval_budget}, "
          f"greedy_iters={greedy_iters}, metric=mean_pairwise_kl):")

    for step in range(top_k):
        pool = [p for p in candidates if p not in chosen]
        if not pool:
            print("  No more candidates.")
            break

        best_pair, best_kl = None, current_kl

        for pair in pool[:eval_budget]:
            try:
                _, syn_trial_binned = _train_and_generate(
                    binned_df, config, label_col, feature_cols, bin_means,
                    epsilon, delta, greedy_iters,
                    gene_gene_cliques=chosen + [pair],
                    enable_privacy=enable_privacy)
                kl = _mean_pairwise_kl(
                    real_binned_gene, syn_trial_binned, feature_cols, n_bins)
            except Exception as exc:
                print(f"    Warning: failed to evaluate {pair}: {exc}")
                continue
            if kl < best_kl:
                best_kl, best_pair = kl, pair

        if best_pair is None:
            print(f"  Step {step+1}: no improvement found, stopping early.")
            break

        chosen.append(best_pair)
        candidates.remove(best_pair)
        print(f"  Step {step+1}: added {best_pair}  "
              f"mean_kl={best_kl:.4f}  (Δ={current_kl - best_kl:+.4f})")
        current_kl = best_kl

    result_df = pd.DataFrame(
        [{"gene_i": gi, "gene_j": gj} for gi, gj in chosen])
    return result_df, dict(
        baseline_corr_mae=baseline_mae,
        baseline_corr_rmse=baseline_rmse,
        baseline_coexp_tpr=baseline_tpr,
        baseline_mean_kl=baseline_kl,
        final_mean_kl=current_kl,
        kl_improvement=baseline_kl - current_kl,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Select gene-gene pairs for Private-PGM to maximise fidelity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", required=True,
                   help="Path to input CSV (continuous gene expression + label).")
    p.add_argument("--label-col", default="label",
                   help="Name of the class-label column (default: label).")
    p.add_argument("--method", choices=["pearson", "gap", "kl", "greedy"], default="kl",
                   help="Selection method (default: kl).")
    p.add_argument("--top-k", type=int, default=30,
                   help="Number of gene-gene pairs to recommend (default: 30).")
    p.add_argument("--n-bins", type=int, default=4,
                   help="Bins per gene for quantile discretisation (default: 4).")

    priv = p.add_argument_group("privacy (gap / kl / greedy only)")
    priv.add_argument("--epsilon", type=float, default=1.0)
    priv.add_argument("--delta", type=float, default=1e-5)
    priv.add_argument("--no-privacy", action="store_true",
                      help="Disable DP during selection (faster, for exploration).")
    priv.add_argument("--num-iters", type=int, default=1000,
                      help="FactoredInference iterations for baseline training "
                           "(default: 1000).")

    greedy = p.add_argument_group("greedy-only options")
    greedy.add_argument("--greedy-eval-budget", type=int, default=20,
                        help="Max candidates evaluated per greedy step (default: 20).")
    greedy.add_argument("--greedy-iters", type=int, default=500,
                        help="Inference iterations per candidate evaluation (default: 500).")

    p.add_argument("--output", default=None,
                   help="Save recommended pairs to this JSON file.")
    p.add_argument("--show-top", type=int, default=20,
                   help="Rows to print in the summary table (default: 20).")
    return p


def main():
    args = build_parser().parse_args()

    print(f"Loading data from {args.data} …")
    real_df, binned_df, bin_means, feature_cols, config = load_and_bin(
        args.data, args.label_col, args.n_bins)
    real_gene_df = real_df[feature_cols]
    real_binned_gene = binned_df[feature_cols]
    n_genes = len(feature_cols)
    n_pairs = n_genes * (n_genes - 1) // 2

    print(f"  {len(real_df)} samples  |  {n_genes} genes  |  {n_pairs} gene pairs total")
    print(f"  Method: {args.method}  |  top-k: {args.top_k}")

    enable_privacy = not args.no_privacy
    extra_info = {}

    if args.method == "pearson":
        result_df = method_pearson(real_gene_df, args.top_k)

    elif args.method == "gap":
        result_df, extra_info = method_gap(
            real_gene_df, real_binned_gene, binned_df, config,
            args.label_col, feature_cols, bin_means,
            args.epsilon, args.delta, args.num_iters,
            args.top_k, args.n_bins, enable_privacy)

    elif args.method == "kl":
        result_df, extra_info = method_kl(
            real_gene_df, real_binned_gene, binned_df, config,
            args.label_col, feature_cols, bin_means,
            args.epsilon, args.delta, args.num_iters,
            args.top_k, args.n_bins, enable_privacy)

    else:  # greedy
        result_df, extra_info = method_greedy(
            real_gene_df, real_binned_gene, binned_df, config,
            args.label_col, feature_cols, bin_means,
            args.epsilon, args.delta, args.num_iters,
            args.top_k, args.greedy_eval_budget, args.greedy_iters,
            args.n_bins, enable_privacy)

    # Print table
    print(f"\n{'='*70}")
    print(f"Top {min(args.show_top, len(result_df))} recommended pairs ({args.method})")
    print(f"{'='*70}")
    print(result_df.head(args.show_top).to_string(index=True))

    if extra_info:
        print("\nSummary statistics:")
        for k, v in extra_info.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save JSON
    pairs_list = [[r["gene_i"], r["gene_j"]] for _, r in result_df.iterrows()]
    output_data = {
        "method": args.method,
        "top_k": len(pairs_list),
        "n_genes": n_genes,
        "data": args.data,
        "label_col": args.label_col,
        "n_bins": args.n_bins,
        "pairs": pairs_list,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in extra_info.items()},
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nPairs saved to {args.output}")
    else:
        print("\nJSON (stdout):")
        print(json.dumps(output_data, indent=2))

    out_path = args.output or "recommended_pairs.json"
    print(f"\nTo use in Private-PGM:")
    print(f"  import json")
    print(f"  pairs = [tuple(p) for p in json.load(open('{out_path}'))['pairs']]")
    print(f"  model.train(binned_df, config, gene_gene_cliques=pairs)")


if __name__ == "__main__":
    main()
