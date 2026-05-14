"""
select_gene_pairs.py — find the gene-gene pairs that most improve Private-PGM
fidelity when passed as gene_gene_cliques to model.train().

Three selection methods:

  pearson  Fast. Ranks every gene pair by |Pearson correlation| in the real
           data. No model training required. Use this as a quick starting
           point when training is expensive.

  gap      Recommended. Trains one baseline Private-PGM model (no gene-gene
           cliques), generates synthetic data, then ranks pairs by how much
           the baseline *fails* to reproduce each gene-gene correlation
           (|real_corr - syn_corr|). Pairs with the largest gap are the ones
           that benefit most from being explicitly measured.

  greedy   Iteratively adds pairs one at a time, keeping whichever candidate
           reduces correlation RMSE the most at each step. Expensive: trains
           `top_k` × `--greedy-eval-budget` models. Uses a reduced
           `--greedy-iters` during search to keep runtime manageable.

Usage examples:

  # Fast ranking from real-data correlations only
  python select_gene_pairs.py --data data/working_bcra_sample.csv \\
      --label-col label --method pearson --top-k 30

  # Gap method (one baseline training)
  python select_gene_pairs.py --data data/working_bcra_sample.csv \\
      --label-col label --method gap --epsilon 1.0 --delta 1e-5 \\
      --top-k 50 --output recommended_pairs.json

  # Greedy (expensive, most accurate)
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

# Make project modules importable regardless of working directory
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "models", "Private_PGM"))

from run_benchmark_pgm import (
    bin_dataframe,
    bins_to_continuous,
    compute_correlation_diff,
    compute_coexpression_tpr,
    compute_histogram_intersection,
)


# ---------------------------------------------------------------------------
# Fidelity helpers
# ---------------------------------------------------------------------------

def _correlation_rmse(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    """RMSE of pairwise Pearson correlations over upper triangle."""
    real_corr = real_df.corr(method="pearson").fillna(0).values
    syn_corr = syn_df.corr(method="pearson").fillna(0).values
    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    return float(np.sqrt(np.mean((real_corr[mask] - syn_corr[mask]) ** 2)))


def _per_pair_gap(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of every gene pair with real_corr and corr_gap columns."""
    cols = real_df.columns.tolist()
    real_corr = real_df.corr(method="pearson").fillna(0)
    syn_corr = syn_df.corr(method="pearson").fillna(0)

    rows = []
    for i, gi in enumerate(cols):
        for j in range(i + 1, len(cols)):
            gj = cols[j]
            rc = real_corr.loc[gi, gj]
            sc = syn_corr.loc[gi, gj]
            rows.append(
                {
                    "gene_i": gi,
                    "gene_j": gj,
                    "real_corr": rc,
                    "syn_corr": sc,
                    "corr_gap": abs(rc - sc),
                }
            )
    return pd.DataFrame(rows).sort_values("corr_gap", ascending=False).reset_index(drop=True)


def _print_metrics(label: str, corr_mae: float, corr_rmse: float, coexp_tpr: float) -> None:
    print(f"\n  [{label}]")
    print(f"    Correlation MAE      : {corr_mae:.4f}  (lower is better)")
    print(f"    Correlation RMSE     : {corr_rmse:.4f}  (lower is better)")
    print(f"    Coexpression TPR     : {coexp_tpr:.4f}  (higher is better, threshold=0.7)")


# ---------------------------------------------------------------------------
# Data loading + binning
# ---------------------------------------------------------------------------

def load_and_bin(data_path: str, label_col: str, n_bins: int):
    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in {data_path}. "
                         f"Columns: {list(df.columns)[:10]} …")

    feature_cols = [c for c in df.columns if c != label_col]

    # Encode label to integers if needed
    if df[label_col].dtype == object or not pd.api.types.is_integer_dtype(df[label_col]):
        df[label_col] = pd.Categorical(df[label_col]).codes

    binned_df, bin_means = bin_dataframe(df, label_col, n_bins=n_bins)

    config = {col: n_bins for col in feature_cols}
    config[label_col] = int(df[label_col].nunique())

    return df, binned_df, bin_means, feature_cols, config


# ---------------------------------------------------------------------------
# Model training + synthetic generation helper
# ---------------------------------------------------------------------------

def _train_and_generate(binned_df, config, label_col, feature_cols, bin_means,
                        epsilon, delta, num_iters, gene_gene_cliques=None,
                        enable_privacy=True):
    from models.Private_PGM.model import Private_PGM

    pgm = Private_PGM(
        target_variable=label_col,
        enable_privacy=enable_privacy,
        target_epsilon=epsilon,
        target_delta=delta,
    )
    pgm.train(
        binned_df,
        config,
        gene_gene_cliques=gene_gene_cliques or [],
        num_iters=num_iters,
    )
    syn_array = pgm.generate(num_rows=len(binned_df))
    syn_cont = bins_to_continuous(syn_array, feature_cols, label_col, bin_means)
    return syn_cont[feature_cols]  # return only gene columns (continuous)


# ---------------------------------------------------------------------------
# Method 1 — pearson
# ---------------------------------------------------------------------------

def method_pearson(real_gene_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Rank all gene pairs by |Pearson correlation| in the real data."""
    cols = real_gene_df.columns.tolist()
    corr = real_gene_df.corr(method="pearson").fillna(0)

    rows = []
    for i, gi in enumerate(cols):
        for j in range(i + 1, len(cols)):
            gj = cols[j]
            rows.append({"gene_i": gi, "gene_j": gj, "abs_corr": abs(corr.loc[gi, gj])})

    df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return df.head(top_k)


# ---------------------------------------------------------------------------
# Method 2 — gap
# ---------------------------------------------------------------------------

def method_gap(real_gene_df, binned_df, config, label_col, feature_cols,
               bin_means, epsilon, delta, num_iters, top_k, enable_privacy):
    """Train baseline, then rank pairs by how poorly the baseline preserves them."""
    print("\nTraining baseline Private-PGM (no gene-gene cliques)…")
    t0 = time.time()
    syn_gene_df = _train_and_generate(
        binned_df, config, label_col, feature_cols, bin_means,
        epsilon, delta, num_iters, gene_gene_cliques=None,
        enable_privacy=enable_privacy,
    )
    print(f"  Baseline trained in {time.time()-t0:.1f}s")

    corr_mae = compute_correlation_diff(real_gene_df, syn_gene_df)
    corr_rmse = _correlation_rmse(real_gene_df, syn_gene_df)
    coexp_tpr = compute_coexpression_tpr(real_gene_df, syn_gene_df)
    _print_metrics("baseline (no gene-gene cliques)", corr_mae, corr_rmse, coexp_tpr)

    gap_df = _per_pair_gap(real_gene_df, syn_gene_df)
    return gap_df.head(top_k), dict(
        corr_mae=corr_mae, corr_rmse=corr_rmse, coexp_tpr=coexp_tpr
    )


# ---------------------------------------------------------------------------
# Method 3 — greedy
# ---------------------------------------------------------------------------

def method_greedy(real_gene_df, binned_df, config, label_col, feature_cols,
                  bin_means, epsilon, delta, num_iters, top_k, eval_budget,
                  greedy_iters, enable_privacy):
    """
    Greedy pair selection: at each step add whichever candidate gene pair
    reduces correlation RMSE the most.

    Candidates are seeded from a gap-ranked baseline run, and each step
    only evaluates the top `eval_budget` remaining candidates to bound cost.
    Model selection uses `greedy_iters` (fewer iterations) for speed; the
    caller should retrain with full `num_iters` using the returned pairs.
    """
    # --- baseline (needed for gap seed + initial metrics) ---
    print("\nTraining baseline Private-PGM for greedy seed…")
    t0 = time.time()
    syn_gene_df = _train_and_generate(
        binned_df, config, label_col, feature_cols, bin_means,
        epsilon, delta, greedy_iters, gene_gene_cliques=None,
        enable_privacy=enable_privacy,
    )
    print(f"  Baseline trained in {time.time()-t0:.1f}s")

    baseline_mae = compute_correlation_diff(real_gene_df, syn_gene_df)
    baseline_rmse = _correlation_rmse(real_gene_df, syn_gene_df)
    baseline_tpr = compute_coexpression_tpr(real_gene_df, syn_gene_df)
    _print_metrics("baseline", baseline_mae, baseline_rmse, baseline_tpr)

    # Seed candidate order from gap ranking
    gap_df = _per_pair_gap(real_gene_df, syn_gene_df)
    candidates = [(row.gene_i, row.gene_j) for _, row in gap_df.iterrows()]

    chosen = []
    current_rmse = baseline_rmse

    print(f"\nGreedy selection (top_k={top_k}, eval_budget={eval_budget}, "
          f"greedy_iters={greedy_iters}):")

    for step in range(top_k):
        pool = [p for p in candidates if p not in chosen]
        if not pool:
            print("  No more candidates.")
            break

        best_pair = None
        best_rmse = current_rmse

        for pair in pool[:eval_budget]:
            trial = chosen + [pair]
            try:
                syn_trial = _train_and_generate(
                    binned_df, config, label_col, feature_cols, bin_means,
                    epsilon, delta, greedy_iters,
                    gene_gene_cliques=trial,
                    enable_privacy=enable_privacy,
                )
                rmse = _correlation_rmse(real_gene_df, syn_trial)
            except Exception as exc:
                print(f"    Warning: failed to evaluate {pair}: {exc}")
                continue

            if rmse < best_rmse:
                best_rmse = rmse
                best_pair = pair

        if best_pair is None:
            print(f"  Step {step+1}: no improvement found, stopping early.")
            break

        chosen.append(best_pair)
        candidates.remove(best_pair)
        delta_rmse = current_rmse - best_rmse
        current_rmse = best_rmse
        print(f"  Step {step+1}: added {best_pair}  "
              f"corr_rmse={current_rmse:.4f}  (Δ={delta_rmse:+.4f})")

    rows = [{"gene_i": gi, "gene_j": gj} for gi, gj in chosen]
    result_df = pd.DataFrame(rows)
    return result_df, dict(
        baseline_corr_mae=baseline_mae,
        baseline_corr_rmse=baseline_rmse,
        baseline_coexp_tpr=baseline_tpr,
        final_corr_rmse=current_rmse,
        rmse_improvement=baseline_rmse - current_rmse,
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
    p.add_argument("--method", choices=["pearson", "gap", "greedy"], default="gap",
                   help="Selection method (default: gap).")
    p.add_argument("--top-k", type=int, default=30,
                   help="Number of gene-gene pairs to recommend (default: 30).")
    p.add_argument("--n-bins", type=int, default=4,
                   help="Bins per gene for quantile discretisation (default: 4).")

    # Privacy parameters (used by gap / greedy)
    priv = p.add_argument_group("privacy (gap / greedy only)")
    priv.add_argument("--epsilon", type=float, default=1.0,
                      help="DP epsilon (default: 1.0).")
    priv.add_argument("--delta", type=float, default=1e-5,
                      help="DP delta (default: 1e-5).")
    priv.add_argument("--no-privacy", action="store_true",
                      help="Disable differential privacy during selection (faster).")
    priv.add_argument("--num-iters", type=int, default=1000,
                      help="FactoredInference iterations for gap/greedy baseline "
                           "(default: 1000; use fewer for speed).")

    # Greedy-only
    greedy = p.add_argument_group("greedy-only options")
    greedy.add_argument("--greedy-eval-budget", type=int, default=20,
                        help="Max candidates evaluated per greedy step (default: 20).")
    greedy.add_argument("--greedy-iters", type=int, default=500,
                        help="Inference iterations per candidate evaluation (default: 500).")

    # Output
    p.add_argument("--output", default=None,
                   help="Save recommended pairs to this JSON file.")
    p.add_argument("--show-top", type=int, default=20,
                   help="Rows to print in the summary table (default: 20).")
    return p


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------
    # Load + bin data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.data} …")
    real_df, binned_df, bin_means, feature_cols, config = load_and_bin(
        args.data, args.label_col, args.n_bins
    )
    real_gene_df = real_df[feature_cols]
    n_genes = len(feature_cols)
    n_pairs = n_genes * (n_genes - 1) // 2

    print(f"  {len(real_df)} samples  |  {n_genes} genes  |  {n_pairs} gene pairs total")
    print(f"  Method: {args.method}  |  top-k: {args.top_k}")

    enable_privacy = not args.no_privacy

    # ------------------------------------------------------------------
    # Run selected method
    # ------------------------------------------------------------------
    extra_info = {}

    if args.method == "pearson":
        result_df = method_pearson(real_gene_df, args.top_k)
        score_col = "abs_corr"

    elif args.method == "gap":
        result_df, extra_info = method_gap(
            real_gene_df, binned_df, config, args.label_col, feature_cols,
            bin_means, args.epsilon, args.delta, args.num_iters,
            args.top_k, enable_privacy,
        )
        score_col = "corr_gap"

    else:  # greedy
        result_df, extra_info = method_greedy(
            real_gene_df, binned_df, config, args.label_col, feature_cols,
            bin_means, args.epsilon, args.delta, args.num_iters,
            args.top_k, args.greedy_eval_budget, args.greedy_iters,
            enable_privacy,
        )
        score_col = None  # greedy result is already ordered by selection

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Top {min(args.show_top, len(result_df))} recommended gene-gene pairs ({args.method} method)")
    print(f"{'='*70}")
    print(result_df.head(args.show_top).to_string(index=True))

    if extra_info:
        print(f"\nSummary statistics:")
        for k, v in extra_info.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Save output JSON
    # ------------------------------------------------------------------
    pairs_list = [
        [row["gene_i"], row["gene_j"]]
        for _, row in result_df.iterrows()
    ]

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

    print(f"\nTo use these pairs in Private-PGM:")
    print(f"  import json")
    print(f"  pairs = [tuple(p) for p in json.load(open('{args.output or 'recommended_pairs.json'}'))['pairs']]")
    print(f"  model.train(binned_df, config, gene_gene_cliques=pairs)")


if __name__ == "__main__":
    main()
