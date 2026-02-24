import pandas as pd
import numpy as np
import argparse


def analyze_quantile_distribution(csv_path, n_bins=10):
    # Load data
    df = pd.read_csv(csv_path)

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return

    print(f"\nAnalyzing {len(numeric_cols)} numeric columns with {n_bins} quantile bins\n")

    results = []

    for col in numeric_cols:
        print(f"\n--- Column: {col} ---")

        series = df[col].dropna()

        # Perform quantile binning
        try:
            bins = pd.qcut(series, q=n_bins, duplicates="drop")
        except ValueError:
            print("  Skipping (not enough unique values)")
            continue

        # Bin counts
        counts = bins.value_counts().sort_index()

        # Bin means
        bin_means = series.groupby(bins).mean().sort_index()

        # Differences between adjacent bin means
        mean_diffs = np.diff(bin_means.values)

        # Coefficient of variation of differences
        if len(mean_diffs) > 0:
            cv = np.std(mean_diffs) / np.mean(mean_diffs)
        else:
            cv = np.nan

        print("  Bin counts:")
        print(counts.values)

        print("  Bin means:")
        print(np.round(bin_means.values, 4))

        print("  Mean differences between bins:")
        print(np.round(mean_diffs, 4))

        print(f"  Coefficient of variation of mean diffs: {cv:.4f}")

        results.append({
            "column": col,
            "mean_diff_cv": cv,
            "min_mean_diff": np.min(mean_diffs) if len(mean_diffs) > 0 else np.nan,
            "max_mean_diff": np.max(mean_diffs) if len(mean_diffs) > 0 else np.nan
        })

    summary_df = pd.DataFrame(results)

    print("\n================ SUMMARY ================\n")
    print(summary_df.sort_values("mean_diff_cv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--bins", type=int, default=10, help="Number of quantile bins")
    args = parser.parse_args()

    analyze_quantile_distribution(args.csv, args.bins)

