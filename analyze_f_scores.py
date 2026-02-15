import pandas as pd
import numpy as np
import argparse
from sklearn.feature_selection import f_classif

def analyze_f_scores(input_file, label_col):
    print("=" * 60)
    print(f"ANALYZING F-SCORES FOR: {input_file}")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(input_file)
    df_numeric = df.select_dtypes(include="number")
    print(f"min:{df_numeric.min().min()}, max:{df_numeric.max().max()}")
    print(f"75th %:   {df_numeric.quantile(0.75).max():.4f}")
    print(f"90th %:   {df_numeric.quantile(0.90).max():.4f}")
    print(f"95th %:   {df_numeric.quantile(0.95).max():.4f}")
    print(f"99th %:   {df_numeric.quantile(0.99).max():.4f}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    # 2. Extract Features and Labels
    y_raw = df[label_col]
    if y_raw.dtype == object or pd.api.types.is_string_dtype(y_raw):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    X = df.drop(columns=[label_col])
    numeric_cols = X.select_dtypes(include=['number']).columns
    X = X[numeric_cols].fillna(0)

    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")

    # 3. Calculate F-Scores
    f_scores, p_values = f_classif(X.values, y.values)
    
    # Handle any NaNs (genes with zero variance)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # 4. Display Distribution
    f_series = pd.Series(f_scores)
    
    print("\n--- F-Score Distribution ---")
    print(f"Min:      {f_series.min():.4f}")
    print(f"Mean:     {f_series.mean():.4f}")
    print(f"Median:   {f_series.median():.4f}")
    print(f"75th %:   {f_series.quantile(0.75):.4f}")
    print(f"90th %:   {f_series.quantile(0.90):.4f}")
    print(f"95th %:   {f_series.quantile(0.95):.4f}")
    print(f"99th %:   {f_series.quantile(0.99):.4f}")
    print(f"Max:      {f_series.max():.4f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("Choose a clipping bound (F_max) around the 95th or 99th percentile.")
    print("Genes with F-scores above this will be capped, but they will still")
    print("have the highest probability of being selected by the DP mechanism.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV data")
    parser.add_argument("--label_col", required=True, help="Name of the target/label column")
    args = parser.parse_args()
    
    analyze_f_scores(args.input, args.label_col)

