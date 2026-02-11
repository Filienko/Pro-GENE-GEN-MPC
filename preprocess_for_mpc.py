#!/usr/bin/env python3
"""
Preprocess data for MPC pipeline with optional DEG feature selection

This script:
1. Encodes string labels to integer class indices
2. Ensures all features are numeric
3. Optionally performs DEG analysis (ANOVA F-test) to select top N features
4. Saves encoding mappings and DEG lists
5. Validates data format

Usage:
    python preprocess_for_mpc.py \
        --input party_1.csv party_2.csv \
        --output party_1_preprocessed.csv party_2_preprocessed.csv \
        --label_col label \
        --n_features 1000
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys

from sklearn.feature_selection import f_classif


def preprocess_party_file(
    input_file,
    output_file,
    label_col='label',
    label_encoder=None,
    n_features=None,
    selected_features=None,
    verbose=True
):
    """
    Preprocess a single party's data file

    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        label_col: Name of label column
        label_encoder: Optional pre-fitted encoder (shared across parties)
        n_features: Number of DEGs to keep (optional)
        selected_features: Precomputed DEG list (for parties > 1)
        verbose: Print progress messages

    Returns:
        dict: Metadata including encoders and selected features
    """
    if verbose:
        print(f"\nProcessing: {input_file}")
        print("-" * 80)

    # Load data
    df = pd.read_csv(input_file)
    if verbose:
        print(f"  Loaded: {len(df)} samples, {len(df.columns)} columns")

    # Separate features and labels
    if label_col in df.columns:
        features = df.drop(label_col, axis=1)
        labels = df[label_col]
    else:
        features = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        label_col = df.columns[-1]
        if verbose:
            print(f"  ⚠️  Label column not found, using last column: {label_col}")

    # Encode labels if needed
    if pd.api.types.is_object_dtype(labels) or pd.api.types.is_string_dtype(labels):
        if label_encoder is None:
            unique_labels = sorted(labels.unique())
            label_encoder = {lab: i for i, lab in enumerate(unique_labels)}
            if verbose:
                print(f"  Creating label encoder ({len(label_encoder)} classes)")

        labels = labels.map(label_encoder)

        if labels.isna().any():
            unmapped = labels[labels.isna()].index.tolist()
            raise ValueError(f"Unmapped labels found: {unmapped}")

        labels = labels.astype(int)
    else:
        label_encoder = None

    # Drop non-numeric features
    non_numeric_cols = [
        col for col in features.columns
        if not pd.api.types.is_numeric_dtype(features[col])
    ]

    if non_numeric_cols and verbose:
        print(f"  Dropping {len(non_numeric_cols)} non-numeric columns")

    features = features.drop(columns=non_numeric_cols)

    # Force numeric
    features = features.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs
    if features.isna().any().any():
        if verbose:
            print("  Filling NaNs with 0")
        features = features.fillna(0)

    # =========================
    # DEG FEATURE SELECTION
    # =========================
    if n_features is not None:
        if selected_features is None:
            if verbose:
                print(f"  Performing DEG analysis (ANOVA F-test)")
                print(f"  Selecting top {n_features} DEGs")

            X = features.values
            y = labels.values

            if X.shape[1] <= n_features:
                selected_features = features.columns.tolist()
                if verbose:
                    print("  ⚠️  Feature count <= n_features, keeping all")
            else:
                f_scores, pvals = f_classif(X, y)

                deg_df = pd.DataFrame({
                    "gene": features.columns,
                    "f_score": f_scores,
                    "pval": pvals
                }).sort_values("f_score", ascending=False)

                selected_features = deg_df.head(n_features)["gene"].tolist()

                if verbose:
                    print(f"  ✓ Selected {len(selected_features)} DEGs")
                    print(f"    Example genes: {selected_features[:5]}")

        # Apply selected DEG set
        features = features[selected_features]

    # Reconstruct dataframe
    processed_df = features.copy()
    processed_df[label_col] = labels

    # Save output
    processed_df.to_csv(output_file, index=False)

    if verbose:
        print(f"  ✓ Saved: {output_file}")
        print(f"     Shape: {processed_df.shape}")
        print(f"     Features: {features.shape[1]}")
        print(f"     Classes: {labels.nunique()}")

    return {
        "label_encoder": label_encoder,
        "selected_features": selected_features,
        "dropped_features": non_numeric_cols,
        "num_classes": int(labels.nunique()),
        "class_range": [int(labels.min()), int(labels.max())]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data files for MPC pipeline (with DEG selection)"
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", nargs="+", required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--encoding_file", default="label_encoding.json")
    parser.add_argument("--n_features", type=int, default=None)

    args = parser.parse_args()

    if len(args.input) != len(args.output):
        print("ERROR: Number of input files must match output files")
        sys.exit(1)

    print("=" * 80)
    print("PREPROCESSING DATA FOR MPC PIPELINE")
    print("=" * 80)

    # First party: create encoder + DEG list
    metadata = preprocess_party_file(
        args.input[0],
        args.output[0],
        label_col=args.label_col,
        label_encoder=None,
        n_features=args.n_features,
        selected_features=None,
        verbose=True
    )

    label_encoder = metadata["label_encoder"]
    selected_features = metadata["selected_features"]

    # Remaining parties
    for inp, out in zip(args.input[1:], args.output[1:]):
        preprocess_party_file(
            inp,
            out,
            label_col=args.label_col,
            label_encoder=label_encoder,
            n_features=args.n_features,
            selected_features=selected_features,
            verbose=True
        )

    # Save label encoding
    if label_encoder:
        with open(args.encoding_file, "w") as f:
            json.dump({
                "label_to_int": label_encoder,
                "int_to_label": {v: k for k, v in label_encoder.items()},
                "num_classes": metadata["num_classes"]
            }, f, indent=2)

    # Save DEG list
    if selected_features:
        with open("selected_degs.json", "w") as f:
            json.dump(selected_features, f, indent=2)

    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"✓ Files processed: {len(args.input)}")
    print(f"✓ Features used: {len(selected_features) if selected_features else 'ALL'}")
    print(f"✓ Classes: {metadata['num_classes']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
