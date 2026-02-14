#!/usr/bin/env python3
"""
Preprocess and split a dataset into N MPC parties with optional DEG selection.
Label column is always encoded and renamed to 'label'.
"""

import pandas as pd
import numpy as np
import argparse
import json
from sklearn.feature_selection import f_classif


def preprocess_and_split(
    input_file,
    output_prefix,
    n_parties,
    label_col,
    n_features=None,
    encoding_file="label_encoding.json",
    verbose=True
):
    print("=" * 80)
    print("PREPROCESSING & SPLITTING DATA FOR MPC")
    print("=" * 80)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(input_file)
    print(f"Loaded dataset: {df.shape}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    # =========================
    # EXTRACT & ENCODE LABELS
    # =========================
    y_raw = df[label_col]

    if y_raw.dtype == object or pd.api.types.is_string_dtype(y_raw):
        classes = sorted(y_raw.unique())
        label_encoder = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(label_encoder).astype(int)
        print(f"Encoded {len(classes)} classes")
    else:
        label_encoder = None
        y = y_raw.astype(int)

    # =========================
    # PROCESS FEATURES
    # =========================
    X = df.drop(columns=[label_col])

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"Dropping {len(non_numeric)} non-numeric columns")
        X = X.drop(columns=non_numeric)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # =========================
    # DEG FEATURE SELECTION
    # =========================
    selected_features = None
    if n_features is not None:
        print(f"Running DEG selection (top {n_features})")

        if X.shape[1] <= n_features:
            selected_features = X.columns.tolist()
            print("⚠ Feature count <= n_features, keeping all")
        else:
            f_scores, _ = f_classif(X.values, y.values)
            deg_df = (
                pd.DataFrame({"gene": X.columns, "f_score": f_scores})
                .sort_values("f_score", ascending=False)
            )
            selected_features = deg_df.head(n_features)["gene"].tolist()

        X = X[selected_features]
        print(f"Selected {len(selected_features)} features")

    # =========================
    # REASSEMBLE DATAFRAME
    # =========================
    df_proc = X.copy()
    df_proc["label"] = y  # <-- forced name

    # =========================
    # SPLIT INTO N PARTIES
    # =========================
    splits = np.array_split(
        df_proc.sample(frac=1, random_state=42),
        n_parties
    )

    for i, part in enumerate(splits, 1):
        out_file = f"{output_prefix}_party_{i}.csv"
        part.to_csv(out_file, index=False)
        print(f"✓ Saved {out_file} | shape={part.shape}")

    # =========================
    # SAVE METADATA
    # =========================
    if label_encoder:
        with open(encoding_file, "w") as f:
            json.dump(
                {
                    "label_to_int": label_encoder,
                    "int_to_label": {v: k for k, v in label_encoder.items()},
                    "num_classes": len(label_encoder),
                },
                f,
                indent=2,
            )

    if selected_features:
        with open("selected_degs.json", "w") as f:
            json.dump(selected_features, f, indent=2)

    print("=" * 80)
    print("DONE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--n_parties", type=int, required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--encoding_file", default="label_encoding.json")

    args = parser.parse_args()

    preprocess_and_split(
        input_file=args.input,
        output_prefix=args.output_prefix,
        n_parties=args.n_parties,
        label_col=args.label_col,
        n_features=args.n_features,
        encoding_file=args.encoding_file,
    )


if __name__ == "__main__":
    main()
