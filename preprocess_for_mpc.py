#!/usr/bin/env python3
"""
Preprocess data for MPC pipeline

This script:
1. Encodes string labels to integer class indices
2. Ensures all features are numeric
3. Saves encoding mappings for later decoding
4. Validates data format

Usage:
    python preprocess_for_mpc.py \
        --input party_1.csv party_2.csv \
        --output party_1_preprocessed.csv party_2_preprocessed.csv \
        --label_col label
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys


def preprocess_party_file(input_file, output_file, label_col='label',
                          label_encoder=None, verbose=True):
    """
    Preprocess a single party's data file

    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        label_col: Name of label column
        label_encoder: Optional pre-fitted encoder (for consistency across parties)
        verbose: Print progress messages

    Returns:
        dict: Encoding mappings
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
        has_labels = True
    else:
        # Try assuming last column is label
        features = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        label_col = df.columns[-1]
        has_labels = True
        if verbose:
            print(f"  ⚠️  No '{label_col}' column found, using last column: {label_col}")

    # Check label types
    if pd.api.types.is_string_dtype(labels) or pd.api.types.is_object_dtype(labels):
        if verbose:
            print(f"  Labels are strings: {labels.dtype}")
            print(f"  Unique labels: {sorted(labels.unique())}")

        # Create or use encoder
        if label_encoder is None:
            unique_labels = sorted(labels.unique())
            label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
            if verbose:
                print(f"  Creating encoder for {len(label_encoder)} classes")

        # Encode labels
        encoded_labels = labels.map(label_encoder)

        # Check for unmapped labels
        if encoded_labels.isna().any():
            unmapped = labels[encoded_labels.isna()].unique()
            print(f"  ❌ ERROR: Found unmapped labels: {unmapped}")
            raise ValueError(f"Labels not in encoder: {unmapped}")

        labels = encoded_labels.astype(int)

        if verbose:
            print(f"  ✓ Encoded labels to integers: {labels.min()}-{labels.max()}")
    else:
        if verbose:
            print(f"  Labels already numeric: {labels.dtype}")
        label_encoder = None

    # Check features for non-numeric columns
    non_numeric_cols = []
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        if verbose:
            print(f"  ⚠️  Found {len(non_numeric_cols)} non-numeric feature columns")
            print(f"     Dropping: {non_numeric_cols[:5]}...")
        features = features.drop(columns=non_numeric_cols)

    # Convert all features to numeric (handle any edge cases)
    features = features.apply(pd.to_numeric, errors='coerce')

    # Check for NaN values
    if features.isna().any().any():
        nan_counts = features.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if verbose:
            print(f"  ⚠️  Found NaN values in {len(nan_cols)} columns")
            print(f"     Filling with 0...")
        features = features.fillna(0)

    # Reconstruct dataframe
    processed_df = features.copy()
    processed_df[label_col] = labels

    # Save
    processed_df.to_csv(output_file, index=False)

    if verbose:
        print(f"  ✓ Saved: {output_file}")
        print(f"     Shape: {processed_df.shape}")
        print(f"     Features: {len(features.columns)} (all numeric)")
        print(f"     Classes: {labels.nunique()} ({labels.min()}-{labels.max()})")

    return {
        'label_encoder': label_encoder,
        'dropped_features': non_numeric_cols,
        'num_classes': int(labels.nunique()),
        'class_range': [int(labels.min()), int(labels.max())]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess data files for MPC pipeline'
    )
    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Input CSV files (one per party)'
    )
    parser.add_argument(
        '--output',
        nargs='+',
        required=True,
        help='Output CSV files (one per party)'
    )
    parser.add_argument(
        '--label_col',
        default='label',
        help='Name of label column (default: label)'
    )
    parser.add_argument(
        '--encoding_file',
        default='label_encoding.json',
        help='File to save label encoding mapping (default: label_encoding.json)'
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.input) != len(args.output):
        print("ERROR: Number of input files must match number of output files")
        sys.exit(1)

    print("=" * 80)
    print("PREPROCESSING DATA FOR MPC PIPELINE")
    print("=" * 80)
    print(f"Input files: {len(args.input)}")
    print(f"Label column: {args.label_col}")
    print("=" * 80)

    # Process first file to create encoder
    print("\nSTEP 1: Processing first file to create encoder")
    metadata = preprocess_party_file(
        args.input[0],
        args.output[0],
        label_col=args.label_col,
        label_encoder=None,
        verbose=True
    )

    label_encoder = metadata['label_encoder']

    # Process remaining files with same encoder
    if len(args.input) > 1:
        print(f"\nSTEP 2: Processing remaining {len(args.input) - 1} files")
        for input_file, output_file in zip(args.input[1:], args.output[1:]):
            party_metadata = preprocess_party_file(
                input_file,
                output_file,
                label_col=args.label_col,
                label_encoder=label_encoder,
                verbose=True
            )

            # Verify consistent class counts
            if party_metadata['num_classes'] != metadata['num_classes']:
                print(f"\n⚠️  WARNING: Class count mismatch!")
                print(f"   First party: {metadata['num_classes']} classes")
                print(f"   This party: {party_metadata['num_classes']} classes")

    # Save encoding
    if label_encoder:
        with open(args.encoding_file, 'w') as f:
            # Convert to serializable format
            encoding_data = {
                'label_to_int': label_encoder,
                'int_to_label': {v: k for k, v in label_encoder.items()},
                'num_classes': metadata['num_classes']
            }
            json.dump(encoding_data, f, indent=2)

        print("\n" + "=" * 80)
        print("ENCODING MAPPING")
        print("=" * 80)
        for label, idx in sorted(label_encoder.items(), key=lambda x: x[1]):
            print(f"  {label:30s} -> {idx}")
        print(f"\n✓ Encoding saved to: {args.encoding_file}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"✓ Processed {len(args.input)} files")
    print(f"✓ All features are now numeric")
    print(f"✓ Labels encoded as integers: 0-{metadata['num_classes']-1}")
    print("\nYou can now run the MPC pipeline with the preprocessed files:")
    print(f"\n  python run_secure_mpc_pipeline.py \\")
    print(f"      --party_files {' '.join(args.output)} \\")
    print(f"      --output_path synthetic_mpc.csv \\")
    print(f"      --epsilon 10.0 \\")
    print(f"      --delta 1e-5")
    print("=" * 80)


if __name__ == '__main__':
    main()
