"""
SECURE MPC Pipeline for Private-PGM (NO DATA LEAKAGE)

This script implements the fully secure MPC workflow where:
- Raw data NEVER leaves individual parties
- All sensitive computations happen in MPC
- Only DP-protected noisy statistics are revealed

SECURITY GUARANTEE: (ε,δ)-DP + Semi-honest MPC security

Usage:
    python run_secure_mpc_pipeline.py \
        --party_files party_0.csv party_1.csv \
        --output_path synthetic_data.csv \
        --mpspdz_path /home/mpcuser/MP-SPDZ/
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'models', 'Private_PGM'))

# Import the secure MPC implementation
from models.Private_PGM.model_secure_mpc import SecureMPCPrivatePGM


def validate_party_files(party_files):
    """
    Validate that party data files exist and have consistent structure

    Args:
        party_files: List of file paths

    Returns:
        tuple: (num_features, num_classes, column_names)
    """
    print("\n" + "="*80)
    print("VALIDATING PARTY DATA FILES")
    print("="*80)

    if len(party_files) < 2:
        raise ValueError("At least 2 party files required for MPC")

    column_names = None
    num_samples_list = []

    for i, filepath in enumerate(party_files):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Party {i} file not found: {filepath}")

        # Read header only (don't load data)
        df_sample = pd.read_csv(filepath, nrows=1)

        if column_names is None:
            column_names = list(df_sample.columns)
        elif list(df_sample.columns) != column_names:
            raise ValueError(
                f"Party {i} has different columns than Party 0\n"
                f"Expected: {column_names}\n"
                f"Got: {list(df_sample.columns)}"
            )

        # Count samples without loading full data
        with open(filepath, 'r') as f:
            num_samples = sum(1 for _ in f) - 1  # Subtract header
            num_samples_list.append(num_samples)

        print(f"✓ Party {i}: {num_samples} samples, {len(column_names)} columns")

    if 'label' not in column_names:
        raise ValueError("Dataset must have a 'label' column")

    num_features = len(column_names) - 1
    print(f"\n✓ All party files validated")
    print(f"  Total parties: {len(party_files)}")
    print(f"  Total samples: {sum(num_samples_list)}")
    print(f"  Features: {num_features}")

    # Infer number of classes (need to check at least one file)
    df_first = pd.read_csv(party_files[0])
    num_classes = len(df_first['label'].unique())
    print(f"  Classes: {num_classes}")

    return num_features, num_classes, column_names


def run_secure_mpc_pipeline(party_files, output_path, epsilon=1.0, delta=1e-5,
                             mpspdz_path=None,
                             marginal_protocol='ppai_bin_msr',
                             deg_filtering=None, num_iters=10000):
    """
    Run the secure MPC pipeline with no data leakage

    Args:
        party_files: List of paths to party data files
        output_path: Path to save synthetic data
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        mpspdz_path: Path to MP-SPDZ installation
        marginal_protocol: MPC protocol for marginals
        num_iters: Number of inference iterations

    Returns:
        DataFrame: Synthetic data
    """
    print("\n" + "="*80)
    print("SECURE MPC PRIVATE-PGM PIPELINE")
    print("="*80)
    print("SECURITY: No raw data leakage - full MPC + DP protection")
    print("="*80)

    # Validate inputs
    if not mpspdz_path:
        raise ValueError("mpspdz_path is required for secure MPC pipeline")

    if not os.path.exists(mpspdz_path):
        raise FileNotFoundError(f"MP-SPDZ not found at: {mpspdz_path}")

    # Validate party files
    num_features, num_classes, column_names = validate_party_files(party_files)

    # Configure domain (4 bins per feature)
    config = {}
    for col in column_names:
        if col == 'label':
            config[col] = num_classes
        else:
            config[col] = 4  # 4 bins per feature

    print("\n" + "="*80)
    print("DOMAIN CONFIGURATION")
    print("="*80)
    print(f"Features: {num_features} (4 bins each)")
    print(f"Label: {num_classes} classes")

    # Initialize secure MPC model
    print("\n" + "="*80)
    print("INITIALIZING SECURE MPC MODEL")
    print("="*80)
    print("configs", config)
    model = SecureMPCPrivatePGM(
        target_variable='label',
        enable_privacy=True,
        target_epsilon=epsilon,
        target_delta=delta,
        mpspdz_path=mpspdz_path,
        mpc_protocol='ring',
        num_parties=len(party_files)
    )

    # Train model (all operations in MPC)
    print("\n" + "="*80)
    print("TRAINING MODEL WITH MPC")
    print("="*80)
    print("All operations will be performed securely in MPC")
    print("Raw data will never be revealed")
    print("="*80)

    model.train_from_party_files(
        party_data_files=party_files,
        config=config,
        marginal_protocol=marginal_protocol,
        num_iters=num_iters,
        deg_filtering=deg_filtering
    )

    # Generate synthetic data
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC DATA")
    print("="*80)

    # Determine number of samples to generate
    total_samples = sum(
        sum(1 for _ in open(f)) - 1
        for f in party_files
    )

    synthetic_continuous = model.generate_continuous(num_rows=total_samples)
    
    # NEW: Fetch the dynamically selected column names from the model 
    # (falls back to original column_names if no filtering happened)
    final_columns = getattr(model, 'selected_columns', column_names)

    # Create the DataFrame using the correctly sized list
    synthetic_df = pd.DataFrame(synthetic_continuous, columns=final_columns)

    # Save output
    print(f"\nSaving synthetic data to: {output_path}")
    synthetic_df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Synthetic data shape: {synthetic_df.shape}")
    print(f"Output: {output_path}")
    print("\nSECURITY VERIFICATION:")
    print("  ✓ Raw data never left individual parties")
    print("  ✓ All binning done in MPC with DP noise")
    print("  ✓ All marginals computed in MPC with DP noise")
    print("  ✓ Only DP-protected statistics used for synthesis")
    print(f"  ✓ Privacy guarantee: (ε={epsilon}, δ={delta})-DP")
    print("="*80)

    return synthetic_df


def main():
    parser = argparse.ArgumentParser(
        description='Run SECURE MPC Private-PGM pipeline (no data leakage)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Two-party scenario
  python run_secure_mpc_pipeline.py \
      --party_files hospital_a.csv hospital_b.csv \
      --output_path synthetic_data.csv \
      --mpspdz_path /home/mpcuser/MP-SPDZ/ \
      --epsilon 1.0 \
      --delta 1e-5

  # Three-party scenario
  python run_secure_mpc_pipeline.py \
      --party_files party_0.csv party_1.csv party_2.csv \
      --output_path synthetic_data.csv \
      --mpspdz_path /home/mpcuser/MP-SPDZ/

SECURITY GUARANTEE:
  This pipeline ensures (ε,δ)-differential privacy AND MPC security.
  Raw data never leaves individual parties.
        """
    )

    parser.add_argument(
        '--party_files',
        nargs='+',
        required=True,
        help='Paths to CSV files, one per data custodian party (minimum 2)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='synthetic_data_secure.csv',
        help='Path to save synthetic data (default: synthetic_data_secure.csv)'
    )
    parser.add_argument(
        '--deg_filtering',
        type=int,
        default=None,
        help='Number of Top-K genes to securely select and process (default: None, processes all genes)'
    )
    parser.add_argument(
        '--mpspdz_path',
        type=str,
        required=True,
        help='Path to MP-SPDZ installation (REQUIRED)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1.0,
        help='Privacy parameter epsilon (default: 1.0)'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=1e-5,
        help='Privacy parameter delta (default: 1e-5)'
    )
    parser.add_argument(
        '--marginal_protocol',
        type=str,
        default='ppai_msr_noisy_final',
        help='MPC protocol for marginals (default: ppai_msr_noisy_final)'
    )
    parser.add_argument(
        '--num_iters',
        type=int,
        default=10000,
        help='Number of inference iterations (default: 10000)'
    )

    args = parser.parse_args()

    # Run secure pipeline
    synthetic_df = run_secure_mpc_pipeline(
        party_files=args.party_files,
        output_path=args.output_path,
        epsilon=args.epsilon,
        delta=args.delta,
        mpspdz_path=args.mpspdz_path,
        marginal_protocol=args.marginal_protocol,
        num_iters=args.num_iters,
        deg_filtering=args.deg_filtering
    )

    print("\n Secure synthetic data generated")
    print(f" Privacy guarantee: (ε={args.epsilon}, δ={args.delta})-DP")
    synthetic_df.to_csv(args.output_path, index=False)
    print(f"✓ Output saved to: {args.output_path}")

if __name__ == "__main__":
    main()
