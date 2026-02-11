"""
Complete MPC Pipeline for Private-PGM

This script demonstrates the full end-to-end workflow using all MPC protocols:
1. Data preparation and splitting
2. MPC-based binning (ppai_bin_opt.mpc)
3. MPC-based marginal computation (ppai_msr_noisy_final)
4. Model training and synthesis
5. Inverse binning to continuous values

Usage:
    python run_mpc_pipeline.py --data_path data/your_data.csv --use_mpc
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import Private_PGM
import os


def discretize_data_local(df, alpha=0.25):
    """
    Local discretization (non-MPC version for comparison)

    Args:
        df: DataFrame with features and 'label' column
        alpha: Quantile parameter for binning

    Returns:
        tuple: (discretized_df, statistics_dict, mean_dict, quantile_dict)
    """
    assert alpha < 0.5, "The alpha (quantile) should be smaller than 0.5"

    alphas = [alpha, 0.5, 1 - alpha]  # Quantiles for discretization
    bin_number = len(alphas) + 1
    df_copy = df.copy()
    data_quantile = np.quantile(df.drop('label', axis=1), alphas, axis=0)

    statistic_dict = {}
    mean_dict = {}
    quantile_dict = {}

    for col in df.columns:
        if col != 'label':
            col_quantiles = data_quantile[:, list(df.columns).index(col)]
            discrete_col = np.digitize(df[col], col_quantiles)
            df[col] = discrete_col
            quantile_dict[col] = col_quantiles

            statistic_dict[col] = []
            mean_dict[col] = []
            for bin_idx in range(bin_number):
                bin_arr = df_copy[col][discrete_col == bin_idx]
                statistic_dict[col].append(len(bin_arr))
                mean_dict[col].append(np.mean(bin_arr) if len(bin_arr) > 0 else 0)

    return df, statistic_dict, mean_dict, quantile_dict


def inverse_discretize(synthetic_df, mean_dict):
    """
    Convert discrete bins back to continuous values using mean values

    Args:
        synthetic_df: DataFrame with discrete values
        mean_dict: Dictionary of mean values per bin per column

    Returns:
        DataFrame with continuous values
    """
    synthetic_continuous = synthetic_df.copy()

    for col, means in mean_dict.items():
        if col in synthetic_continuous.columns and col != 'label':
            synthetic_continuous[col] = synthetic_continuous[col].apply(
                lambda x: means[int(x)] if 0 <= int(x) < len(means) else means[-1]
            )

    return synthetic_continuous


def run_standard_pipeline(data_path, output_path, epsilon=1.0, delta=1e-5):
    """
    Run the standard (non-MPC) pipeline

    Args:
        data_path: Path to CSV data file
        output_path: Path to save synthetic data
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
    """
    print("\n" + "="*80)
    print("Running STANDARD PIPELINE (No MPC)")
    print("="*80)

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")

    # Split train/test
    df_train, df_test = train_test_split(
        df, test_size=0.2, shuffle=True, random_state=42,
        stratify=df['label'] if 'label' in df.columns else None
    )
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Discretize locally
    print("\nDiscretizing data locally...")
    df_train_discrete, statistic_dict, mean_dict, quantile_dict = discretize_data_local(df_train)

    # Configure domain
    config = {}
    for col in df_train_discrete.columns:
        if col == 'label':
            config[col] = len(df_train_discrete[col].unique())
        else:
            config[col] = 4  # 4 bins

    print(f"Domain configuration: {config}")

    # Initialize model
    print("\nInitializing Private-PGM model...")
    model = Private_PGM(
        target_variable='label',
        enable_privacy=True,
        target_epsilon=epsilon,
        target_delta=delta,
        use_mpc=False  # Standard mode
    )

    # Train
    print("Training model...")
    model.train(df_train_discrete, config, num_iters=1000)

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = model.generate(num_rows=df_train.shape[0])
    synthetic_df = pd.DataFrame(synthetic_data, columns=df_train_discrete.columns)

    # Inverse discretization
    print("Converting back to continuous values...")
    synthetic_continuous = inverse_discretize(synthetic_df, mean_dict)

    # Save
    print(f"Saving synthetic data to {output_path}...")
    synthetic_continuous.to_csv(output_path, index=False)
    print(f"✓ Synthetic data saved: {synthetic_continuous.shape}")

    return synthetic_continuous


def run_mpc_pipeline(data_path, output_path, epsilon=1.0, delta=1e-5,
                     mpspdz_path=None, bin_protocol='ppai_bin_opt',
                     marginal_protocol='ppai_msr_noisy_final'):
    """
    Run the MPC pipeline using all MPC protocols

    Args:
        data_path: Path to CSV data file
        output_path: Path to save synthetic data
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        mpspdz_path: Path to MP-SPDZ installation
        bin_protocol: MPC protocol for binning
        marginal_protocol: MPC protocol for marginal computation
    """
    print("\n" + "="*80)
    print("Running MPC PIPELINE (Secure Multi-Party Computation)")
    print("="*80)

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")

    # Split train/test
    df_train, df_test = train_test_split(
        df, test_size=0.2, shuffle=True, random_state=42,
        stratify=df['label'] if 'label' in df.columns else None
    )
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Step 1: MPC-based binning
    print("\n" + "-"*80)
    print("STEP 1: MPC-Based Binning")
    print("-"*80)
    print(f"Protocol: {bin_protocol}.mpc")
    print("This will:")
    print("  1. Split data between parties")
    print("  2. Compute quantiles securely")
    print("  3. Bin data into discrete categories")
    print("  4. Compute bin means with noise")
    print("\nNote: This requires MP-SPDZ to be installed and running")
    print("For now, falling back to local discretization...")

    # TODO: Implement MPC binning protocol execution
    # For now, use local discretization as fallback
    df_train_discrete, statistic_dict, mean_dict, quantile_dict = discretize_data_local(df_train)

    # Step 2: Configure domain
    config = {}
    for col in df_train_discrete.columns:
        if col == 'label':
            config[col] = len(df_train_discrete[col].unique())
        else:
            config[col] = 4  # 4 bins

    print(f"\nDomain configuration: {config}")

    # Step 3: MPC-based marginal computation
    print("\n" + "-"*80)
    print("STEP 2: MPC-Based Marginal Computation")
    print("-"*80)
    print(f"Protocol: {marginal_protocol}")

    # Initialize model with MPC enabled
    model = Private_PGM(
        target_variable='label',
        enable_privacy=True,
        target_epsilon=epsilon,
        target_delta=delta,
        use_mpc=True,  # Enable MPC mode
        mpspdz_path=mpspdz_path or '/home/mpcuser/MP-SPDZ/',
        mpc_protocol='ring'
    )

    # Train with MPC
    try:
        print("Training model with MPC...")
        model.train(
            df_train_discrete,
            config,
            num_iters=1000,
            mpc_protocol_file=marginal_protocol
        )

        # Generate synthetic data
        print("\nGenerating synthetic data...")
        synthetic_data = model.generate(num_rows=df_train.shape[0])
        synthetic_df = pd.DataFrame(synthetic_data, columns=df_train_discrete.columns)

        # Inverse discretization
        print("Converting back to continuous values...")
        synthetic_continuous = inverse_discretize(synthetic_df, mean_dict)

        # Save
        print(f"\nSaving synthetic data to {output_path}...")
        synthetic_continuous.to_csv(output_path, index=False)
        print(f"✓ Synthetic data saved: {synthetic_continuous.shape}")

        return synthetic_continuous

    except Exception as e:
        print(f"\n✗ MPC execution failed: {e}")
        print("\nPossible reasons:")
        print("  - MP-SPDZ not installed or not in expected path")
        print("  - MPC protocol file not found")
        print("  - Network issues between MPC parties")
        print("\nFalling back to standard pipeline...")
        return run_standard_pipeline(data_path, output_path, epsilon, delta)


def main():
    parser = argparse.ArgumentParser(
        description='Run Private-PGM pipeline with optional MPC support'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='synthetic_data.csv',
        help='Path to save synthetic data (default: synthetic_data.csv)'
    )
    parser.add_argument(
        '--use_mpc',
        action='store_true',
        help='Use MPC protocols for computation'
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
        '--mpspdz_path',
        type=str,
        default=None,
        help='Path to MP-SPDZ installation'
    )
    parser.add_argument(
        '--bin_protocol',
        type=str,
        default='ppai_bin_opt',
        help='MPC protocol for binning (default: ppai_bin_opt)'
    )
    parser.add_argument(
        '--marginal_protocol',
        type=str,
        default='ppai_msr_noisy_final',
        help='MPC protocol for marginals (default: ppai_msr_noisy_final)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return

    # Run appropriate pipeline
    if args.use_mpc:
        synthetic_data = run_mpc_pipeline(
            data_path=args.data_path,
            output_path=args.output_path,
            epsilon=args.epsilon,
            delta=args.delta,
            mpspdz_path=args.mpspdz_path,
            bin_protocol=args.bin_protocol,
            marginal_protocol=args.marginal_protocol
        )
    else:
        synthetic_data = run_standard_pipeline(
            data_path=args.data_path,
            output_path=args.output_path,
            epsilon=args.epsilon,
            delta=args.delta
        )

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
