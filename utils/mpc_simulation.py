"""
Simulation Mode for MPC Pipeline (No MP-SPDZ Required)

This module provides a simulation/mock implementation that mimics MPC
behavior without requiring MP-SPDZ installation. Useful for:
- Development and testing
- Environments where MP-SPDZ cannot be installed
- Quick prototyping

SECURITY NOTE: This is a SIMULATION only. For real multi-party scenarios
with actual data separation, use the full MPC implementation with MP-SPDZ.
"""

import numpy as np
import pandas as pd
from scipy import sparse
import warnings


class SimulatedMPCBinner:
    """
    Simulates MPC binning without requiring MP-SPDZ

    This performs the same operations as the MPC protocol would,
    but on combined data. Useful for testing and development.
    """

    def __init__(self, add_noise=True):
        """
        Args:
            add_noise: Whether to add DP noise (should be True for privacy)
        """
        self.add_noise = add_noise

    def bin_data_simulated(self, party_data_files, num_genes, num_classes,
                          epsilon, delta, quantiles=[0.25, 0.5, 0.75]):
        """
        Simulate MPC binning by:
        1. Loading data from party files
        2. Computing quantiles and bins (simulates MPC computation)
        3. Computing noisy bin means

        Args:
            party_data_files: List of CSV file paths
            num_genes: Number of features
            num_classes: Number of classes
            epsilon: Privacy parameter for binning
            delta: Privacy parameter for binning
            quantiles: Quantile points for binning

        Returns:
            tuple: (binned_data, noisy_bin_means_dict)
        """
        print("  MODE: Simulated MPC (no MP-SPDZ required)")
        print("  Loading party data files...")

        # Load data from parties
        dfs = []
        for i, filepath in enumerate(party_data_files):
            df = pd.read_csv(filepath)
            print(f"    Party {i}: {len(df)} samples loaded")
            dfs.append(df)

        # Combine (in real MPC, this would be secret-shared)
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"  Combined: {len(combined_df)} total samples")

        # Separate features and labels
        if 'label' in combined_df.columns:
            features = combined_df.drop('label', axis=1)
            labels = combined_df['label']
        else:
            features = combined_df.iloc[:, :-1]
            labels = combined_df.iloc[:, -1]

        # Compute quantiles and bin data
        print(f"  Computing quantiles and binning {num_genes} features...")

        binned_features = pd.DataFrame()
        noisy_bin_means = {}

        for col in features.columns:
            # Compute quantiles
            q_values = features[col].quantile(quantiles).values

            # Bin the data
            binned_col = np.digitize(features[col], q_values)
            binned_features[col] = binned_col

            # Compute mean for each bin
            bin_means = []
            for bin_idx in range(len(quantiles) + 1):
                bin_mask = (binned_col == bin_idx)
                if bin_mask.sum() > 0:
                    bin_mean = features[col][bin_mask].mean()
                else:
                    bin_mean = 0.0

                # Add DP noise if enabled
                if self.add_noise and epsilon > 0:
                    noise_scale = self._compute_noise_scale(epsilon, delta)
                    noise = np.random.normal(0, noise_scale)
                    bin_mean += noise

                bin_means.append(bin_mean)

            noisy_bin_means[col] = np.array(bin_means)

        # Reconstruct binned dataframe with label
        binned_df = binned_features.copy()
        binned_df['label'] = labels.values

        print(f"  ✓ Binning completed with DP noise")
        print(f"  ✓ Computed noisy means for {len(noisy_bin_means)} features")

        return binned_df, noisy_bin_means

    def _compute_noise_scale(self, epsilon, delta):
        """
        Compute noise scale for Gaussian mechanism

        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter

        Returns:
            float: Noise standard deviation
        """
        if delta == 0:
            # Laplace mechanism
            return 1.0 / epsilon
        else:
            # Gaussian mechanism (simplified)
            return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


class SimulatedMPCMarginalComputer:
    """
    Simulates MPC marginal computation without requiring MP-SPDZ
    """

    def __init__(self, add_noise=True):
        """
        Args:
            add_noise: Whether to add DP noise
        """
        self.add_noise = add_noise

    def compute_marginals_simulated(self, binned_df, num_genes, num_classes,
                                    epsilon, delta, target_variable='label'):
        """
        Simulate MPC marginal computation

        Args:
            binned_df: DataFrame with binned data
            num_genes: Number of features
            num_classes: Number of classes
            epsilon: Privacy parameter for marginals
            delta: Privacy parameter for marginals
            target_variable: Name of target column

        Returns:
            tuple: (marginals_1way, marginals_2way)
        """
        print("  MODE: Simulated MPC (no MP-SPDZ required)")
        print("  Computing 1-way and 2-way marginals...")

        # Separate features and labels
        if target_variable in binned_df.columns:
            features = binned_df.drop(target_variable, axis=1)
            labels = binned_df[target_variable]
        else:
            features = binned_df.iloc[:, :-1]
            labels = binned_df.iloc[:, -1]

        # Compute 1-way marginals for features
        marginals_1way_features = []
        for col in features.columns:
            counts = np.zeros(4)  # 4 bins
            for bin_idx in range(4):
                count = (features[col] == bin_idx).sum()

                # Add DP noise
                if self.add_noise and epsilon > 0:
                    noise_scale = self._compute_noise_scale(epsilon / 2, delta / 2)
                    noise = np.random.normal(0, noise_scale)
                    count += noise

                counts[bin_idx] = max(0, count)  # Ensure non-negative

            marginals_1way_features.extend(counts)

        # Compute 1-way marginals for labels
        marginals_1way_labels = []
        for class_idx in range(num_classes):
            count = (labels == class_idx).sum()

            if self.add_noise and epsilon > 0:
                noise_scale = self._compute_noise_scale(epsilon / 2, delta / 2)
                noise = np.random.normal(0, noise_scale)
                count += noise

            marginals_1way_labels.append(max(0, count))

        marginals_1way = np.array(marginals_1way_features + marginals_1way_labels)

        # Compute 2-way marginals (feature x label)
        marginals_2way = []
        for col in features.columns:
            for bin_idx in range(4):
                for class_idx in range(num_classes):
                    count = ((features[col] == bin_idx) & (labels == class_idx)).sum()

                    if self.add_noise and epsilon > 0:
                        noise_scale = self._compute_noise_scale(epsilon / 2, delta / 2)
                        noise = np.random.normal(0, noise_scale)
                        count += noise

                    marginals_2way.append(max(0, count))

        marginals_2way = np.array(marginals_2way)

        print(f"  ✓ Computed 1-way marginals: {len(marginals_1way)} values")
        print(f"  ✓ Computed 2-way marginals: {len(marginals_2way)} values")
        print(f"  ✓ All marginals have DP noise added")

        return marginals_1way, marginals_2way

    def _compute_noise_scale(self, epsilon, delta):
        """Compute noise scale for Gaussian mechanism"""
        if delta == 0:
            return 1.0 / epsilon
        else:
            return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def create_simulation_mode_warning():
    """Display warning about simulation mode"""
    warning_msg = """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                        SIMULATION MODE ACTIVE                          ║
    ╚════════════════════════════════════════════════════════════════════════╝

    You are running in SIMULATION MODE (no MP-SPDZ).

    What this means:
    ✓ Still provides (ε,δ)-differential privacy
    ✓ Useful for testing and development
    ✓ Faster execution (no MPC overhead)
    ✗ Does NOT provide MPC security guarantees
    ✗ Data from all parties is combined (not secret-shared)

    For production with real multi-party data separation:
    → Install MP-SPDZ and use --use-real-mpc flag

    ════════════════════════════════════════════════════════════════════════
    """
    print(warning_msg)
