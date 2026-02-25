# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
#
# Secure MPC implementation of Private-PGM with no data leakage

"""
Secure MPC Private-PGM Implementation

This module provides a secure implementation that ensures:
1. Raw data never leaves individual parties
2. All computations on sensitive data happen in MPC
3. Only DP-protected noisy statistics are revealed
4. Full (ε,δ)-DP + MPC security guarantees

SECURITY GUARANTEES:
- Input: Each party keeps their raw data locally
- Computation: All operations in MPC with secret sharing
- Output: Only noisy DP-protected statistics revealed
- End-to-End: (ε,δ)-DP + semi-honest MPC security
"""

from scipy import optimize, sparse
import numpy as np
import pandas as pd
import sys
import os
import warnings
import math

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.rdp_accountant import compute_rdp, get_privacy_spent
from mbi import FactoredInference, Domain
from utils.mpc_helper import MPCMarginalComputer, MPCBinningComputer


class SecureMPCPrivatePGM:
    """
    Secure MPC implementation of Private-PGM with no data leakage

    This class ensures that:
    - Raw data is never combined across parties
    - All sensitive computations happen in MPC
    - Only DP-protected outputs are revealed
    """

    def __init__(self, target_variable, enable_privacy, target_epsilon, target_delta,
                 mpspdz_path=None, mpc_protocol="ring", num_parties=2):
        """
        Initialize Secure MPC Private PGM model

        Args:
            target_variable: Target variable name for classification
            enable_privacy: Whether to enable differential privacy
            target_epsilon: Privacy parameter epsilon
            target_delta: Privacy parameter delta
            mpspdz_path: Path to MP-SPDZ installation (required)
            mpc_protocol: MPC protocol to use (default: ring)
            num_parties: Number of data custodian parties (default: 2)
        """
        self.target_epsilon = target_epsilon
        self.enable_privacy = enable_privacy
        self.target_delta = target_delta
        self.target_variable = target_variable
        self.model = None
        self.mpspdz_path = mpspdz_path
        self.mpc_protocol = mpc_protocol
        self.num_parties = num_parties

        # Convert to absolute path for robust checking
        abs_mpspdz_path = os.path.abspath(mpspdz_path)

        if not os.path.exists(abs_mpspdz_path):
            print(f"⚠️  WARNING: MP-SPDZ directory not found at {abs_mpspdz_path}")
            exit(1)  # For secure MPC, we require MP-SPDZ. Exit if not found.
        else:
            # Update to use absolute path
            self.mpspdz_path = abs_mpspdz_path

        self.mpc_binner = MPCBinningComputer(
            mpspdz_path=self.mpspdz_path,
            protocol=self.mpc_protocol
        )

        self.mpc_computer = MPCMarginalComputer(
            mpspdz_path=self.mpspdz_path,
            protocol=self.mpc_protocol
        )

        # Privacy budget allocation
        self.epsilon_binning = target_epsilon / 2  # Half budget for binning
        self.epsilon_marginals = target_epsilon / 2  # Half for marginals
        self.delta_binning = target_delta / 2
        self.delta_marginals = target_delta / 2

        print("="*80)
        print("SECURE MPC PRIVATE-PGM INITIALIZED")
        print("="*80)
        print(f"Total privacy budget: (ε={target_epsilon}, δ={target_delta})")
        print(f"Binning budget: (ε={self.epsilon_binning}, δ={self.delta_binning})")
        print(f"Marginals budget: (ε={self.epsilon_marginals}, δ={self.delta_marginals})")
        print(f"MPC parties: {num_parties}")
        print("="*80)

    @staticmethod
    def moments_calibration(round1, round2, eps, delta):
        """
        Calibrate noise for differential privacy using moments accountant

        Args:
            round1: L2 sensitivity of first round
            round2: L2 sensitivity of second round
            eps: Target epsilon
            delta: Target delta

        Returns:
            float: Calibrated sigma value
        """
        orders = range(2, 4096)

        def obj(sigma):
            rdp1 = compute_rdp(1.0, sigma / round1, 1, orders)
            rdp2 = compute_rdp(1.0, sigma / round2, 1, orders)
            rdp = rdp1 + rdp2
            privacy = get_privacy_spent(orders, rdp, delta=delta)
            return privacy[0] - eps + 1e-8

        low = 1.0
        high = 1.0
        while obj(low) < 0:
            low /= 2.0
        while obj(high) > 0:
            high *= 2.0
        sigma = optimize.bisect(obj, low, high)
        assert (
            obj(sigma) - 1e-8 <= 0
        ), "not differentially private"
        return sigma

    def train_from_party_files(self, party_data_files, config,
                                marginal_protocol='ppai_msr_noisy_final',
                                cliques=None, num_iters=10000, 
                                deg_filtering=None, max_gene_val = 15, bin_num = 4):
        """
        Train model from separate party data files (SECURE - no data leakage)
        """
        print("\n" + "="*80)
        print("SECURE MPC TRAINING - NO DATA LEAKAGE")
        print("="*80)

        # Validate inputs
        if len(party_data_files) != self.num_parties:
            raise ValueError(
                f"Expected {self.num_parties} party files, got {len(party_data_files)}"
            )

        for i, filepath in enumerate(party_data_files):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Party {i} data file not found: {filepath}")

        # Get metadata
        num_genes = len([col for col in config.keys() if col != self.target_variable])
        num_classes = config[self.target_variable]

        # Calculate total samples
        total_samples = sum(sum(1 for _ in open(filepath, 'r')) - 1 for filepath in party_data_files)

        # ==========================================================
        # PRIVACY BUDGET ALLOCATION (Dynamic 3-way or 2-way split)
        # ==========================================================
        eps_topk, delta_topk = 0.0, 0.0
        
        if self.enable_privacy:
            if deg_filtering is not None and deg_filtering > 0:
                print("\n[Privacy Accountant] Dynamic 3-Way Budget Split Engaged:")
                # 20% Top-K, 20% Binning, 60% Marginals
                eps_topk = self.target_epsilon * 0.2
                self.epsilon_binning = self.target_epsilon * 0.2
                self.epsilon_marginals = self.target_epsilon * 0.6
                
                delta_topk = self.target_delta * 0.2
                self.delta_binning = self.target_delta * 0.2
                self.delta_marginals = self.target_delta * 0.6
                
                print(f"  Top-K DEG: ε={eps_topk:.3f}, δ={delta_topk}")
                print(f"  Binning:   ε={self.epsilon_binning:.3f}, δ={self.delta_binning}")
                print(f"  Marginals: ε={self.epsilon_marginals:.3f}, δ={self.delta_marginals}")
            else:
                # Guided by the realization that marginal selection does not matter as much, since even random features worked well. 
                self.epsilon_binning = self.target_epsilon * 0.5
                self.epsilon_marginals = self.target_epsilon * 0.5
                self.delta_binning = self.target_delta * 0.5
                self.delta_marginals = self.target_delta * 0.5

            if self.target_delta > 0:
                # Calculate the smallest possible bin size
                # Using exact quartiles means each bin has 1/4th of the data
                num_train = 945 # TODO: change to computed number so that I do not need to do this
                bin_size = num_train / bin_num 

                # Calculate the L2 sensitivity of the means
                # How much one patient can change the mean * sqrt(number of genes)
                l2_sensitivity_mean = (max_gene_val / bin_size) * math.sqrt(num_genes)                
                sigma_bin = self.moments_calibration(l2_sensitivity_mean, 1e-9, self.epsilon_binning, self.delta_binning)
                sigma_marginal = self.moments_calibration(1.0, 1.0, self.epsilon_marginals, self.delta_marginals)
            else:
                sigma_bin = 1.0 / num_genes / 2.0
                sigma_marginal = 1.0 / num_genes / 2.0
        else:
            sigma_bin = 0.0
            sigma_marginal = 0.0

        # ==========================================================
        # INTEGRATED MPC WORKFLOW
        # ==========================================================
        marginals_1way, marginals_2way, bin_means_array, selected_indices = self.mpc_computer.compute_marginals_with_binning(
            party_data_files=party_data_files,
            num_genes=num_genes,
            num_classes=num_classes,
            target_delta=self.target_delta,
            sigma=sigma_marginal,
            sigma_bin=sigma_bin,
            deg_filtering=deg_filtering, # Pass DEG k
            epsilon_topk=eps_topk,       # Pass Top-K allocated epsilon
            delta_topk=delta_topk,        # Pass Top-K allocated delta
            protocol_name=marginal_protocol,
            max_val=max_gene_val
        )
        all_feature_names = [k for k in config.keys() if k != self.target_variable]
        
        if deg_filtering is not None and deg_filtering > 0 and selected_indices:
            selected_gene_names = [all_feature_names[i] for i in selected_indices]
        else:
            selected_gene_names = all_feature_names
            
        # Store the final correct columns on the model object so the orchestrator can read them
        self.selected_columns = selected_gene_names + [self.target_variable]

        # 2. Map bin means correctly
        if bin_means_array is not None:
            self.noisy_bin_means = {}
            for i, col_name in enumerate(selected_gene_names):
                self.noisy_bin_means[col_name] = bin_means_array[i]
        else:
            self.noisy_bin_means = None

        # STEP 3: Public Inference (on noisy public statistics)
        print("\n" + "-"*80)
        print("STEP 3: PUBLIC INFERENCE ON NOISY STATISTICS")
        print("-"*80)
        
        # 3. Build a filtered config matching ONLY the selected genes
        filtered_config = {k: config[k] for k in selected_gene_names}
        filtered_config[self.target_variable] = config[self.target_variable]

        measurements = self._convert_to_measurements(
            marginals_1way,
            marginals_2way,
            filtered_config,
            sigma_marginal,
            cliques
        )

        domain = Domain(filtered_config.keys(), filtered_config.values())
        engine = FactoredInference(domain, log=True, iters=num_iters)
        self.model = engine.estimate(measurements, total=total_samples, engine="MD")

        print("✓ Model training completed")


    def _convert_to_measurements(self, marginals_1way, marginals_2way, config,
                                  sigma, cliques=None):
        """
        Convert MPC marginal outputs to measurement format

        Args:
            marginals_1way: Noisy 1-way marginals from MPC
            marginals_2way: Noisy 2-way marginals from MPC
            config: Domain configuration
            sigma: Noise parameter
            cliques: List of cliques

        Returns:
            list: Measurements in format expected by FactoredInference
        """
        measurements = []
        domain_keys = list(config.keys())

        print(f"\nDEBUG _convert_to_measurements:")
        print(f"  domain_keys: {domain_keys}")
        print(f"  marginals_1way length: {len(marginals_1way)}")
        print(f"  marginals_2way length: {len(marginals_2way)}")
        print(f"  target_variable: {self.target_variable}")

        # Process 1-way marginals
        weights = np.ones(len(config))
        weights /= np.linalg.norm(weights)

        col_idx = 0
        for col, wgt in zip(domain_keys, weights):
            if col == self.target_variable:
                # Label marginals
                num_classes = config[col]
                y = marginals_1way[-num_classes:]
            else:
                # Feature marginals (4 bins per feature)
                y = marginals_1way[col_idx * 4:(col_idx + 1) * 4]
                col_idx += 1

            I = sparse.eye(len(y))

            if self.target_delta > 0:
                measurements.append((I, y / wgt, 1.0 / wgt, (col,)))
            else:
                measurements.append((I, y, sigma, (col,)))

        # Process 2-way marginals
        if cliques is None:
            cliques = []
            for col in domain_keys:
                if col != self.target_variable:
                    cliques.append((col, self.target_variable))

        weights = np.ones(len(cliques))
        weights /= np.linalg.norm(weights)

        print(f"  Processing {len(cliques)} 2-way cliques...")
        for clique_idx, (cl, wgt) in enumerate(zip(cliques, weights)):
            # Each 2-way marginal is 4 feature bins * num_classes label bins
            num_classes = config[self.target_variable]
            marginal_size = 4 * num_classes
            start_idx = clique_idx * marginal_size
            end_idx = (clique_idx + 1) * marginal_size
            print(f"    Clique {clique_idx} {cl}: marginal_size={marginal_size}, indices [{start_idx}:{end_idx}]")
            y = marginals_2way[start_idx:end_idx]
            print(f"      Got {len(y)} values")
            I = sparse.eye(len(y))

            if self.target_delta > 0:
                measurements.append((I, y / wgt, 1.0 / wgt, cl))
            else:
                measurements.append((I, y, sigma, cl))

        print(f"\nDEBUG: Created {len(measurements)} total measurements")
        return measurements

    def _load_bin_means(self, filepath):
        """
        Load noisy bin means from MPC output

        Args:
            filepath: Path to file containing noisy bin means

        Returns:
            dict: Dictionary mapping column names to bin means
        """
        if not os.path.exists(filepath):
            warnings.warn(f"Bin means file not found: {filepath}")
            return {}

        # Load noisy bin means
        # Format: num_genes rows, 4 columns (one per bin)
        bin_means_array = np.loadtxt(filepath)

        # Convert to dictionary format
        bin_means_dict = {}
        for i in range(len(bin_means_array)):
            bin_means_dict[f'gene_{i}'] = bin_means_array[i]

        return bin_means_dict

    def generate(self, num_rows=None):
        """
        Generate synthetic data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_from_party_files() first.")

        syn_df = self.model.synthetic_data(rows=num_rows).df
        
        # SAFTEY FIX: Private-PGM can sometimes generate NaNs if DP noise 
        # causes zeroed probabilities. We fill them with bin 0 to prevent crashes.
        if syn_df.isnull().values.any():
            warnings.warn("NaNs detected in generated synthetic data. Filling with default bin 0.")
            syn_df = syn_df.fillna(0)

        X_syn = syn_df.drop([self.target_variable], axis=1).values
        y_syn = syn_df[self.target_variable].values
        return np.concatenate([X_syn, np.expand_dims(y_syn, axis=1)], axis=1)

    def generate_continuous(self, num_rows=None):
        """
        Generate synthetic data and convert back to continuous values
        using noisy bin means (DP-protected)
        """
        discrete_data = self.generate(num_rows)

        if not self.noisy_bin_means:
            warnings.warn("Noisy bin means not available. Returning discrete data.")
            return discrete_data

        # Convert using noisy bin means (which are DP-protected)
        continuous_data = discrete_data.astype(float)

        for col_idx, (col_name, bin_means) in enumerate(self.noisy_bin_means.items()):
            if col_idx < discrete_data.shape[1] - 1:  # Exclude label column
                for i in range(len(discrete_data)):
                    val = discrete_data[i, col_idx]
                    
                    # Extra safety check against NaNs
                    if np.isnan(val):
                        bin_idx = 0
                    else:
                        bin_idx = int(val)
                        
                    if 0 <= bin_idx < len(bin_means):
                        continuous_data[i, col_idx] = bin_means[bin_idx]

        return continuous_data

