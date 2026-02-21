"""
MPC Helper Module for integrating MP-SPDZ with Private-PGM

This module provides utilities for executing MPC protocols with MP-SPDZ framework.
"""

import os
import subprocess
import numpy as np
import tempfile
import math
import time
import re
from pathlib import Path

import pandas as pd
import numpy as np

MPC_METRICS = {
    'compile_time': 0.0, 
    'execute_time': 0.0,
    'integer_bits': 0,
    'integer_opens': 0,
    'integer_triples': 0,
    'vm_rounds': 0
}

def calculate_f_stat_noise(epsilon_topk, delta_topk, k, num_genes, f_max_clip=10.0):
    if k <= 0: return 0
    
    # 1. L2 Sensitivity for the returned F-statistic value
    sensitivity_l2 = f_max_clip * math.sqrt(num_genes)
    
    # 2. Strict splitting of epsilon and delta for sequential composition
    eps_per_step = epsilon_topk / k
    delta_per_step = delta_topk / k 
    
    # 3. Gaussian noise scale formula
    sigma = (sensitivity_l2 * math.sqrt(2 * math.log(1.25 / delta_per_step))) / eps_per_step
    print(f"    [Noise Calc] Sensitivity: {sensitivity_l2:.4f}, Eps/Step: {eps_per_step:.4f}, Delta/Step: {delta_per_step:.4e}, Sigma: {sigma:.4f}")
    return int(sigma * 10000)

class MPCProtocolExecutor:
    """
    Executor for MP-SPDZ protocols that handles compilation and execution
    of .mpc files for secure multi-party computation.
    """

    def __init__(self, mpspdz_path=None, protocol="ring", working_dir=None):
        """
        Initialize MPC Protocol Executor

        Args:
            mpspdz_path: Path to MP-SPDZ installation (default: /home/mpcuser/MP-SPDZ/)
            protocol: MPC protocol to use (default: ring)
            working_dir: Working directory for MPC execution (default: current directory)
        """
        self.mpspdz_path = mpspdz_path or os.environ.get('MPSPDZ_PATH', '/home/mpcuser/MP-SPDZ/')
        self.protocol = protocol
        self.working_dir = working_dir or os.getcwd()

    def compile_protocol(self, mpc_file, args=None):
        protocol_name = mpc_file.replace('.mpc', '')
        compile_cmd = f"cd {self.mpspdz_path} && ./compile.py -R 64 {protocol_name}"
        if args:
            compile_cmd += " " + " ".join([str(arg) for arg in args])

        print(f"Compiling: {compile_cmd}")

        start_time = time.time()
        # Changed from os.system to subprocess.run to capture the text output
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        MPC_METRICS['compile_time'] += (time.time() - start_time)

        if result.returncode == 0:
            print(result.stdout) # Print it so you can still see it in the console
            
            # Parse the text to extract the metrics
            bits_match = re.search(r'(\d+)\s+integer bits', result.stdout)
            opens_match = re.search(r'(\d+)\s+integer opens', result.stdout)
            triples_match = re.search(r'(\d+)\s+integer triples', result.stdout)
            vm_match = re.search(r'(\d+)\s+virtual machine rounds', result.stdout)
            
            if bits_match: MPC_METRICS['integer_bits'] += int(bits_match.group(1))
            if opens_match: MPC_METRICS['integer_opens'] += int(opens_match.group(1))
            if triples_match: MPC_METRICS['integer_triples'] += int(triples_match.group(1))
            if vm_match: MPC_METRICS['vm_rounds'] += int(vm_match.group(1))

            print(f"Compilation successful: {protocol_name}")
            return True
        else:
            raise RuntimeError(f"MPC compilation failed for {protocol_name}\nSTDERR:\n{result.stderr}")

    def execute_protocol(self, protocol_name, num_parties=2, args=None):
        protocol_script = os.path.join(self.mpspdz_path, 'Scripts', f'{self.protocol}.sh')
        full_protocol_name = protocol_name + "-" + "-".join([str(arg) for arg in args]) if args else protocol_name
        cmd = [protocol_script, full_protocol_name]

        try:
            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.mpspdz_path, capture_output=True, text=True, check=True)
            MPC_METRICS['execute_time'] += (time.time() - start_time)
            return result
        except subprocess.CalledProcessError as e:
            print(f"\n{'='*80}")
            print(f"MPC PROTOCOL EXECUTION FAILED")
            print(f"{'='*80}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Exit code: {e.returncode}")
            if e.stdout:
                print(f"\n--- STDOUT ---")
                print(e.stdout)
            if e.stderr:
                print(f"\n--- STDERR ---")
                print(e.stderr)
            print(f"{'='*80}\n")
            raise


class HorizontalDataSplitter:
    """
    Splits data horizontally across multiple parties for MPC computation.
    Each party gets a subset of rows (patients/samples).
    """

    def __init__(self, num_parties=2):
        """
        Initialize data splitter

        Args:
            num_parties: Number of MPC parties to split data across
        """
        self.num_parties = num_parties

    def split_data(self, data, party_ratios=None):
        """
        Split data horizontally across parties

        Args:
            data: numpy array or DataFrame to split
            party_ratios: List of ratios for each party (default: equal split)

        Returns:
            list: List of data splits, one per party
        """
        n_samples = len(data)

        if party_ratios is None:
            # Equal split by default
            party_ratios = [1.0 / self.num_parties] * self.num_parties

        if abs(sum(party_ratios) - 1.0) > 1e-6:
            raise ValueError("Party ratios must sum to 1.0")

        splits = []
        start_idx = 0

        for i, ratio in enumerate(party_ratios):
            if i == len(party_ratios) - 1:
                # Last party gets remaining data
                end_idx = n_samples
            else:
                end_idx = start_idx + int(n_samples * ratio)

            splits.append(data[start_idx:end_idx])
            start_idx = end_idx

        return splits

    def write_party_data(self, data_splits, output_dir, prefix="party_data"):
        """
        Write party data to CSV files for MPC input

        Args:
            data_splits: List of data arrays for each party
            output_dir: Directory to write output files
            prefix: Prefix for output filenames

        Returns:
            list: List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        for i, data in enumerate(data_splits):
            filepath = os.path.join(output_dir, f"{prefix}_{i}.csv")
            np.savetxt(filepath, data, delimiter=',')
            output_files.append(filepath)

        return output_files


class MPCBinningComputer:
    """
    Performs secure binning using MPC protocols
    """

    def __init__(self, mpspdz_path=None, protocol="ring"):
        """
        Initialize MPC Binning Computer

        Args:
            mpspdz_path: Path to MP-SPDZ installation
            protocol: MPC protocol to use
        """
        self.executor = MPCProtocolExecutor(mpspdz_path, protocol)
        self.splitter = HorizontalDataSplitter(num_parties=2)

    def _parse_binned_output(self, stdout):
        """
        Parse binned data output from ppai_bin.mpc stdout

        Args:
            stdout: Standard output from MPC execution

        Returns:
            list of lists: Binned data rows
        """
        lines = stdout.split('\n')
        binned_data = []
        in_output_section = False

        for line in lines:
            line = line.strip()
            if line == '=== BINNING_OUTPUT_START ===':
                in_output_section = True
                continue
            elif line == '=== BINNING_OUTPUT_END ===':
                break
            elif in_output_section and line:
                # Skip the size line
                if ' ' in line and not line.startswith('Binned'):
                    parts = line.split()
                    if len(parts) > 2:  # Data line
                        binned_data.append([float(x) for x in parts])

        if not binned_data:
            raise ValueError("No binned data found in MPC output")

        return binned_data

class MPCMarginalComputer:
    def __init__(self, mpspdz_path=None, protocol="ring"):
        self.executor = MPCProtocolExecutor(mpspdz_path, protocol)
        self.splitter = HorizontalDataSplitter(num_parties=2)
        self.binner = MPCBinningComputer(mpspdz_path, protocol)
    def compute_marginals_with_binning(self, party_data_files, num_genes,
                                       num_classes, target_delta, sigma, mpc_sigma_bin,
                                       deg_filtering=None, epsilon_topk=None, delta_topk=None,
                                       protocol_name='ppai_bin_msr'): # <-- Added protocol_name
        """
        Complete workflow: Pre-computes histograms or handles raw binning.
        """
        print("\n" + "="*80)
        
        # 1. BRANCH LOGIC: Histogram vs Standard
        if protocol_name == 'histogram_marginals':
            print("INTEGRATED MPC WORKFLOW: Pre-computed Histograms (O(1) respect to N)")
            mpc_sigma_marginal = int(sigma * 10000)
            mpc_sigma_f_stat = 0 # Not used for this branch
            
        elif deg_filtering is not None and deg_filtering > 0:
            print(f"INTEGRATED MPC WORKFLOW: DP-DEG Top-{deg_filtering} Filter → Binning → MSR")
            protocol_name = 'deg_dp_pipeline'
            mpc_sigma_marginal = int(sigma * 10000)
            eps_k = epsilon_topk if epsilon_topk else 1.0
            del_k = delta_topk if delta_topk else target_delta
            mpc_sigma_f_stat = calculate_f_stat_noise(
                epsilon_topk=eps_k, delta_topk=del_k, k=deg_filtering, num_genes=num_genes
            )
        else:
            print("INTEGRATED MPC WORKFLOW: Binning → MSR (Standard, All Genes)")
            protocol_name = 'ppai_bin_msr'
            mpc_sigma_bin = int(sigma_bin * 10000)
            mpc_sigma_marginal = int(sigma * 10000)
            mpc_sigma_f_stat = 0
            
        print("="*80)
        
        mpc_file_path = os.path.join(self.executor.mpspdz_path, f'{protocol_name}.mpc')
        if not os.path.exists(mpc_file_path):
            raise FileNotFoundError(f"MPC protocol file not found: {mpc_file_path}")

        # Calculate party sizes
        party_sizes = []
        import pandas as pd
        import numpy as np # Ensure numpy is available
        
        for party_file in party_data_files:
            party_sizes.append(len(pd.read_csv(party_file)))

        # Prepare MPC input files
        player_data_dir = os.path.join(self.executor.mpspdz_path, 'Player-Data')
        os.makedirs(player_data_dir, exist_ok=True)

        print(f"Preparing MPC input files in {player_data_dir}...")
        for party_idx, party_file in enumerate(party_data_files):
            df = pd.read_csv(party_file)
            input_file = os.path.join(player_data_dir, f'Input-P{party_idx}-0')
            
            # --- NEW LOCAL HISTOGRAM LOGIC ---
            if protocol_name == 'histogram_marginals':
                print(f"  [Party {party_idx}] Generating local 1D histograms...")
                # Extract labels and purely numeric features
                labels = df['label'].values if 'label' in df.columns else df.iloc[:, -1].values
                df_numeric = df.drop(columns=['label']) if 'label' in df.columns else df.iloc[:, :-1]
                df_numeric = df_numeric.select_dtypes(include=['number'])

                B = 10000
                g_min, g_max = 0.0, 16.0
                bins = np.linspace(g_min, g_max, B + 1)
                
                # Pre-allocate 1D array
                flat_hist = np.zeros(num_genes * B * num_classes, dtype=int)
                
                for g in range(num_genes):
                    gene_col = df_numeric.iloc[:, g].values
                    for c in range(num_classes):
                        class_mask = (labels == c)
                        counts, _ = np.histogram(gene_col[class_mask], bins=bins)
                        
                        # Highly optimized vectorized assignment to matching MPC indices
                        start_idx = g * (B * num_classes) + c
                        end_idx = start_idx + (B * num_classes)
                        flat_hist[start_idx:end_idx:num_classes] = counts

                # Fast string write
                with open(input_file, 'w') as f:
                    f.write(' '.join(map(str, flat_hist)) + '\n')
                print(f"  [Party {party_idx}] Successfully wrote {len(flat_hist)} integers.")
            else:
                df_numeric = df[df.select_dtypes(include=['number']).columns.tolist()]
                with open(input_file, 'w') as f:
                    for row in df_numeric.values:
                        f.write(' '.join([str(val) for val in row]) + '\n')

        # 2. CONSTRUCT ARGUMENTS DYNAMICALLY
        if protocol_name == 'deg_dp_pipeline':
            args = party_sizes + [num_genes, num_classes, deg_filtering, mpc_sigma_marginal, mpc_sigma_f_stat, mpc_sigma_bin]
        else:
            args = party_sizes + [num_genes, num_classes, mpc_sigma_marginal, mpc_sigma_bin]

        print(f"\nCompiling integrated MPC protocol: {protocol_name}...")
        self.executor.compile_protocol(protocol_name, args=args)

        print(f"\nExecuting integrated MPC protocol...")
        result = self.executor.execute_protocol(protocol_name, num_parties=len(party_sizes), args=args)

        print("\nParsing noisy marginals and bin means from output...")
        
        # 3. ADJUST OUTPUT PARSER (Extract k genes if filtered, else all genes)
        output_genes_count = deg_filtering if (deg_filtering is not None and deg_filtering > 0) else num_genes
        
        measurements_1way, measurements_2way, bin_means_array, selected_indices = self._parse_marginals_output(
            result.stdout, output_genes_count, num_classes
        )

        print("\n" + "="*80)
        print("✓ Complete! Only noisy marginals and means revealed - binned data stayed secret")
        print("="*80 + "\n")

        return measurements_1way, measurements_2way, bin_means_array, selected_indices

    def _parse_marginals_output(self, stdout, num_genes, num_classes):
            lines = stdout.split('\n')
            in_marginals_section = False
            in_means_section = False
            section = None

            marginals_1way_features = []
            marginals_1way_labels = []
            marginals_2way = []
            bin_means_list = []  
            current_values = [] 
            selected_indices = [] # <-- NEW: Array to track selected genes

            for line in lines:
                line = line.strip()

                # Extract DEG outputs and parse the exact index
                if line.startswith('DP Selected Gene'):
                    print(f"  [MPC Output] {line}")
                    try:
                        # Parse out the integer from "DP Selected Gene 0 (Global Index: 13)"
                        idx_str = line.split('Global Index:')[1].replace(')', '').strip()
                        selected_indices.append(int(idx_str))
                    except Exception:
                        pass
                    continue
                
                if line == '=== BIN_MEANS_START ===': in_means_section = True; continue
                elif line == '=== BIN_MEANS_END ===': in_means_section = False; continue
                elif in_means_section and line:
                    try: bin_means_list.append(float(line.strip()))
                    except ValueError: pass
                    continue

                if line == '=== MARGINALS_OUTPUT_START ===': in_marginals_section = True; continue
                elif line == '=== MARGINALS_OUTPUT_END ===':
                    if section == '2way' and current_values: marginals_2way = np.array(current_values)
                    in_marginals_section = False  
                    section = None
                    current_values = []
                    continue
                elif not in_marginals_section: continue

                if line == '1WAY_FEATURES:':
                    section = '1way_features'
                    current_values = []
                elif line == '1WAY_LABELS:':
                    if section == '1way_features' and current_values:
                        marginals_1way_features = np.array(current_values).reshape(-1, 4)
                    section = '1way_labels'
                    current_values = []
                elif line == '2WAY:':
                    if section == '1way_labels' and current_values:
                        marginals_1way_labels = np.array(current_values)
                    section = '2way'
                    current_values = []
                elif line and section:
                    try: current_values.append(float(line.strip()))
                    except ValueError: pass

            if isinstance(marginals_1way_features, list): marginals_1way_features = np.array([]).reshape(0, 4)
            if isinstance(marginals_1way_labels, list): marginals_1way_labels = np.array(marginals_1way_labels)
            if isinstance(marginals_2way, list): marginals_2way = np.array(marginals_2way)

            marginals_1way_flat = marginals_1way_features.flatten()
            measurements_1way = np.concatenate([marginals_1way_flat, marginals_1way_labels])
            measurements_2way = marginals_2way

            bin_means_array = None
            if bin_means_list:
                bin_means_array = np.array(bin_means_list).reshape(num_genes, 4)
                print(f"  DEBUG: Successfully parsed {len(bin_means_list)} bin means.")

            return measurements_1way, measurements_2way, bin_means_array, selected_indices

    def compute_marginals_from_party_files(self, party_data_files, num_genes,
                                           num_classes, target_delta, sigma,
                                           mpc_protocol_file='ppai_msr_noisy_final'):
        """
        [DEPRECATED] Compute marginals using MPC from party data files

        WARNING: This expects already binned data. Use compute_marginals_with_binning() instead
        for the complete workflow.

        Args:
            party_data_files: List of file paths containing data for each party
            num_genes: Number of gene/feature columns
            num_classes: Number of class labels
            target_delta: Delta parameter for DP
            sigma: Noise scale parameter
            mpc_protocol_file: Protocol name (without .mpc extension) or path to .mpc file

        Returns:
            tuple: (measurements_1way, measurements_2way) - both DP-protected
        """
        # Extract protocol name (remove .mpc extension if present)
        protocol_name = Path(mpc_protocol_file).stem

        # Check if .mpc file exists in MP-SPDZ directory
        mpc_file_path = os.path.join(self.executor.mpspdz_path, f'{protocol_name}.mpc')
        if not os.path.exists(mpc_file_path):
            raise FileNotFoundError(
                f"MPC marginal protocol file not found: {mpc_file_path}\n"
                f"Expected: ppai_msr.mpc or ppai_msr_noisy_final.mpc in {self.executor.mpspdz_path}"
            )

        print(f"Using MPC marginal protocol: {protocol_name}.mpc")
        print("SECURITY: Computing marginals securely - raw data never revealed")

        # Calculate total samples from party files
        n_samples = sum(
            sum(1 for _ in open(f)) - 1  # Subtract header
            for f in party_data_files
        )

        # Prepare MPC arguments
        args = [n_samples, num_genes, num_classes]

        print(f"Compiling MPC protocol with arguments: {args}")
        # Compile MPC protocol WITH arguments (needed for program.args)
        # Pass protocol name WITHOUT .mpc extension (MP-SPDZ expects this)
        self.executor.compile_protocol(protocol_name, args=args)

        print(f"Executing MPC marginal computation...")
        print(f"Total samples (public): {n_samples}")
        print(f"Number of parties: {len(party_data_files)}")

        result = self.executor.execute_protocol(
            protocol_name,
            num_parties=len(party_data_files),
            args=args
        )

        # Read results from MPC output
        output_dir = os.path.join(self.executor.mpspdz_path, 'Player-Data')

        # These outputs are DP-protected (noisy)
        measurements_1way = self._read_mpc_output(
            os.path.join(output_dir, 'marginals_1way.txt'),
            num_genes * 4 + num_classes  # 4 bins per gene + label classes
        )

        measurements_2way = self._read_mpc_output(
            os.path.join(output_dir, 'marginals_2way.txt'),
            num_genes * 20  # 4 feature values * 5 label values per gene
        )

        print("✓ Marginals computed securely with DP protection")

        return measurements_1way, measurements_2way

    def compute_marginals_mpc(self, data, num_genes, num_classes,
                               target_delta, sigma, mpc_protocol_file):
        """
        [DEPRECATED - INSECURE] Compute marginals from combined data

        WARNING: This method combines data from all parties before MPC,
        which reveals raw data. Use compute_marginals_from_party_files() instead.

        Args:
            data: Input data array (n_samples, n_features+1)
            num_genes: Number of gene/feature columns
            num_classes: Number of class labels
            target_delta: Delta parameter for DP
            sigma: Noise scale parameter
            mpc_protocol_file: Path to .mpc protocol file for marginal computation

        Returns:
            tuple: (measurements_1way, measurements_2way)
        """
        print("WARNING: Using insecure method that exposes raw data!")
        print("Consider using compute_marginals_from_party_files() instead")

        # Split data between parties
        data_splits = self.splitter.split_data(data, party_ratios=[0.5, 0.5])

        # Write party data to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            party_files = self.splitter.write_party_data(
                data_splits,
                temp_dir,
                prefix="mpc_input"
            )

            # Use the secure method
            return self.compute_marginals_from_party_files(
                party_files,
                num_genes,
                num_classes,
                target_delta,
                sigma,
                mpc_protocol_file
            )

    def _read_mpc_output(self, filepath, *shape):
        """
        Read MPC output from file

        Args:
            filepath: Path to output file
            shape: Shape dimensions for output array

        Returns:
            numpy array with output values
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MPC output file not found: {filepath}")

        data = np.loadtxt(filepath)

        if shape:
            data = data.reshape(shape)

        return data
