"""
MPC Helper Module for integrating MP-SPDZ with Private-PGM

This module provides utilities for executing MPC protocols with MP-SPDZ framework.
"""

import os
import subprocess
import numpy as np
import tempfile
from pathlib import Path


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
        """
        Compile .mpc protocol file using MP-SPDZ compiler

        Args:
            mpc_file: Path to .mpc file to compile
            args: Optional list of compile-time arguments

        Returns:
            bool: True if compilation successful
        """
        compile_script = os.path.join(self.mpspdz_path, 'compile.py')

        if not os.path.exists(compile_script):
            raise FileNotFoundError(f"MP-SPDZ compile.py not found at {compile_script}")

        # MP-SPDZ compile command with ring protocol support
        cmd = [compile_script, '-R', '64', mpc_file]

        # Add compile-time arguments if provided (needed for program.args)
        if args:
            cmd.extend([str(arg) for arg in args])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.mpspdz_path,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Compilation successful: {mpc_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr}")
            return False

    def execute_protocol(self, protocol_name, num_parties=2, args=None):
        """
        Execute compiled MPC protocol

        Args:
            protocol_name: Name of compiled protocol (without .mpc extension)
            num_parties: Number of MPC parties
            args: List of command-line arguments to pass to the protocol

        Returns:
            subprocess.CompletedProcess: Result of execution
        """
        protocol_script = os.path.join(
            self.mpspdz_path,
            'Scripts',
            f'{self.protocol}.sh'
        )

        if not os.path.exists(protocol_script):
            raise FileNotFoundError(f"Protocol script not found at {protocol_script}")

        cmd = [protocol_script, protocol_name]

        if args:
            cmd.extend([str(arg) for arg in args])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.mpspdz_path,
                capture_output=True,
                text=True,
                check=True
            )
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

    def bin_data_mpc(self, party_data_files, num_genes, num_classes,
                     mpc_protocol_file='ppai_bin_opt'):
        """
        Bin data using MPC protocol - ensures raw data never revealed

        Args:
            party_data_files: List of file paths containing data for each party
            num_genes: Number of gene/feature columns
            num_classes: Number of class labels
            mpc_protocol_file: Protocol name (without .mpc extension) or path to .mpc file

        Returns:
            tuple: (binned_data_file, noisy_bin_means_file)
        """
        # Extract protocol name (remove .mpc extension if present)
        protocol_name = Path(mpc_protocol_file).stem

        # Check if .mpc file exists in MP-SPDZ directory
        mpc_file_path = os.path.join(self.executor.mpspdz_path, f'{protocol_name}.mpc')
        if not os.path.exists(mpc_file_path):
            raise FileNotFoundError(
                f"MPC binning protocol file not found: {mpc_file_path}\n"
                f"Expected one of: ppai_bin.mpc, ppai_bin_opt.mpc, ppai_bin_test.mpc in {self.executor.mpspdz_path}"
            )

        print(f"Using MPC binning protocol: {protocol_name}.mpc")
        print("SECURITY: Raw data will never be revealed - all binning done in MPC")

        # Calculate party sizes from input files
        party_sizes = []
        for party_file in party_data_files:
            with open(party_file, 'r') as f:
                party_sizes.append(sum(1 for _ in f) - 1)  # Subtract header

        # Prepare MPC input files (raw data as secret shares)
        player_data_dir = os.path.join(self.executor.mpspdz_path, 'Player-Data')
        os.makedirs(player_data_dir, exist_ok=True)

        print(f"Preparing MPC input files in {player_data_dir}...")
        for party_idx, party_file in enumerate(party_data_files):
            import pandas as pd
            df = pd.read_csv(party_file)

            # Filter to only numeric columns (remove IDs, string columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            df_numeric = df[numeric_cols]

            print(f"  Party {party_idx}: {len(df)} rows, {len(numeric_cols)} numeric columns")

            # Convert to numpy array (all numeric)
            data_array = df_numeric.values

            # Write in MP-SPDZ input format (space-separated values)
            input_file = os.path.join(player_data_dir, f'Input-P{party_idx}-0')
            with open(input_file, 'w') as f:
                for row in data_array:
                    f.write(' '.join([str(float(val)) for val in row]) + '\n')

            print(f"  → Written to {input_file}")

        # Prepare MPC arguments
        args = party_sizes + [num_genes, num_classes]

        print(f"Compiling MPC protocol with arguments: {args}")
        # Compile MPC protocol WITH arguments (needed for program.args)
        # Pass protocol name WITHOUT .mpc extension (MP-SPDZ expects this)
        self.executor.compile_protocol(protocol_name, args=args)

        print(f"Executing MPC binning with {len(party_sizes)} parties...")
        print(f"Party sizes: {party_sizes}")

        result = self.executor.execute_protocol(
            protocol_name,
            num_parties=len(party_sizes),
            args=args
        )

        # Read results from MPC output
        output_dir = os.path.join(self.executor.mpspdz_path, 'Player-Data')

        binned_data_file = os.path.join(output_dir, 'binned_data.txt')
        bin_means_file = os.path.join(output_dir, 'noisy_bin_means.txt')

        if not os.path.exists(binned_data_file):
            raise FileNotFoundError(
                f"MPC binning output not found: {binned_data_file}\n"
                f"The MPC protocol may have failed or produced different output files."
            )

        return binned_data_file, bin_means_file


class MPCMarginalComputer:
    """
    Computes marginals using MPC protocols for Private-PGM
    """

    def __init__(self, mpspdz_path=None, protocol="ring"):
        """
        Initialize MPC Marginal Computer

        Args:
            mpspdz_path: Path to MP-SPDZ installation
            protocol: MPC protocol to use
        """
        self.executor = MPCProtocolExecutor(mpspdz_path, protocol)
        self.splitter = HorizontalDataSplitter(num_parties=2)
        self.binner = MPCBinningComputer(mpspdz_path, protocol)

    def compute_marginals_from_party_files(self, party_data_files, num_genes,
                                           num_classes, target_delta, sigma,
                                           mpc_protocol_file='ppai_msr_noisy_final'):
        """
        Compute marginals using MPC from party data files (SECURE - no data leakage)

        This method ensures raw data never leaves individual parties.
        Each party's data is read separately and input as secret shares.

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
