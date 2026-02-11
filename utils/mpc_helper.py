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

    def compile_protocol(self, mpc_file):
        """
        Compile .mpc protocol file using MP-SPDZ compiler

        Args:
            mpc_file: Path to .mpc file to compile

        Returns:
            bool: True if compilation successful
        """
        compile_script = os.path.join(self.mpspdz_path, 'compile.py')

        if not os.path.exists(compile_script):
            raise FileNotFoundError(f"MP-SPDZ compile.py not found at {compile_script}")

        cmd = [compile_script, mpc_file]

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
            print(f"Protocol execution failed: {e.stderr}")
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

    def compute_marginals_mpc(self, data, num_genes, num_classes,
                               target_delta, sigma, mpc_protocol_file):
        """
        Compute 1-way and 2-way marginals using MPC

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
        # Split data between parties
        data_splits = self.splitter.split_data(data, party_ratios=[0.5, 0.5])

        # Write party data to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            party_files = self.splitter.write_party_data(
                data_splits,
                temp_dir,
                prefix="mpc_input"
            )

            # Compile MPC protocol
            self.executor.compile_protocol(mpc_protocol_file)

            # Extract protocol name from file path
            protocol_name = Path(mpc_protocol_file).stem

            # Execute MPC protocol with arguments
            n_samples = len(data)
            args = [n_samples, num_genes, num_classes]

            result = self.executor.execute_protocol(
                protocol_name,
                num_parties=2,
                args=args
            )

            # Read results from MPC output
            # Note: This assumes the MPC protocol writes results to specific files
            # Adjust based on your actual MPC protocol output format
            output_dir = os.path.join(self.executor.mpspdz_path, 'Player-Data')

            measurements_1way = self._read_mpc_output(
                os.path.join(output_dir, 'marginals_1way.txt'),
                num_genes,
                num_classes
            )

            measurements_2way = self._read_mpc_output(
                os.path.join(output_dir, 'marginals_2way.txt'),
                num_genes * 20  # 4 feature values * 5 label values
            )

        return measurements_1way, measurements_2way

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
