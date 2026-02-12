"""
MPC utilities for Private PGM
Based on the implementation from https://github.com/sikhapentyala/MPC_SDG
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional


class HDataHolder:
    """
    Data holder class for MPC parties (similar to the reference implementation)
    Handles horizontal data partitioning for multi-party computation
    """

    def __init__(self, name: str, data_path: Optional[str] = None,
                 partition_type: str = "horizontal", n: float = 0.5):
        """
        Initialize data holder for an MPC party

        Args:
            name: Name of the party (e.g., "Alice", "Bob")
            data_path: Path to the data CSV file (optional if data will be set directly)
            partition_type: Type of partitioning ("horizontal" or "vertical")
            n: Fraction of data for this party (for horizontal partitioning)
        """
        self.name = name
        self.data_path = data_path
        self.partition_type = partition_type
        self.n = n
        self.data = None
        self.workload_answers = None

    def load_data(self, df: Optional[pd.DataFrame] = None):
        """
        Load data either from file or from a provided DataFrame

        Args:
            df: Optional DataFrame to load directly
        """
        if df is not None:
            self.data = df
        elif self.data_path is not None:
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Either df or data_path must be provided")

    def set_data(self, df: pd.DataFrame):
        """Set data directly from a DataFrame"""
        self.data = df

    def compute_answers(self, candidates, domain, max_domain_size, keep_pad=True):
        """
        Compute workload answers on local data partition
        Similar to the reference implementation

        Args:
            candidates: List of candidate queries/marginals
            domain: Domain specification
            max_domain_size: Maximum domain size for padding
            keep_pad: Whether to keep padding
        """
        # This is a placeholder - actual implementation would compute marginals
        # For now, we'll compute basic statistics that will be used by MPC protocols
        self.workload_answers = []

        # Compute 1-way and 2-way marginals on local data
        # This will be implemented based on the specific requirements
        pass

    def write_to_mpc_input(self, player_id: int, input_dir: str):
        """
        Write data to MPC input files for the specified player

        Args:
            player_id: Player ID (0 or 1 for two-party computation)
            input_dir: Directory for MPC input files
        """
        os.makedirs(input_dir, exist_ok=True)
        input_file = os.path.join(input_dir, f'Input-P{player_id}-0')

        # Convert DataFrame to numpy array and write to file
        # Format expected by MP-SPDZ: space-separated values, one row per line
        if self.data is not None:
            data_array = self.data.values
            with open(input_file, 'w') as f:
                for row in data_array:
                    f.write(' '.join([str(val) for val in row]) + '\n')

    def read_from_mpc_output(self, output_file: str) -> np.ndarray:
        """
        Read results from MPC output file

        Args:
            output_file: Path to MPC output file

        Returns:
            numpy array of results
        """
        results = []
        with open(output_file, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                results.append(values)
        return np.array(results)


def split_data_horizontal(df: pd.DataFrame, n_parties: int = 2,
                         random_seed: Optional[int] = None) -> list:
    """
    Split data horizontally for multiple parties

    Args:
        df: DataFrame to split
        n_parties: Number of parties (default: 2)
        random_seed: Random seed for shuffling

    Returns:
        List of DataFrames, one per party
    """
    if random_seed is not None:
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n_rows = len(df)
    rows_per_party = n_rows // n_parties

    parties_data = []
    for i in range(n_parties):
        start_idx = i * rows_per_party
        if i == n_parties - 1:
            # Last party gets remaining rows
            end_idx = n_rows
        else:
            end_idx = (i + 1) * rows_per_party
        parties_data.append(df.iloc[start_idx:end_idx].reset_index(drop=True))

    return parties_data


def compile_mpc_protocol(mpc_path: str, protocol_name: str, *args) -> int:
    """
    Compile an MPC protocol using MP-SPDZ

    Args:
        mpc_path: Path to MP-SPDZ installation
        protocol_name: Name of the .mpc protocol file (without .mpc extension)
        *args: Arguments to pass to the protocol

    Returns:
        Exit code of compilation
    """
    args_str = ' '.join([str(arg) for arg in args])
    compile_cmd = f"cd {mpc_path} && ./compile.py -R 64 {protocol_name} {args_str}"

    print(f"Compiling MPC protocol: {compile_cmd}")
    exit_code = os.system(compile_cmd)

    if exit_code != 0:
        raise RuntimeError(f"MPC compilation failed with exit code {exit_code}")

    return exit_code


def run_mpc_protocol(mpc_path: str, protocol: str, program_name: str,
                    verbose: bool = True) -> int:
    """
    Run a compiled MPC protocol

    Args:
        mpc_path: Path to MP-SPDZ installation
        protocol: Protocol type (e.g., "ring", "semi2k")
        program_name: Name of the compiled program
        verbose: Whether to print verbose output

    Returns:
        Exit code of execution
    """
    verbose_flag = "-v" if verbose else ""
    output_file = os.path.join(mpc_path, "mpc_out.txt")

    run_cmd = f"cd {mpc_path} && Scripts/{protocol}.sh {program_name} {verbose_flag} > {output_file}"

    print(f"Running MPC protocol: {run_cmd}")
    exit_code = os.system(run_cmd)

    if exit_code != 0:
        raise RuntimeError(f"MPC execution failed with exit code {exit_code}")

    return exit_code
