#!/usr/bin/env python3
"""
DataHolder class for MPC-based synthetic data generation

Each party creates a DataHolder object that:
1. Loads their private data
2. Computes workload answers locally
3. Writes inputs for MPC computation
"""

import numpy as np
import pandas as pd
from mbi import Dataset, Domain
from typing import List, Tuple, Dict


class DataHolder:
    """
    Represents a single party in the MPC computation

    Each party:
    - Holds their own private data
    - Computes marginals/answers on their data locally
    - Provides inputs for MPC without revealing raw data
    """

    def __init__(self, name: str, data_path: str, partition_type: str = "horizontal", n: float = 1.0):
        """
        Initialize a data holder

        Args:
            name: Party name (e.g., "Alice", "Bob")
            data_path: Path to this party's CSV file
            partition_type: "horizontal" or "vertical" partitioning
            n: Fraction of total data this party holds (for horizontal)
        """
        self.name = name
        self.data_path = data_path
        self.partition_type = partition_type
        self.n = n

        self.data = None
        self.workload_answers = None
        self.total_samples = 0
        self.weights = None  # For weighted sampling (optional)

    def load_data(self):
        """
        Load this party's private data

        Note: This loads the FULL file, but each party only keeps their portion
        based on horizontal/vertical partitioning
        """
        print(f"[{self.name}] Loading data from {self.data_path}")

        df = pd.read_csv(self.data_path)

        # For horizontal partitioning: split by rows
        if self.partition_type == "horizontal":
            n = int(df.shape[0] * self.n)
            if self.name == "Alice":
                self.data = df.iloc[0:n, :]
            elif self.name == "Bob":
                self.data = df.iloc[n:df.shape[0], :]
            else:
                # For additional parties, split evenly
                self.data = df
        else:
            # For vertical partitioning: split by columns
            n = int(df.shape[1] * self.n)
            if self.name == "Alice":
                self.data = df.iloc[:, 0:n]
            elif self.name == "Bob":
                self.data = df.iloc[:, n:df.shape[1]]
            else:
                self.data = df

        self.total_samples = self.data.shape[0]
        print(f"[{self.name}] Loaded {len(self.data)} samples, {len(self.data.columns)} columns")

    def number_of_samples(self):
        """Get number of samples this party holds"""
        self.total_samples = self.data.shape[0]
        return self.total_samples

    def compute_answers(self, candidates: Dict[Tuple, float], domain: Domain,
                       max_domain_size: int, keep_pad: bool = True, flatten: bool = True):
        """
        Compute workload answers (marginals) on this party's local data

        Uses numpy histogramdd to compute histogram counts for each marginal.

        Args:
            candidates: Dictionary of candidate marginals {(col1, col2, ...): weight}
            domain: Data domain specification
            max_domain_size: Maximum domain size for padding
            keep_pad: Whether to pad answers to max_domain_size
            flatten: Whether to flatten multi-dimensional histograms
        """
        print(f"[{self.name}] Computing answers for {len(candidates)} candidates")

        if self.data is None:
            raise ValueError(f"[{self.name}] Must call load_data() first")

        self.workload_answers = []

        for cl in candidates.keys():
            # Get the shape for this marginal
            shape = domain.project(cl).shape

            # Create bins: [0, 1, 2, ..., n] for each dimension
            bins = [range(n + 1) for n in shape]

            # Compute histogram on this party's local data
            # This is the KEY: each party only sees their own data!
            ans = np.histogramdd(
                self.data[list(cl)].values,
                bins,
                weights=None  # Equal weights for all samples
            )[0]

            # Flatten if needed
            data_vector = ans.flatten() if flatten else ans

            if keep_pad:
                # Pad to max_domain_size for uniform MPC input
                padded = np.pad(
                    data_vector,
                    (0, max_domain_size - len(data_vector)),
                    'constant'
                )
                self.workload_answers.append(padded)
            else:
                self.workload_answers.append(data_vector)

        print(f"[{self.name}] Computed {len(self.workload_answers)} answers")
        print(f"[{self.name}] Answer shape: ({len(self.workload_answers)}, {len(self.workload_answers[0])})")

    def write_mpc_input(self, output_path: str):
        """
        Write this party's workload answers to MPC input file

        Args:
            output_path: Path to write input file (e.g., 'Player-Data/Input-P0-0')
        """
        if self.workload_answers is None:
            raise ValueError(f"[{self.name}] Must call compute_answers() first")

        print(f"[{self.name}] Writing MPC input to {output_path}")

        with open(output_path, 'w') as outfile:
            # Each row is a workload answer (marginal histogram)
            # Format: space-separated numbers, one row per marginal
            outfile.write('\n'.join([
                ' '.join([str(int(num)) for num in row])
                for row in self.workload_answers
            ]))

        print(f"[{self.name}] Wrote {len(self.workload_answers)} rows")

    def get_data_stats(self) -> Dict:
        """Get statistics about this party's data"""
        if self.data is None:
            return {}

        return {
            'name': self.name,
            'num_samples': len(self.data),
            'num_features': len(self.data.columns),
            'columns': list(self.data.columns),
            'label_distribution': self.data.iloc[:, -1].value_counts().to_dict() if len(self.data.columns) > 0 else {}
        }


class HDataHolder(DataHolder):
    """
    Horizontally partitioned data holder

    Each party holds different samples (rows) but same features (columns)
    """

    def __init__(self, name: str, data_path: str, partition_type: str = "horizontal", n: float = 0.5):
        super().__init__(name, data_path, partition_type, n)


class VDataHolder(DataHolder):
    """
    Vertically partitioned data holder

    Each party holds different features (columns) but same samples (rows)
    """

    def __init__(self, name: str, data_path: str, partition_type: str = "vertical", n: float = 0.5):
        super().__init__(name, data_path, partition_type, n)


def create_parties(party_files: List[str], domain: Domain,
                   partition_type: str = "horizontal") -> List[DataHolder]:
    """
    Create DataHolder objects for all parties

    Args:
        party_files: List of CSV files, one per party
        domain: Data domain specification
        partition_type: "horizontal" or "vertical"

    Returns:
        List of DataHolder objects
    """
    party_names = ["Alice", "Bob", "Charlie", "Dave"][:len(party_files)]

    parties = []
    for name, file_path in zip(party_names, party_files):
        if partition_type == "horizontal":
            party = HDataHolder(name, file_path, "horizontal", n=1.0/len(party_files))
        else:
            party = VDataHolder(name, file_path, "vertical", n=1.0/len(party_files))

        parties.append(party)

    return parties


if __name__ == '__main__':
    # Example usage
    import json

    # Load domain
    domain_config = {
        'feature1': 10,
        'feature2': 5,
        'label': 3
    }
    domain = Domain(domain_config.keys(), domain_config.values())

    # Create parties
    alice = HDataHolder("Alice", "party_1.csv", "horizontal", n=0.5)
    bob = HDataHolder("Bob", "party_2.csv", "horizontal", n=0.5)

    # Load data
    alice.load_data()
    bob.load_data()

    # Compute answers
    candidates = {
        ('feature1',): 1.0,
        ('feature2',): 1.0,
        ('label',): 1.0,
        ('feature1', 'label'): 1.0
    }

    max_domain_size = max([domain.size(cl) for cl in candidates.keys()])

    alice.compute_answers(candidates, domain, max_domain_size, keep_pad=True)
    bob.compute_answers(candidates, domain, max_domain_size, keep_pad=True)

    # Write MPC inputs
    alice.write_mpc_input('Player-Data/Input-P0-0')
    bob.write_mpc_input('Player-Data/Input-P1-0')

    print("\n✓ Ready for MPC computation!")
