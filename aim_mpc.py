#!/usr/bin/env python3
"""
AIM-MPC: Adaptive and Iterative Mechanism with Multi-Party Computation

Implements differentially private synthetic data generation using
secure multi-party computation to protect sensitive data.

Based on: https://github.com/sikhapentyala/MPC_SDG
"""

import json
import os
import sys
import time
import argparse
import itertools
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from mbi import Dataset, GraphicalModel, FactoredInference, Domain, Factor
from scipy.optimize import bisect

from data_holder import HDataHolder, VDataHolder, create_parties
from matrix import Identity


# Configuration
PATH_MPC = os.environ.get('MP_SPDZ_PATH', '/home/user/MP-SPDZ/')
PROTOCOL = "ring"


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s)+1)
    )


def downward_closure(Ws):
    """Compute downward closure of a set of marginals"""
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    """Estimate model size in MB"""
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    """
    Compile workload into candidates with scores

    Score = how many workload queries does this marginal help answer
    """
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(candidates, model, size_limit):
    """Filter candidates based on model size constraints"""
    ans = {}
    free_cliques = downward_closure(model.cliques)

    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques

        if cond1 or cond2:
            ans[cl] = candidates[cl]

    return ans


class AIM_MPC:
    """
    Adaptive and Iterative Mechanism with MPC

    Generates synthetic data using:
    - Differentially private marginal measurements
    - Secure multi-party computation for privacy
    - Graphical model inference for consistency
    """

    def __init__(self, epsilon: float, delta: float,
                 rounds: int = None, max_model_size: float = 80,
                 structural_zeros: Dict = None,
                 mpc_path: str = PATH_MPC,
                 protocol: str = PROTOCOL):
        """
        Initialize AIM-MPC mechanism

        Args:
            epsilon: DP epsilon parameter
            delta: DP delta parameter
            rounds: Number of adaptive rounds (default: 16 * num_features)
            max_model_size: Maximum model size in MB
            structural_zeros: Known impossible combinations
            mpc_path: Path to MP-SPDZ installation
            protocol: MPC protocol to use (ring, mascot, etc.)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros or {}
        self.mpc_path = mpc_path
        self.protocol = protocol

        # Compute rho (concentrated DP parameter)
        self.rho = self._compute_rho()

    def _compute_rho(self):
        """Compute concentrated DP parameter from (epsilon, delta)"""
        # Using standard conversion from (ε, δ)-DP to ρ-zCDP
        # ρ = ε² / (2 * log(1/δ))
        return self.epsilon**2 / (2 * np.log(1/self.delta))

    def run(self, party_files: List[str], domain_config: Dict,
            workload: List[Tuple], partition_type: str = "horizontal") -> Dataset:
        """
        Run AIM-MPC to generate synthetic data

        Args:
            party_files: List of CSV files, one per party
            domain_config: Dictionary mapping feature names to domain sizes
            workload: List of marginal queries as tuples of column names
            partition_type: "horizontal" or "vertical" partitioning

        Returns:
            Synthetic dataset
        """
        print("=" * 80)
        print("AIM-MPC: Adaptive and Iterative Mechanism with MPC")
        print("=" * 80)
        print(f"Privacy: (ε={self.epsilon}, δ={self.delta}) → ρ={self.rho:.6f}")
        print(f"Parties: {len(party_files)}")
        print(f"Workload: {len(workload)} marginals")
        print("=" * 80)

        # Create domain
        domain = Domain(domain_config.keys(), domain_config.values())
        rounds = self.rounds or 16 * len(domain)

        # Compile workload into candidates
        candidates = compile_workload(workload)
        print(f"\nGenerated {len(candidates)} candidate marginals")

        # Compute domain sizes for padding
        workload_domain_size = [domain.size(cl) for cl in candidates.keys()]
        max_domain_size = max(workload_domain_size)
        num_of_candidates = len(candidates)

        print(f"Max domain size: {max_domain_size}")
        print(f"Number of candidates: {num_of_candidates}")

        # ===================================================================
        # STEP 1: Create parties and compute local answers
        # ===================================================================
        print("\n" + "=" * 80)
        print("STEP 1: Parties compute local workload answers")
        print("=" * 80)

        parties = create_parties(party_files, domain, partition_type)

        # Each party loads their data
        for party in parties:
            party.load_data()

        # Each party computes answers on their local data
        for party in parties:
            party.compute_answers(candidates, domain, max_domain_size, keep_pad=True)

        # Write inputs for MPC
        player_data_dir = os.path.join(self.mpc_path, 'Player-Data')
        os.makedirs(player_data_dir, exist_ok=True)

        for i, party in enumerate(parties):
            input_file = os.path.join(player_data_dir, f'Input-P{i}-0')
            party.write_mpc_input(input_file)

        # ===================================================================
        # STEP 2: Compile MPC programs
        # ===================================================================
        print("\n" + "=" * 80)
        print("STEP 2: Compile MPC programs")
        print("=" * 80)

        # Identify one-way marginals
        oneway = [cl for cl in candidates if len(cl) == 1]
        oneway_indices = {value: i for i, value in enumerate(candidates) if value in oneway}

        print(f"One-way marginals: {len(oneway)}")

        start_compile = time.time()

        # Compile one-way marginal program
        compile_cmd = (
            f"cd {self.mpc_path} && "
            f"./compile.py -R 64 aim_H_1way {max_domain_size} {num_of_candidates}"
        )
        print(f"\nCompiling: aim_H_1way-{max_domain_size}-{num_of_candidates}")
        ret = os.system(compile_cmd)
        if ret != 0:
            print(f"⚠️  Warning: Compilation returned code {ret}")

        # Compile main iterative program
        compile_cmd = (
            f"cd {self.mpc_path} && "
            f"./compile.py -R 64 aim_H {max_domain_size} {num_of_candidates}"
        )
        print(f"Compiling: aim_H-{max_domain_size}-{num_of_candidates}")
        ret = os.system(compile_cmd)
        if ret != 0:
            print(f"⚠️  Warning: Compilation returned code {ret}")

        print(f"Compile time: {time.time() - start_compile:.2f}s")

        # ===================================================================
        # STEP 3: Initialize with one-way marginals
        # ===================================================================
        print("\n" + "=" * 80)
        print("STEP 3: Measure one-way marginals with MPC")
        print("=" * 80)

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        print(f'Initial σ = {sigma:.4f}, ε = {epsilon:.4f}')

        rho_used = len(oneway) * 0.5 / sigma**2

        for cl in oneway:
            marginal_index = oneway_indices[cl]

            # Write public input
            public_input_file = os.path.join(
                self.mpc_path,
                f'Programs/Public-Input/aim_H_1way-{max_domain_size}-{num_of_candidates}'
            )

            with open(public_input_file, 'w') as f:
                f.write(' '.join(str(num) for num in workload_domain_size))
                f.write("\n")
                f.write(str(round(sigma * pow(2, 16))))
                f.write("\n")
                f.write(str(marginal_index))

            # Run MPC
            run_cmd = (
                f"cd {self.mpc_path} && "
                f"Scripts/{self.protocol}.sh "
                f"aim_H_1way-{max_domain_size}-{num_of_candidates} "
                f"-v > {self.mpc_path}/mpc_out.txt"
            )

            print(f"  Measuring {cl}...", end=" ")
            os.system(run_cmd)

            # Parse output
            n = domain.size(cl)
            with open(os.path.join(self.mpc_path, 'mpc_out.txt')) as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Y"):
                        y_str = line.split(":")[1].strip()[1:-1].split(', ')
                        y = np.array([float(val) for val in y_str])[:n]
                        break

            I = Identity(n)
            measurements.append((I, y, sigma, cl))
            print(f"✓ (size={n})")

        # ===================================================================
        # STEP 4: Build initial model
        # ===================================================================
        print("\n" + "=" * 80)
        print("STEP 4: Build initial graphical model")
        print("=" * 80)

        zeros = self.structural_zeros
        engine = FactoredInference(domain, iters=1000, warm_start=True,
                                   structural_zeros=zeros)
        model = engine.estimate(measurements)

        print(f"Initial model cliques: {model.cliques}")

        # ===================================================================
        # STEP 5: Adaptive iteration
        # ===================================================================
        print("\n" + "=" * 80)
        print("STEP 5: Adaptive marginal selection")
        print("=" * 80)

        t = 0
        terminate = False

        while not terminate:
            t += 1

            # Check remaining budget
            if self.rho - rho_used < 2 * (0.5/sigma**2 + 1.0/8 * epsilon**2):
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True
                print(f"\n[Round {t}] FINAL ROUND - using remaining budget")

            rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
            size_limit = self.max_model_size * rho_used / self.rho

            # Filter candidates by size
            small_candidates = filter_candidates(candidates, model, size_limit)
            small_candidates_indices = {
                value: i for i, value in enumerate(candidates)
                if value in small_candidates.keys()
            }

            print(f"\n[Round {t}] Budget: {rho_used/self.rho:.1%}, "
                  f"Candidates: {len(small_candidates)}, σ={sigma:.4f}")

            # Compute weights and sensitivities
            bias = np.zeros(len(candidates))
            wgt = np.ones(len(candidates))
            sensitivity = []

            for i, cl in zip(small_candidates_indices.values(), small_candidates.keys()):
                wt = small_candidates[cl]
                wgt[i] = wt
                bias[i] = np.sqrt(2/np.pi) * sigma * model.domain.size(cl)
                sensitivity.append(abs(wt))

            max_sensitivity = max(sensitivity) if sensitivity else 1.0

            # Compute estimated answers from current model
            est_ans = []
            for cl in candidates:
                data_vector = model.project(cl).datavector()
                padded = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
                est_ans.append(padded)

            # Write player 0 input (party 0's answers + estimated answers)
            input_p0 = os.path.join(player_data_dir, 'Input-P0-0')
            with open(input_p0, 'w') as f:
                f.write('\n'.join([' '.join([str(int(num)) for num in row])
                                  for row in parties[0].workload_answers]))
                f.write("\n")
                f.write('\n'.join([' '.join([str(num) for num in row])
                                  for row in est_ans]))

            # Write public input for main program
            public_input_file = os.path.join(
                self.mpc_path,
                f'Programs/Public-Input/aim_H-{max_domain_size}-{num_of_candidates}'
            )

            with open(public_input_file, 'w') as f:
                f.write(str(round(epsilon * pow(2, 16))) + "\n")
                f.write(str(round(max_sensitivity * pow(2, 16))) + "\n")
                f.write(str(round(sigma * pow(2, 16))) + "\n")
                f.write(' '.join(str(num) for num in small_candidates_indices.values()) + "\n")
                f.write(' '.join(str(num) for num in workload_domain_size) + "\n")
                f.write(' '.join(str(round(num * pow(2, 16))) for num in bias) + "\n")
                f.write(' '.join(str(round(num * pow(2, 16))) for num in wgt))

            # Run MPC
            run_cmd = (
                f"cd {self.mpc_path} && "
                f"Scripts/{self.protocol}.sh "
                f"aim_H-{max_domain_size}-{num_of_candidates} "
                f"-v > {self.mpc_path}/mpc_out_1.txt"
            )

            os.system(run_cmd)

            # Parse output
            with open(os.path.join(self.mpc_path, 'mpc_out_1.txt')) as f:
                lines = f.readlines()
                ax = 0
                for line in lines:
                    if line.startswith("Ax"):
                        ax = int(line.split(":")[1])
                    if line.startswith("Y"):
                        y_str = line.split(":")[1].strip()[1:-1].split(', ')

            # Get selected marginal
            cl = next((key for key, value in small_candidates_indices.items()
                      if value == ax), None)

            if cl is None:
                print(f"⚠️  Could not find marginal for index {ax}, terminating")
                break

            n = domain.size(cl)
            y = np.array([float(val) for val in y_str])[:n]

            # Add measurement
            Q = Identity(n)
            measurements.append((Q, y, sigma, cl))

            # Update model
            z = model.project(cl).datavector()
            model = engine.estimate(measurements)
            w = model.project(cl).datavector()

            print(f"  Selected: {cl}, Size: {n}, "
                  f"Change: {np.linalg.norm(w-z, 1):.4f}")

            # Adaptive sigma reduction
            if np.linalg.norm(w-z, 1) <= sigma * np.sqrt(2/np.pi) * n:
                print(f"  → Reducing σ: {sigma:.4f} → {sigma/2:.4f}")
                sigma /= 2
                epsilon *= 2

        # ===================================================================
        # STEP 6: Generate synthetic data
        # ===================================================================
        print("\n" + "=" * 80)
        print("STEP 6: Generate synthetic data")
        print("=" * 80)

        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data()

        print(f"✓ Generated {len(synth.df)} synthetic samples")
        print("=" * 80)

        return synth


def default_params():
    """Default parameters"""
    return {
        'party_files': [
            'data/aml/party_1_preprocessed.csv',
            'data/aml/party_2_preprocessed.csv'
        ],
        'domain': 'data/aml/domain.json',
        'epsilon': 1.0,
        'delta': 1e-9,
        'max_model_size': 80,
        'degree': 2,
        'num_marginals': None,
        'max_cells': 10000,
        'save': 'synthetic_aim_mpc.csv'
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='AIM-MPC: Generate synthetic data with MPC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--party_files', nargs='+', help='CSV files for each party')
    parser.add_argument('--domain', help='Domain JSON file')
    parser.add_argument('--epsilon', type=float, help='Privacy epsilon')
    parser.add_argument('--delta', type=float, help='Privacy delta')
    parser.add_argument('--max_model_size', type=float, help='Max model size (MB)')
    parser.add_argument('--degree', type=int, help='Degree of marginals')
    parser.add_argument('--num_marginals', type=int, help='Number of marginals')
    parser.add_argument('--max_cells', type=int, help='Max cells per marginal')
    parser.add_argument('--save', help='Output path for synthetic data')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    # Load domain
    with open(args.domain) as f:
        domain_config = json.load(f)

    domain = Domain(domain_config.keys(), domain_config.values())

    # Create workload
    workload = list(itertools.combinations(domain.attrs, args.degree))
    workload = [cl for cl in workload if domain.size(cl) <= args.max_cells]

    if args.num_marginals is not None:
        prng = np.random.RandomState(0)
        workload = [workload[i] for i in prng.choice(
            len(workload), args.num_marginals, replace=False
        )]

    print(f"Workload: {len(workload)} marginals of degree {args.degree}")

    # Run AIM-MPC
    start_time = time.time()

    mech = AIM_MPC(
        epsilon=args.epsilon,
        delta=args.delta,
        max_model_size=args.max_model_size
    )

    synth = mech.run(
        party_files=args.party_files,
        domain_config=domain_config,
        workload=workload,
        partition_type="horizontal"
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Generated in {elapsed:.2f}s")

    # Save
    if args.save:
        synth.df.to_csv(args.save, index=False)
        print(f"✓ Saved to {args.save}")
