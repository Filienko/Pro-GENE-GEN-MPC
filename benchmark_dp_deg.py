#!/usr/bin/env python3
import os
import time
import math
import argparse
import subprocess
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif

# ==============================================================================
# 1. ORCHESTRATOR CLASS
# ==============================================================================
class SecureDEGOrchestrator:
    def __init__(self, mpspdz_root="mpc_spdz", protocol_name="dp_topk"):
        self.mpspdz_root = os.path.abspath(mpspdz_root)
        self.protocol_name = protocol_name
        self.binary_name = "./replicated-ring-party.x"
        self.compile_script = "./compile.py"
        self.F_STAT_SENSITIVITY = 10.0

    def _calculate_sigma(self, epsilon, delta):
        if epsilon <= 0: return 1000.0
        val = math.sqrt(2 * math.log(1.25 / delta))
        return (self.F_STAT_SENSITIVITY * val) / epsilon

    def _prepare_inputs(self, df0, df1, n_classes):
        data_dir = os.path.join(self.mpspdz_root, "Player-Data")
        os.makedirs(data_dir, exist_ok=True)

        def _write(df, pid):
            path = os.path.join(data_dir, f"Input-P{pid}-0")
            target = df.columns[-1]
            X = df.drop(columns=[target]).values
            y = df[target].values.astype(int)

            with open(path, "w") as f:
                for i in range(len(df)):
                    y_vec = [0] * n_classes
                    if y[i] < n_classes:
                        y_vec[y[i]] = 1
                    line = " ".join(map(str, X[i])) + " " + " ".join(map(str, y_vec))
                    f.write(line + "\n")

        _write(df0, 0)
        _write(df1, 1)
        with open(os.path.join(data_dir, "Input-P2-0"), "w") as f:
            f.write("")

    def run_protocol(self, df0, df1, k, epsilon=10.0, delta=1e-5):
        # 1. Setup
        n0, n1 = len(df0), len(df1)
        m_genes = df0.shape[1] - 1
        all_classes = pd.concat([df0.iloc[:, -1], df1.iloc[:, -1]])
        n_classes = int(all_classes.max()) + 1
        
        self._prepare_inputs(df0, df1, n_classes)
        
        sigma = self._calculate_sigma(epsilon, delta)
        sigma_int = int(sigma * 10000)

        # 2. Compile
        # Note: We cast args to str to pass to subprocess
        compiler_args = [str(n0), str(n1), str(m_genes), str(n_classes), str(k), str(sigma_int)]
        full_program_name = f"{self.protocol_name}-" + "-".join(compiler_args)

        print(f"   [Compiling] {full_program_name}...")
        try:
            subprocess.run(
                [self.compile_script, "-R", "64", self.protocol_name] + compiler_args,
                cwd=self.mpspdz_root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"   [Error] Compilation Failed:\n{e.stderr.decode()}")
            return [], 0.0

        # 3. Execute
        start_time = time.time()
        procs = []

        # Start Parties 1 & 2
        for p_id in [1, 2]:
            cmd = [self.binary_name, str(p_id), full_program_name]
            procs.append(subprocess.Popen(
                cmd, cwd=self.mpspdz_root, 
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ))

        # Start Coordinator (Party 0) and CAPTURE OUTPUT
        cmd_p0 = [self.binary_name, "0", full_program_name]
        try:
            result = subprocess.run(
                cmd_p0, cwd=self.mpspdz_root, capture_output=True, text=True
            )
        finally:
            for p in procs: p.terminate()
            for p in procs: p.wait()

        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"   [Error] MPC Failed (Code {result.returncode}):")
            print(result.stderr) # Print stderr to see what happened
            return [], duration

        # 4. Parse Output (UPDATED LOGIC)
        ranking = []
        raw_output = result.stdout
        
        for line in raw_output.splitlines():
            # MATCHING: "DP Rank <step>: Feature <index>"
            if "DP Rank" in line and "Feature" in line:
                try:
                    # Split by "Feature " and take the last part
                    # Example: "DP Rank 0: Feature 42" -> parts=["DP Rank 0: ", "42"]
                    parts = line.split("Feature")
                    idx_str = parts[-1].strip()
                    ranking.append(int(idx_str))
                except ValueError:
                    continue

        # DEBUG: If ranking is empty, show us what the MPC actually printed
        if not ranking:
            print("   [Warning] No features found. Raw MPC Output:")
            print(raw_output)

        return ranking, duration

# ==============================================================================
# 2. BENCHMARK RUNNER
# ==============================================================================
class BenchmarkRunner:
    def __init__(self, file0, file1):
        self.df0 = pd.read_csv(file0)
        self.df1 = pd.read_csv(file1)
        self.orchestrator = SecureDEGOrchestrator()

    def _run_clear(self, df, k):
        start = time.time()
        # Assume last column is target, rest are features
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Calculate F-Statistics
        with np.errstate(divide='ignore', invalid='ignore'):
            f_vals, _ = f_classif(X, y)
            f_vals = np.nan_to_num(f_vals)
            
        # Get top k indices
        ranks = np.argsort(f_vals)[::-1][:k]
        return ranks.tolist(), time.time() - start

    def run_benchmark(self, sizes, k_vals):
        print(f"{'Features':<10} | {'K':<5} | {'Clear (s)':<10} | {'MPC (s)':<10} | {'Overlap':<10}")
        print("-" * 65)

        total_feats = self.df0.shape[1] - 1

        for m in sizes:
            # Ensure we don't ask for more features than exist
            current_m = min(m, total_feats)
            
            # Subset features (always keep the last column as target)
            cols = list(range(current_m)) + [total_feats]
            df0_sub = self.df0.iloc[:, cols]
            df1_sub = self.df1.iloc[:, cols]
            df_comb = pd.concat([df0_sub, df1_sub])

            for k in k_vals:
                if k >= current_m: continue

                # Run Cleartext Baseline
                clear_rnk, t_clear = self._run_clear(df_comb, k)

                # Run Secure Protocol
                mpc_rnk, t_mpc = self.orchestrator.run_protocol(df0_sub, df1_sub, k)

                # Metrics: Calculate overlap of top-k indices
                overlap = len(set(clear_rnk) & set(mpc_rnk))
                
                print(f"{current_m:<10} | {k:<5} | {t_clear:<10.4f} | {t_mpc:<10.4f} | {overlap}/{k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file0", help="Path to Party 0 CSV")
    parser.add_argument("file1", help="Path to Party 1 CSV")
    args = parser.parse_args()

    # Define benchmark parameters
    # Adjust sizes based on your actual dataset width
    runner = BenchmarkRunner(args.file0, args.file1)
    runner.run_benchmark(sizes=[100, 500, 1000], k_vals=[10, 50, 500])
