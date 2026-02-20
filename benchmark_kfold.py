#!/usr/bin/env python3
import os
import time
import math
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# 1. ORCHESTRATOR (Manages MP-SPDZ Interaction)
# ==============================================================================
class SecureDEGOrchestrator:
    def __init__(self, mpspdz_root="mpc_spdz", protocol_name="dp_topk"):
        self.mpspdz_root = os.path.abspath(mpspdz_root)
        self.protocol_name = protocol_name
        self.binary_name = "./replicated-ring-party.x"
        self.compile_script = "./compile.py"
        self.F_STAT_SENSITIVITY = 20.0 # Adjust based on your privacy needs

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
        # We re-compile for every fold because N might change slightly in K-Fold
        compiler_args = [str(n0), str(n1), str(m_genes), str(n_classes), str(k), str(sigma_int)]
        full_program_name = f"{self.protocol_name}-" + "-".join(compiler_args)

        # print(f"   [Compiling] {full_program_name}...")
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

        # Start Coordinator (Party 0)
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
            print(result.stderr)
            return [], duration

        # 4. Parse Output
        ranking = []
        for line in result.stdout.splitlines():
            if "DP Rank" in line and "Feature" in line:
                try:
                    parts = line.split("Feature")
                    ranking.append(int(parts[-1].strip()))
                except ValueError:
                    continue
        
        return ranking, duration

# ==============================================================================
# 2. K-FOLD BENCHMARK RUNNER
# ==============================================================================
class KFoldBenchmarkRunner:
    def __init__(self, filepath, n_folds=5):
        print(f">> Loading dataset: {filepath}")
        self.df = pd.read_csv(filepath)
        self.n_folds = n_folds
        self.orchestrator = SecureDEGOrchestrator()
        self.results = [] # Stores dicts of results

    def _run_clear(self, df, k):
        start = time.time()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            f_vals, _ = f_classif(X, y)
            f_vals = np.nan_to_num(f_vals)
        ranks = np.argsort(f_vals)[::-1][:k]
        return ranks.tolist(), time.time() - start

    def run_benchmark(self, feature_sizes, k_vals):
        total_feats_avail = self.df.shape[1] - 1
        print(f">> Starting {self.n_folds}-Fold Benchmark")
        print(f"{'Original Total':<6} | {'DEGs':<4} | {'Fold':<4} | {'Clear(s)':<8} | {'MPC(s)':<8} | {'Overlap':<8}")
        print("-" * 65)

        for m in feature_sizes:
            current_m = min(m, total_feats_avail)
            
            # Subset features + Target (last col)
            cols = list(range(current_m)) + [total_feats_avail]
            df_subset = self.df.iloc[:, cols]
            
            # Stratified K-Fold ensures classes are balanced in splits
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            X = df_subset.iloc[:, :-1].values
            y = df_subset.iloc[:, -1].values

            for k in k_vals:
                if k >= current_m: continue
                
                fold_idx = 0
                for train_index, test_index in skf.split(X, y):
                    fold_idx += 1
                    
                    # Get Training Data for this fold
                    df_train = df_subset.iloc[train_index]
                    
                    # SIMULATE MPC SPLIT: Split Training Data into Party 0 and Party 1
                    # We just split the dataframe in half
                    split_point = len(df_train) // 2
                    df_p0 = df_train.iloc[:split_point]
                    df_p1 = df_train.iloc[split_point:]

                    # 1. Cleartext Baseline (on full training set)
                    clear_rnk, t_clear = self._run_clear(df_train, k)

                    # 2. MPC Protocol (on split parties)
                    mpc_rnk, t_mpc = self.orchestrator.run_protocol(df_p0, df_p1, k)

                    # 3. Metrics
                    overlap_count = len(set(clear_rnk) & set(mpc_rnk))
                    accuracy = overlap_count / k if k > 0 else 0

                    print(f"{current_m:<6} | {k:<4} | {fold_idx:<4} | {t_clear:<8.4f} | {t_mpc:<8.4f} | {overlap_count}/{k}")

                    self.results.append({
                        'features': current_m,
                        'k_deg': k,
                        'fold': fold_idx,
                        'time_clear': t_clear,
                        'time_mpc': t_mpc,
                        'accuracy': accuracy
                    })

    def plot_results(self):
        df_res = pd.DataFrame(self.results)
        if df_res.empty:
            print("No results to plot.")
            return

        # Aggregate over folds (Mean)
        df_agg = df_res.groupby(['features', 'k_deg']).mean().reset_index()
        
        output_dir = "benchmark_plots"
        os.makedirs(output_dir, exist_ok=True)
        print(f">> Generating plots in {output_dir}/ ...")

        # Plot 1: MPC Time vs K_DEG (Grouped by Features)
        plt.figure(figsize=(10, 6))
        for feat in df_agg['features'].unique():
            subset = df_agg[df_agg['features'] == feat]
            plt.plot(subset['k_deg'], subset['time_mpc'], marker='o', label=f'M={feat}')
        plt.xlabel('Number of Top Genes (K)')
        plt.ylabel('Time (seconds)')
        plt.title('MPC Execution Time vs K (by Feature Size)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/time_vs_k.png")
        plt.close()

        # Plot 2: MPC Time vs Feature Size (Grouped by K_DEG)
        plt.figure(figsize=(10, 6))
        for k in df_agg['k_deg'].unique():
            subset = df_agg[df_agg['k_deg'] == k]
            plt.plot(subset['features'], subset['time_mpc'], marker='o', label=f'K={k}')
        plt.xlabel('Total Feature Size (M)')
        plt.ylabel('Time (seconds)')
        plt.title('MPC Execution Time vs Feature Size (by K)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/time_vs_features.png")
        plt.close()

        # Plot 3: Accuracy vs K_DEG (Grouped by Features)
        plt.figure(figsize=(10, 6))
        for feat in df_agg['features'].unique():
            subset = df_agg[df_agg['features'] == feat]
            plt.plot(subset['k_deg'], subset['accuracy'], marker='o', label=f'M={feat}')
        plt.xlabel('Number of Top Genes (K)')
        plt.ylabel('Overlap Accuracy (0.0 - 1.0)')
        plt.title('Accuracy vs K (by Feature Size)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/acc_vs_k.png")
        plt.close()

        # Plot 4: Accuracy vs Feature Size (Grouped by K_DEG)
        plt.figure(figsize=(10, 6))
        for k in df_agg['k_deg'].unique():
            subset = df_agg[df_agg['k_deg'] == k]
            plt.plot(subset['features'], subset['accuracy'], marker='o', label=f'K={k}')
        plt.xlabel('Total Feature Size (M)')
        plt.ylabel('Overlap Accuracy (0.0 - 1.0)')
        plt.title('Accuracy vs Feature Size (by K)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/acc_vs_features.png")
        plt.close()
        
        print(">> Done.")

# ==============================================================================
# 3. MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to single CSV dataset")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    args = parser.parse_args()

    # Define benchmark parameters here
    SIZES = [100, 500, 1000, 10000]
    K_VALS = [10, 50, 100]

    runner = KFoldBenchmarkRunner(args.file, n_folds=args.folds)
    runner.run_benchmark(feature_sizes=SIZES, k_vals=K_VALS)
    runner.plot_results()
