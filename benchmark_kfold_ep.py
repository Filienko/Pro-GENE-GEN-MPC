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
        self.F_STAT_SENSITIVITY = 15.0

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
        n0, n1 = len(df0), len(df1)
        m_genes = df0.shape[1] - 1
        all_classes = pd.concat([df0.iloc[:, -1], df1.iloc[:, -1]])
        n_classes = int(all_classes.max()) + 1

        self._prepare_inputs(df0, df1, n_classes)

        sigma = self._calculate_sigma(epsilon, delta)
        sigma_int = int(sigma * 10000)

        compiler_args = [str(n0), str(n1), str(m_genes), str(n_classes), str(k), str(sigma_int)]
        full_program_name = f"{self.protocol_name}-" + "-".join(compiler_args)

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

        start_time = time.time()
        procs = []
        for p_id in [1, 2]:
            cmd = [self.binary_name, str(p_id), full_program_name]
            procs.append(subprocess.Popen(
                cmd, cwd=self.mpspdz_root,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ))
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
        self.results = []          # Experiment 1: varying M and K
        self.epsilon_results = []  # Experiment 2: varying epsilon

    @staticmethod
    def _drop_non_numeric(df):
        """Return a copy of df with only numeric feature columns + the last (label) column."""
        feature_cols = df.columns[:-1]
        label_col = df.columns[-1]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols + [label_col]]

    def _run_clear(self, df, k):
        start = time.time()
        df = self._drop_non_numeric(df)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            f_vals, _ = f_classif(X, y)
            f_vals = np.nan_to_num(f_vals)
        ranks = np.argsort(f_vals)[::-1][:k]
        return ranks.tolist(), time.time() - start

    # --------------------------------------------------------------------------
    # Experiment 1: Sweep feature sizes and K values (fixed epsilon)
    # --------------------------------------------------------------------------
    def run_benchmark(self, feature_sizes, k_vals, epsilon=10.0):
        total_feats_avail = self.df.shape[1] - 1
        print(f"\n{'='*65}")
        print(f">> EXPERIMENT 1: Varying M and K  (epsilon={epsilon})")
        print(f"{'='*65}")
        print(f"{'Feats':<6} | {'K':<4} | {'Fold':<4} | {'Clear(s)':<8} | {'MPC(s)':<8} | {'Overlap':<8}")
        print("-" * 65)

        for m in feature_sizes:
            current_m = min(m, total_feats_avail)
            cols = list(range(current_m)) + [total_feats_avail]
            df_subset = self._drop_non_numeric(self.df.iloc[:, cols])
            current_m = df_subset.shape[1] - 1  # recount after dropping non-numeric

            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            X = df_subset.iloc[:, :-1].values
            y = df_subset.iloc[:, -1].values

            for k in k_vals:
                if k >= current_m: continue

                fold_idx = 0
                for train_index, _ in skf.split(X, y):
                    fold_idx += 1
                    df_train = df_subset.iloc[train_index]
                    split_point = len(df_train) // 2
                    df_p0 = df_train.iloc[:split_point]
                    df_p1 = df_train.iloc[split_point:]

                    clear_rnk, t_clear = self._run_clear(df_train, k)
                    mpc_rnk, t_mpc = self.orchestrator.run_protocol(df_p0, df_p1, k, epsilon=epsilon)

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

    # --------------------------------------------------------------------------
    # Experiment 2: Fixed M=1000, K=50 — sweep epsilon values
    # --------------------------------------------------------------------------
    def run_epsilon_benchmark(self, epsilon_vals, fixed_m=1000, fixed_k=50, delta=1e-5):
        total_feats_avail = self.df.shape[1] - 1
        current_m = min(fixed_m, total_feats_avail)

        print(f"\n{'='*65}")
        print(f">> EXPERIMENT 2: Varying Epsilon  (M={current_m}, K={fixed_k})")
        print(f"{'='*65}")
        print(f"{'Epsilon':<10} | {'Fold':<4} | {'Clear(s)':<8} | {'MPC(s)':<8} | {'Overlap':<8} | {'Sigma':<10}")
        print("-" * 65)

        cols = list(range(current_m)) + [total_feats_avail]
        df_subset = self._drop_non_numeric(self.df.iloc[:, cols])
        current_m = df_subset.shape[1] - 1  # recount after dropping non-numeric

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        X = df_subset.iloc[:, :-1].values
        y = df_subset.iloc[:, -1].values

        if fixed_k >= current_m:
            print(f"[Warning] K={fixed_k} >= M={current_m}, skipping epsilon experiment.")
            return

        for epsilon in epsilon_vals:
            sigma = self.orchestrator._calculate_sigma(epsilon, delta)
            fold_idx = 0
            for train_index, _ in skf.split(X, y):
                fold_idx += 1
                df_train = df_subset.iloc[train_index]
                split_point = len(df_train) // 2
                df_p0 = df_train.iloc[:split_point]
                df_p1 = df_train.iloc[split_point:]

                clear_rnk, t_clear = self._run_clear(df_train, fixed_k)
                mpc_rnk, t_mpc = self.orchestrator.run_protocol(
                    df_p0, df_p1, fixed_k, epsilon=epsilon, delta=delta
                )

                overlap_count = len(set(clear_rnk) & set(mpc_rnk))
                accuracy = overlap_count / fixed_k if fixed_k > 0 else 0
                print(f"{epsilon:<10} | {fold_idx:<4} | {t_clear:<8.4f} | {t_mpc:<8.4f} | {overlap_count}/{fixed_k} | {sigma:<10.4f}")

                self.epsilon_results.append({
                    'epsilon': epsilon,
                    'sigma': sigma,
                    'fold': fold_idx,
                    'time_clear': t_clear,
                    'time_mpc': t_mpc,
                    'accuracy': accuracy
                })

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    def plot_results(self):
        output_dir = "benchmark_plots"
        os.makedirs(output_dir, exist_ok=True)
        if self.results:
            pd.DataFrame(self.results).to_csv(f"{output_dir}/exp1_results.csv", index=False)

        if self.epsilon_results:
            pd.DataFrame(self.epsilon_results).to_csv(f"{output_dir}/exp2_epsilon_results.csv", index=False)


        # --- Experiment 1 plots ---
        if self.results:
            df_res = pd.DataFrame(self.results)
            df_agg = df_res.groupby(['features', 'k_deg']).mean().reset_index()

            print(f"\n>> Generating Experiment 1 plots in {output_dir}/ ...")

            plt.figure(figsize=(10, 6))
            for feat in df_agg['features'].unique():
                subset = df_agg[df_agg['features'] == feat]
                plt.plot(subset['k_deg'], subset['time_mpc'], marker='o', label=f'M={feat}')
            plt.xlabel('Number of Top Genes (K)')
            plt.ylabel('Time (seconds)')
            plt.title('MPC Execution Time vs K (by Feature Size)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exp1_time_vs_k.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            for k in df_agg['k_deg'].unique():
                subset = df_agg[df_agg['k_deg'] == k]
                plt.plot(subset['features'], subset['time_mpc'], marker='o', label=f'K={k}')
            plt.xlabel('Total Feature Size (M)')
            plt.ylabel('Time (seconds)')
            plt.title('MPC Execution Time vs Feature Size (by K)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exp1_time_vs_features.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            for feat in df_agg['features'].unique():
                subset = df_agg[df_agg['features'] == feat]
                plt.plot(subset['k_deg'], subset['accuracy'], marker='o', label=f'M={feat}')
            plt.xlabel('Number of Top Genes (K)')
            plt.ylabel('Overlap Accuracy (0.0 - 1.0)')
            plt.title('Accuracy vs K (by Feature Size)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exp1_acc_vs_k.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            for k in df_agg['k_deg'].unique():
                subset = df_agg[df_agg['k_deg'] == k]
                plt.plot(subset['features'], subset['accuracy'], marker='o', label=f'K={k}')
            plt.xlabel('Total Feature Size (M)')
            plt.ylabel('Overlap Accuracy (0.0 - 1.0)')
            plt.title('Accuracy vs Feature Size (by K)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exp1_acc_vs_features.png")
            plt.close()

        # --- Experiment 2 plots ---
        if self.epsilon_results:
            df_eps = pd.DataFrame(self.epsilon_results)
            df_eps_agg = df_eps.groupby(['epsilon', 'sigma']).mean().reset_index()
            df_eps_agg = df_eps_agg.sort_values('epsilon')

            print(f">> Generating Experiment 2 plots in {output_dir}/ ...")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Experiment 2: Privacy Budget (ε) vs Performance\n(M=1000 genes, K=50 DEGs)', fontsize=13)

            # Left: Accuracy vs Epsilon
            ax = axes[0]
            ax.plot(df_eps_agg['epsilon'], df_eps_agg['accuracy'], marker='o',
                    color='steelblue', linewidth=2, markersize=8)
            ax.set_xscale('log')
            ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
            ax.set_ylabel('Overlap Accuracy (0.0 – 1.0)', fontsize=12)
            ax.set_title('Accuracy vs Privacy Budget')
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            # Annotate sigma on each point
            for _, row in df_eps_agg.iterrows():
                ax.annotate(f"σ={row['sigma']:.1f}",
                            xy=(row['epsilon'], row['accuracy']),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=8, color='dimgray')

            # Right: MPC Time vs Epsilon
            ax = axes[1]
            ax.plot(df_eps_agg['epsilon'], df_eps_agg['time_mpc'], marker='s',
                    color='tomato', linewidth=2, markersize=8)
            ax.set_xscale('log')
            ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
            ax.set_ylabel('MPC Execution Time (seconds)', fontsize=12)
            ax.set_title('MPC Time vs Privacy Budget')
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/exp2_epsilon_sweep.png", dpi=150)
            plt.close()

            # Also save a standalone accuracy-only plot
            plt.figure(figsize=(9, 5))
            plt.plot(df_eps_agg['epsilon'], df_eps_agg['accuracy'],
                     marker='o', color='steelblue', linewidth=2, markersize=9)
            plt.xscale('log')
            plt.xticks(df_eps_agg['epsilon'].tolist(),
                       [str(int(e)) for e in df_eps_agg['epsilon'].tolist()])
            plt.xlabel('Privacy Budget (ε)  [log scale]', fontsize=12)
            plt.ylabel('Overlap Accuracy (0.0 – 1.0)', fontsize=12)
            plt.title('DP Accuracy vs Privacy Budget\n(M=1000, K=50 DEGs)', fontsize=13)
            for _, row in df_eps_agg.iterrows():
                plt.annotate(f"ε={int(row['epsilon'])}\nσ={row['sigma']:.2f}",
                             xy=(row['epsilon'], row['accuracy']),
                             xytext=(6, -14), textcoords='offset points',
                             fontsize=8, color='dimgray')
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exp2_accuracy_vs_epsilon.png", dpi=150)
            plt.close()

        print(">> Done. All plots saved.")


# ==============================================================================
# 3. MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to single CSV dataset")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip Experiment 1 (M/K sweep)")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip Experiment 2 (epsilon sweep)")
    args = parser.parse_args()

    runner = KFoldBenchmarkRunner(args.file, n_folds=args.folds)

    # ------------------------------------------------------------------
    # Experiment 1: Vary total gene count (M) and number of DEGs (K)
    # ------------------------------------------------------------------
    if not args.skip_exp1:
        SIZES  = [100, 500, 1000, 10000]
        K_VALS = [10, 50, 100]
        runner.run_benchmark(feature_sizes=SIZES, k_vals=K_VALS, epsilon=10.0)

    # ------------------------------------------------------------------
    # Experiment 2: Fixed M=1000, K=50 — vary epsilon (privacy budget)
    # ------------------------------------------------------------------
    if not args.skip_exp2:
        EPSILON_VALS = [1, 10, 100, 1000]
        runner.run_epsilon_benchmark(
            epsilon_vals=EPSILON_VALS,
            fixed_m=1000,
            fixed_k=100,
            delta=1e-3
        )

    runner.plot_results()
