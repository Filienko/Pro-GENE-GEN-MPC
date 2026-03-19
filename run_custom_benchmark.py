import os
import time
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set this to the name of your MP-SPDZ folder
MPC_DIR = "mpc_spdz"

def get_time_and_memory(cmd, cwd="."):
    """Runs a command via /usr/bin/time to capture Execution Time and Max RAM (RSS)"""
    full_cmd = ["/usr/bin/time", "-v"] + cmd
    start_time = time.time()
    
    res = subprocess.run(full_cmd, capture_output=True, text=True, cwd=cwd)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    mem_mb = 0
    
    for line in res.stderr.split('\n'):
        if "Maximum resident set size (kbytes):" in line:
            mem_mb = int(line.split(':')[-1].strip()) / 1024.0
            break
            
    if res.returncode != 0:
        print(f"\n{'='*60}")
        print(f"[!] ERROR RUNNING COMMAND: {' '.join(cmd)}")
        print(f"{'='*60}")
        print(f"--- STDOUT ---\n{res.stdout}")
        print(f"--- STDERR ---\n{res.stderr}")
        
        if "Scripts/" in cmd[0]:
            script_name = cmd[1]
            log_file_p0 = os.path.join(cwd, "logs", f"{script_name}-0")
            log_file_p1 = os.path.join(cwd, "logs", f"{script_name}-1")
            
            if os.path.exists(log_file_p0):
                print(f"\n--- TAIL OF Player 0 Log (logs/{script_name}-0) ---")
                with open(log_file_p0, "r") as f:
                    print("".join(f.readlines()[-20:]))
            if os.path.exists(log_file_p1):
                print(f"\n--- TAIL OF Player 1 Log (logs/{script_name}-1) ---")
                with open(log_file_p1, "r") as f:
                    print("".join(f.readlines()[-20:]))
                    
        print(f"{'='*60}\n")
        exit(1)
        
    return elapsed_time, mem_mb

def write_shares(df, num_features):
    """Splits the dataframe into two halves and writes to MP-SPDZ Input-Px-0 format"""
    player_data_dir = os.path.join(MPC_DIR, 'Player-Data')
    os.makedirs(player_data_dir, exist_ok=True)
    
    half = len(df) // 2
    df0 = df.iloc[:half]
    df1 = df.iloc[half:] 
    
    def write_to_file(data, player):
        filepath = os.path.join(player_data_dir, f'Input-P{player}-0')
        with open(filepath, 'w') as f:
            for _, row in data.iterrows():
                f.write(" ".join([str(val) for val in row]) + "\n")

    write_to_file(df0, 0)
    write_to_file(df1, 1)
    
    return len(df0), len(df1)

def plot_results(csv_file, x_col, extrap_limit, out_png):
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = [
        ('Compile Time Mean (s)', 'Compile Time Std (s)', axes[0, 0]), 
        ('Exec Time Mean (s)', 'Exec Time Std (s)', axes[0, 1]), 
        ('Compile RAM Mean (MB)', 'Compile RAM Std (MB)', axes[1, 0]), 
        ('Exec RAM Mean (MB)', 'Exec RAM Std (MB)', axes[1, 1])
    ]
    
    extrapolate_x = np.linspace(0, extrap_limit, 100)
    
    for mean_col, std_col, ax in metrics:
        for script in df['Script'].unique():
            sub = df[df['Script'] == script]
            x = sub[x_col].values
            y_mean = sub[mean_col].values
            y_std = sub[std_col].values
            
            # Scatter actual points with error bars
            ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5, label=f"{script} (Actual)")
            
            # Linear regression extrapolation
            if len(x) > 1:
                coefs = np.polyfit(x, y_mean, 1)
                poly = np.poly1d(coefs)
                ax.plot(extrapolate_x, poly(extrapolate_x), linestyle='--', alpha=0.5, label=f"{script} (Extrap)")
        
        ax.set_title(mean_col.replace(' Mean', ''))
        ax.set_xlabel(f'Number of {x_col}')
        ax.set_ylabel(mean_col)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Plot saved to {out_png}")

def run_suite(target_df, subset_rows, num_features, num_classes, sigma, scripts, runs):
    """Helper to run the compilation and execution pipeline for a specific configuration"""
    
    # 1. Subset Data
    cols = [c for c in target_df.columns if c != target_df.columns[-1]][:num_features] + [target_df.columns[-1]]
    work_df = target_df[cols].iloc[:subset_rows]
    pat_0, pat_1 = write_shares(work_df, num_features)
    
    c_args = [str(pat_0), str(pat_1), str(num_features), str(num_classes), sigma]
    res_list = []
    
    for script in scripts:
        print(f"  > Running {script} ...")
        comp_times, comp_mems, exec_times, exec_mems = [], [], [], []
        
        for run in range(runs):
            # Compile Phase
            compile_cmd = ["./compile.py", "-R", "64", script] + c_args
            ct, cm = get_time_and_memory(compile_cmd, cwd=MPC_DIR)
            comp_times.append(ct)
            comp_mems.append(cm)
            
            # Execution Phase
            compiled_name = f"{script}-{'-'.join(c_args)}"
            exec_cmd = ["Scripts/ring.sh", compiled_name]
            
            et, em = get_time_and_memory(exec_cmd, cwd=MPC_DIR)
            exec_times.append(et)
            exec_mems.append(em)

        res_list.append({
            "Features": num_features, "Samples": subset_rows, "Script": script,
            "Compile Time Mean (s)": np.mean(comp_times), "Compile Time Std (s)": np.std(comp_times),
            "Compile RAM Mean (MB)": np.mean(comp_mems), "Compile RAM Std (MB)": np.std(comp_mems),
            "Exec Time Mean (s)": np.mean(exec_times), "Exec Time Std (s)": np.std(exec_times),
            "Exec RAM Mean (MB)": np.mean(exec_mems), "Exec RAM Std (MB)": np.std(exec_mems)
        })
        print(f"    Avg Exec: {np.mean(exec_times):.2f}s | Avg Compile: {np.mean(comp_times):.2f}s")
        
    return res_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to input CSV")
    parser.add_argument('--label_col', type=str, required=True, help="Name of the label column")
    parser.add_argument('--runs', type=int, default=3, help="Number of times to run each benchmark")
    
    # Feature Experiment Params
    parser.add_argument('--features', type=int, nargs='+', default=[20, 30, 40], help="Feature counts to benchmark")
    parser.add_argument('--fixed_samples', type=int, default=None, help="Fixed sample size when varying features (default: all)")
    
    # Sample Experiment Params
    parser.add_argument('--samples', type=int, nargs='+', default=[100, 500, 1000], help="Sample counts to benchmark")
    parser.add_argument('--fixed_features', type=int, default=50, help="Fixed feature size when varying samples")
    
    args = parser.parse_args()

    # Make label the last column for easier slicing
    df = pd.read_csv(args.csv).dropna()
    cols = [c for c in df.columns if c != args.label_col] + [args.label_col]
    df = df[cols]
    
    num_classes = df[args.label_col].nunique()
    sigma = "0.5"
    
    scripts = [
        "orig_bin", "opt_bin", 
        "orig_marg", "opt_marg", 
        "ppai_original_full",   
        "ppai_bin_wo_dp_msr_opt"
    ]
    
    print(f"Dataset Loaded. Classes: {num_classes}. Total Rows Available: {len(df)}")
    max_s = len(df) if args.fixed_samples is None else min(len(df), args.fixed_samples)

    # -------------------------------------------------------------
    # EXPERIMENT 1: VARYING FEATURES (Fixed Samples)
    # -------------------------------------------------------------
    print(f"\n{'='*50}\nEXPERIMENT 1: VARYING FEATURES (Fixed Samples: {max_s})\n{'='*50}")
    res_feats = []
    for f in args.features:
        print(f"\n--- Benchmarking {f} Features ---")
        results = run_suite(df, max_s, f, num_classes, sigma, scripts, args.runs)
        res_feats.extend(results)

    df_feats = pd.DataFrame(res_feats)
    df_feats.to_csv("scaling_results/benchmark_features.csv", index=False)
    plot_results("scaling_results/benchmark_features.csv", x_col="Features", extrap_limit=100, out_png="scaling_results/benchmark_features.png")


    # -------------------------------------------------------------
    # EXPERIMENT 2: VARYING SAMPLES (Fixed Features)
    # -------------------------------------------------------------
    print(f"\n{'='*50}\nEXPERIMENT 2: VARYING SAMPLES (Fixed Features: {args.fixed_features})\n{'='*50}")
    res_samples = []
    for s in args.samples:
        if s > len(df):
            print(f"Skipping {s} samples (Dataset only has {len(df)})")
            continue
            
        print(f"\n--- Benchmarking {s} Samples ---")
        results = run_suite(df, s, args.fixed_features, num_classes, sigma, scripts, args.runs)
        res_samples.extend(results)

    df_samples = pd.DataFrame(res_samples)
    df_samples.to_csv("scaling_results/benchmark_samples.csv", index=False)
    
    # Extrapolate samples to roughly double the max tested, or at least 2000
    extrap_limit_samples = max(max(args.samples) * 1.5, 2000)
    plot_results("scaling_results/benchmark_samples.csv", x_col="Samples", extrap_limit=extrap_limit_samples, out_png="scaling_results/benchmark_samples.png")
    
    print("\nAll Benchmarks Complete! Saved to CSVs and PNGs.")

if __name__ == "__main__":
    main()
