import os
import time
import argparse
import subprocess
import numpy as np

# ==========================================
# 1. INPUT GENERATORS
# ==========================================
def generate_raw_inputs(train_data, train_labels, party_id):
    """
    Format for ppai_bin_msr.mpc (Raw Floating Point Data)
    Expects flattened array of shape (N, Genes + 1)
    """
    # Append labels as the last column
    labels_col = train_labels.reshape(-1, 1)
    combined_data = np.hstack((train_data, labels_col))

    flat_data = combined_data.flatten()

    os.makedirs("mpc_spdz/Player-Data", exist_ok=True)
    out_file = f"mpc_spdz/Player-Data/Input-P{party_id}-0"

    # Save as space-separated floats for sfix.input_from()
    np.savetxt(out_file, flat_data, fmt='%f', newline=' ')
    print(f"      [Party {party_id}] Saved {len(flat_data)} raw floats.")

def generate_histogram_inputs(train_data, train_labels, party_id, num_genes, num_classes, B=100):
    """
    Format for histogram_marginals.mpc (Federated Log-Space Histograms)
    Expects flattened integer array of shape (Genes, Buckets, Classes)
    """
    log_data = np.log2(train_data + 1.0)
    bins = np.linspace(0.0, 22.0, B + 1)

    local_histogram = np.zeros((num_genes, B, num_classes), dtype=int)

    for j in range(num_genes):
        gene_col = log_data[:, j]
        b_indices = np.digitize(gene_col, bins) - 1
        b_indices = np.clip(b_indices, 0, B - 1)

        for i in range(len(gene_col)):
            c = int(train_labels[i])
            b = b_indices[i]
            local_histogram[j, b, c] += 1

    flat_hist = local_histogram.flatten()

    os.makedirs("mpc_spdz/Player-Data", exist_ok=True)
    out_file = f"mpc_spdz/Player-Data/Input-P{party_id}-0"

    # Save as space-separated integers for sint.input_from()
    np.savetxt(out_file, flat_hist, fmt='%d', newline=' ')
    print(f"      [Party {party_id}] Saved {len(flat_hist)} histogram ints.")

# ==========================================
# 2. MPC RUNNER
# ==========================================
def compile_mpc(mpc_script, args_list):
    # MP-SPDZ appends the arguments to the compiled filename. We need to construct this name!
    compiled_name = f"{mpc_script}-{'-'.join(str(a) for a in args_list)}"
    
    print(f"    -> Compiling {mpc_script}.mpc with args {args_list}...")
    start_time = time.time()
    
    # We call ./compile.py directly because we set cwd="mpc_spdz" in subprocess
    compile_cmd = ["./compile.py", "-R", "64", mpc_script] + [str(a) for a in args_list]

    try:
        # cwd="mpc_spdz" ensures it runs as if we "cd mpc_spdz" first
        subprocess.run(compile_cmd, cwd="mpc_spdz", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return time.time() - start_time, compiled_name
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Compilation failed:\n{e.stderr.decode()}")
        exit(1)

def execute_mpc(compiled_name):
    print(f"    -> Executing protocol: {compiled_name}...")
    start_time = time.time()
    
    # Pass the fully formatted name (e.g. ppai_bin_msr-951-950...) to ring.sh
    run_cmd = ["./Scripts/ring.sh", compiled_name]

    try:
        # cwd="mpc_spdz" ensures it runs as if we "cd mpc_spdz" first
        result = subprocess.run(run_cmd, cwd="mpc_spdz", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        return time.time() - start_time
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Execution failed:\n{e.stdout.decode()}")
        exit(1)

# ==========================================
# 3. BENCHMARK SUITE
# ==========================================
def run_benchmark():
    # Setup Protocol Scripts
    scripts = {
        "msr": "ppai_bin_msr", 
        "opt": "histogram_marginals"  
    }

    # Benchmark Parameters
    gene_counts = [100, 500, 1000]
    num_patients_0 = 951
    num_patients_1 = 950
    num_classes = 5
    sigma = 1000

    results = { "msr": {}, "opt": {} }

    print("=====================================================")
    print(" MPC ALGORITHM BENCHMARK SUITE")
    print("=====================================================")

    for genes in gene_counts:
        for mode in ["opt"]: # msr
            print(f"  --- Testing with {genes} Genes ---")

            # 1. Generate Fake Skewed Genomic Data
            data_0 = np.random.exponential(scale=100000.0, size=(num_patients_0, genes))
            labels_0 = np.random.randint(0, num_classes, size=(num_patients_0,))

            data_1 = np.random.exponential(scale=100000.0, size=(num_patients_1, genes))
            labels_1 = np.random.randint(0, num_classes, size=(num_patients_1,))

            # 2. Format inputs based on protocol requirement
            if mode == "msr":
                generate_raw_inputs(data_0, labels_0, 0)
                generate_raw_inputs(data_1, labels_1, 1)
            else:
                generate_histogram_inputs(data_0, labels_0, 0, genes, num_classes)
                generate_histogram_inputs(data_1, labels_1, 1, genes, num_classes)

            # 3. Compile
            args_list = [num_patients_0, num_patients_1, genes, num_classes, sigma]
            comp_time, compiled_name = compile_mpc(scripts[mode], args_list)

            # 4. Execute
            exec_time = execute_mpc(compiled_name)

            print(f"      Compile Time: {comp_time:.2f} s")
            print(f"      Execute Time: {exec_time:.2f} s")

            results[mode][genes] = {"comp": comp_time, "exec": exec_time}

    # ==========================================
    # 4. PRINT SUMMARY TABLE
    # ==========================================
    print("\n\n=========================================================================")
    print(f"{'Genes':<10} | {'MSR (Raw Data + Sort)':<30} | {'OPT (Federated Histograms)':<30}")
    print(f"{'':<10} | {'Compile (s)':<14} {'Execute (s)':<15} | {'Compile (s)':<14} {'Execute (s)':<15}")
    print("=========================================================================")
    for genes in gene_counts:
        r_msr = results["msr"].get(genes, {"comp": 0.0, "exec": 0.0})
        r_opt = results["opt"].get(genes, {"comp": 0.0, "exec": 0.0})
        print(f"{genes:<10} | {r_msr['comp']:<14.2f} {r_msr['exec']:<15.2f} | {r_opt['comp']:<14.2f} {r_opt['exec']:<15.2f}")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_benchmark()
