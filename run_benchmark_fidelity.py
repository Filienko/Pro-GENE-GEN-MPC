import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.stats import kendalltau

from run_secure_mpc_pipeline import run_secure_mpc_pipeline
import utils.mpc_helper as mpc_helper

# ==========================================
# FIDELITY EVALUATION HELPERS
# ==========================================

def compute_1d_marginals_and_zero_rates(real_df, synth_df):
    """Computes MARE (Mean Average Relative Error) for means, variances, and zero-inflation"""
    # Zero rates
    z_real = (real_df == 0).mean()
    z_synth = (synth_df == 0).mean()
    
    # Non-zero means
    mean_real = real_df.replace(0, np.nan).mean().fillna(0)
    mean_synth = synth_df.replace(0, np.nan).mean().fillna(0)
    
    def mare(a, b):
        denom = np.maximum(a, b)
        denom[denom == 0] = 1e-9 # avoid div by zero
        return (np.abs(a - b) / denom).mean()

    return float(mare(z_real, z_synth)), float(mare(mean_real, mean_synth))

def compute_correlation_diff(real_df, synth_df):
    """Mean Absolute Error between the feature correlation matrices"""
    corr_real = real_df.corr().fillna(0).values
    corr_synth = synth_df.corr().fillna(0).values
    return float(np.mean(np.abs(corr_real - corr_synth)))

def compute_feature_importance_agreement(X_real, y_real, X_synth, y_synth):
    """Kendall-Tau agreement of Random Forest feature importances"""
    clf_real = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_real, y_real)
    clf_synth = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_synth, y_synth)
    
    tau, _ = kendalltau(clf_real.feature_importances_, clf_synth.feature_importances_)
    return float(tau) if not np.isnan(tau) else 0.0

def compute_cluster_preservation(X_real, y_real, X_synth, y_synth):
    """Adjusted Rand Index (ARI) difference for K-Means clustering (patient overlap)"""
    n_clusters = len(np.unique(y_real))
    
    # How well do natural clusters map to the labels in real data?
    km_real = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_real)
    ari_real = adjusted_rand_score(y_real, km_real.labels_)
    
    # How well do natural clusters map to the labels in synthetic data?
    km_synth = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_synth)
    ari_synth = adjusted_rand_score(y_synth, km_synth.labels_)
    
    return float(ari_real), float(ari_synth)

# ==========================================
# BASELINE: TRAIN ON REAL DATA
# ==========================================

def compute_baseline_accuracy(train_df, test_df):
    """
    Upper-bound baseline: train a Random Forest on the full (concatenated) real
    training data from both parties and evaluate on the held-out real test set.
    This represents the best accuracy we could expect without any privacy constraints.
    """
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test  = test_df.drop(columns=['label'])
    y_test  = test_df['label']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return float(acc)

# ==========================================
# MAIN BENCHMARK PIPELINE
# ==========================================

def run_benchmark(full_data_path, label_column, mpspdz_path, feature_sizes=[800,1000], n_runs=3):
    print(f"Loading full dataset from {full_data_path}...")
    df = pd.read_csv(full_data_path)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    # Drop non-numeric ID columns
    for col in df.select_dtypes(include=['object']).columns:
        if col != label_column:
            df = df.drop(columns=[col])

    # Encode string labels if necessary
    if df[label_column].dtype == 'object':
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])

    y = df[label_column]
    X = df.drop(columns=[label_column])

    results = []

    for n_features in feature_sizes:
        print(f"\n{'='*80}")
        print(f" RUNNING BENCHMARK FOR {n_features} FEATURES (Averaging over {n_runs} runs)")
        print(f"{'='*80}")

        n_features_to_use = min(n_features, X.shape[1])
        X_sub = X.iloc[:, :n_features_to_use].copy()

        df_sub = pd.concat([X_sub, y], axis=1)
        df_sub.rename(columns={label_column: 'label'}, inplace=True)

        # Split data ONCE per feature size so all 3 runs use identical underlying data
        train_df, test_df = train_test_split(df_sub, test_size=0.2, random_state=42, stratify=df_sub['label'])
        party1_df, party2_df = train_test_split(train_df, test_size=0.5, random_state=42, stratify=train_df['label'])

        # ----------------------------------------------------------
        # BASELINE: concatenate both parties' real training data and
        # evaluate on the shared real test set (no MPC / DP involved).
        # This gives an upper-bound reference for downstream accuracy.
        # ----------------------------------------------------------
        print(f"\n--- [Features: {n_features_to_use}] Computing real-data baseline ---")
        baseline_acc = compute_baseline_accuracy(train_df, test_df)
        print(f"    Baseline accuracy (train on real, test on real): {baseline_acc:.4f}")

        party1_path, party2_path = f"tmp_p1_{n_features_to_use}.csv", f"tmp_p2_{n_features_to_use}.csv"
        synth_out_path = f"tmp_synthetic_{n_features_to_use}.csv"

        party1_df.to_csv(party1_path, index=False)
        party2_df.to_csv(party2_path, index=False)

        run_metrics_list = []

        for run_id in range(n_runs):
            print(f"\n--- [Features: {n_features_to_use}] Executing DP-MPC Run {run_id + 1}/{n_runs} ---")
            
            # Ensure network tracking metric exists and reset ALL metrics
            if 'data_sent_mb' not in mpc_helper.MPC_METRICS:
                mpc_helper.MPC_METRICS['data_sent_mb'] = 0.0
            for key in mpc_helper.MPC_METRICS:
                mpc_helper.MPC_METRICS[key] = 0 if isinstance(mpc_helper.MPC_METRICS[key], int) else 0.0

            try:
                synth_df = run_secure_mpc_pipeline(
                    party_files=[party1_path, party2_path],
                    output_path=synth_out_path,
                    epsilon=10.0,
                    delta=1e-5,
                    marginal_protocol='ppai_bin_dp_msr',
                    mpspdz_path=mpspdz_path,
                )
            except Exception as e:
                print(f"Pipeline failed for {n_features_to_use} features on Run {run_id + 1}: {e}")
                continue

            print(f"--- Evaluating Downstream Utility & Fidelity (Run {run_id + 1}) ---")
            X_train_real = train_df.drop(columns=['label'])
            y_train_real = train_df['label']

            X_train_synth = synth_df.drop(columns=['label'])
            y_train_synth = synth_df['label'].round().astype(int) 

            X_test = test_df.drop(columns=['label'])
            y_test = test_df['label']

            # 1. Downstream Accuracy (Train on Synth, Test on Real)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_synth, y_train_synth)
            acc = accuracy_score(y_test, clf.predict(X_test))

            # 2. Fidelity Metrics (Real vs Synth representations)
            mare_zero, mare_mean = compute_1d_marginals_and_zero_rates(X_train_real, X_train_synth)
            corr_diff = compute_correlation_diff(X_train_real, X_train_synth)
            feat_kendall_tau = compute_feature_importance_agreement(X_train_real, y_train_real, X_train_synth, y_train_synth)
            ari_real, ari_synth = compute_cluster_preservation(X_train_real, y_train_real, X_train_synth, y_train_synth)

            # Store run results
            metrics = mpc_helper.MPC_METRICS.copy()
            metrics['accuracy'] = acc
            metrics['mare_zero_rate'] = mare_zero
            metrics['mare_nz_mean'] = mare_mean
            metrics['corr_diff_mae'] = corr_diff
            metrics['feat_rank_tau'] = feat_kendall_tau
            metrics['ari_real'] = ari_real
            metrics['ari_synth'] = ari_synth

            run_metrics_list.append(metrics)

        # Aggregate the runs for this feature size
        if not run_metrics_list:
            continue

        print(f"\n--- Averaging metrics across {len(run_metrics_list)} successful runs ---")
        avg_metrics = {'num_features': n_features_to_use, 'baseline_accuracy': round(baseline_acc, 4)}
        keys_to_avg = ['accuracy', 'feat_rank_tau', 'mare_zero_rate', 'mare_nz_mean', 
                       'corr_diff_mae', 'ari_real', 'ari_synth', 
                       'compile_time', 'execute_time', 'data_sent_mb',
                       'integer_bits', 'integer_opens', 'integer_triples', 'vm_rounds']
        
        for k in keys_to_avg:
            vals = [rm[k] for rm in run_metrics_list]
            if isinstance(vals[0], float):
                avg_metrics[k] = round(sum(vals) / len(vals), 4)
            else:
                avg_metrics[k] = int(sum(vals) / len(vals))

        # Format MPC times explicitly
        avg_metrics['compile_time'] = round(avg_metrics['compile_time'], 2)
        avg_metrics['execute_time'] = round(avg_metrics['execute_time'], 2)
        avg_metrics['data_sent_mb'] = round(avg_metrics['data_sent_mb'], 2)

        results.append(avg_metrics)

        # Cleanup
        for p in [party1_path, party2_path, synth_out_path]:
            if os.path.exists(p): os.remove(p)

    print("\n" + "="*80)
    print(f" FINAL BENCHMARK RESULTS (Averaged over {n_runs} runs)")
    print("="*80)

    # Reorder columns — baseline_accuracy sits right next to accuracy for easy comparison
    cols = ['num_features', 'baseline_accuracy', 'accuracy', 'feat_rank_tau',
            'mare_zero_rate', 'mare_nz_mean', 'corr_diff_mae', 'ari_real', 'ari_synth', 
            'compile_time', 'execute_time', 'data_sent_mb',
            'integer_bits', 'integer_opens', 'integer_triples', 'vm_rounds']
    
    results_df = pd.DataFrame(results)[cols]
    print(results_df.to_string(index=False))
    results_df.to_csv("benchmark_mpc_fidelity_results.csv", index=False)
    print("\n✓ Full averaged results saved to: benchmark_mpc_fidelity_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper to benchmark Secure MPC Pipeline + Fidelity (Averaged over N runs)")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--mpspdz', type=str, required=True)
    parser.add_argument('--runs', type=int, default=3, help='Number of DP estimation runs to average')
    args = parser.parse_args()

    run_benchmark(args.data, args.label, args.mpspdz, n_runs=args.runs)
