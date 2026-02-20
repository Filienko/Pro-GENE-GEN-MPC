import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from run_secure_mpc_pipeline import run_secure_mpc_pipeline
import utils.mpc_helper as mpc_helper

def run_benchmark(full_data_path, label_column, mpspdz_path, feature_sizes=[5, 10]):
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
        print(f"\n{'='*60}")
        print(f" RUNNING BENCHMARK FOR {n_features} FEATURES")
        print(f"{'='*60}")
        
        # Reset ALL metrics for this iteration
        for key in mpc_helper.MPC_METRICS:
            mpc_helper.MPC_METRICS[key] = 0 if isinstance(mpc_helper.MPC_METRICS[key], int) else 0.0

        n_features_to_use = min(n_features, X.shape[1])
        X_sub = X.iloc[:, :n_features_to_use].copy()
        
        df_sub = pd.concat([X_sub, y], axis=1)
        df_sub.rename(columns={label_column: 'label'}, inplace=True)

        train_df, test_df = train_test_split(df_sub, test_size=0.2, random_state=42, stratify=df_sub['label'])
        party1_df, party2_df = train_test_split(train_df, test_size=0.5, random_state=42, stratify=train_df['label'])

        party1_path, party2_path = f"tmp_p1_{n_features_to_use}.csv", f"tmp_p2_{n_features_to_use}.csv"
        synth_out_path = f"tmp_synthetic_{n_features_to_use}.csv"

        party1_df.to_csv(party1_path, index=False)
        party2_df.to_csv(party2_path, index=False)

        print(f"\n--- Starting MPC Pipeline ---")
        try:
            synth_df = run_secure_mpc_pipeline(
                party_files=[party1_path, party2_path],
                output_path=synth_out_path,
                epsilon=10.0,
                delta=1e-5,
                mpspdz_path=mpspdz_path
            )
        except Exception as e:
            print(f"Pipeline failed for {n_features_to_use} features: {e}")
            continue

        print(f"\n--- Evaluating Downstream Utility ---")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(synth_df.drop(columns=['label']), synth_df['label'])
        acc = accuracy_score(test_df['label'], clf.predict(test_df.drop(columns=['label'])))

        # Append everything to results
        metrics = mpc_helper.MPC_METRICS.copy()
        metrics['num_features'] = n_features_to_use
        metrics['accuracy'] = round(acc, 4)
        metrics['compile_time'] = round(metrics['compile_time'], 2)
        metrics['execute_time'] = round(metrics['execute_time'], 2)
        
        results.append(metrics)

        # Cleanup
        for p in [party1_path, party2_path, synth_out_path]:
            if os.path.exists(p): os.remove(p)

    print("\n" + "="*60)
    print(" FINAL BENCHMARK RESULTS")
    print("="*60)
    
    # Reorder columns to make it look nice
    cols = ['num_features', 'accuracy', 'compile_time', 'execute_time', 
            'integer_bits', 'integer_opens', 'integer_triples', 'vm_rounds']
    results_df = pd.DataFrame(results)[cols]
    
    print(results_df.to_string(index=False))
    results_df.to_csv("benchmark_mpc_scaling_results.csv", index=False)
    print("\n✓ Full results saved to: benchmark_mpc_scaling_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper to benchmark Secure MPC Pipeline")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--mpspdz', type=str, required=True)
    args = parser.parse_args()
    
    run_benchmark(args.data, args.label, args.mpspdz)

