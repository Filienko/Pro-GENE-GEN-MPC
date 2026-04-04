import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kendalltau, wasserstein_distance, ranksums

from run_secure_mpc_pipeline import run_secure_mpc_pipeline
import utils.mpc_helper as mpc_helper

# ==========================================
# FIDELITY EVALUATION HELPERS
# ==========================================

def compute_histogram_intersection(real_df, synth_df, bins=20):
    scores = []
    for col in real_df.columns:
        min_val = min(real_df[col].min(), synth_df[col].min())
        max_val = max(real_df[col].max(), synth_df[col].max())

        if min_val == max_val:
            scores.append(1.0 if real_df[col].mean() == synth_df[col].mean() else 0.0)
            continue

        real_hist, _ = np.histogram(real_df[col], bins=bins, range=(min_val, max_val), density=True)
        synth_hist, _ = np.histogram(synth_df[col], bins=bins, range=(min_val, max_val), density=True)

        bin_width = (max_val - min_val) / bins
        real_prob = real_hist * bin_width
        synth_prob = synth_hist * bin_width

        intersection = np.sum(np.minimum(real_prob, synth_prob))
        scores.append(intersection)
    return float(np.mean(scores))

def compute_dcr(real_df, synth_df, k=1):
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real_df)
    distances, _ = nn.kneighbors(synth_df)
    return float(np.mean(distances))

def compute_differential_expression_tpr(X_real, y_real, X_synth, y_synth, p_val_thresh=0.05):
    classes = np.unique(y_real)
    if len(classes) < 2: return 0.0

    tpr_list = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1, c2 = classes[i], classes[j]

            real_c1 = X_real[y_real == c1]
            real_c2 = X_real[y_real == c2]
            synth_c1 = X_synth[y_synth == c1]
            synth_c2 = X_synth[y_synth == c2]

            if len(real_c1) == 0 or len(real_c2) == 0 or len(synth_c1) == 0 or len(synth_c2) == 0:
                continue

            real_degs_up, real_degs_down = set(), set()
            for col in X_real.columns:
                stat, p = ranksums(real_c1[col], real_c2[col])
                if p < p_val_thresh:
                    if stat > 0: real_degs_up.add(col)
                    else: real_degs_down.add(col)

            if not real_degs_up and not real_degs_down:
                continue

            synth_degs_up, synth_degs_down = set(), set()
            for col in X_synth.columns:
                stat, p = ranksums(synth_c1[col], synth_c2[col])
                if p < p_val_thresh:
                    if stat > 0: synth_degs_up.add(col)
                    else: synth_degs_down.add(col)

            if real_degs_up:
                tpr_up = len(real_degs_up.intersection(synth_degs_up)) / len(real_degs_up)
                tpr_list.append(tpr_up)
            if real_degs_down:
                tpr_down = len(real_degs_down.intersection(synth_degs_down)) / len(real_degs_down)
                tpr_list.append(tpr_down)

    return float(np.mean(tpr_list)) if tpr_list else 0.0

def compute_coexpression_tpr(X_real, X_synth, threshold=0.7):
    corr_real = X_real.corr(method='pearson').fillna(0).values
    corr_synth = X_synth.corr(method='pearson').fillna(0).values

    triu_idx = np.triu_indices_from(corr_real, k=1)

    edges_real = (np.abs(corr_real[triu_idx]) > threshold)
    edges_synth = (np.abs(corr_synth[triu_idx]) > threshold)

    real_edge_count = np.sum(edges_real)
    if real_edge_count == 0:
        return 0.0

    true_positives = np.sum(edges_real & edges_synth)
    return float(true_positives / real_edge_count)

def compute_1d_marginals_and_zero_rates(real_df, synth_df):
    z_real = (real_df == 0).mean()
    z_synth = (synth_df == 0).mean()
    mean_real = real_df.replace(0, np.nan).mean().fillna(0)
    mean_synth = synth_df.replace(0, np.nan).mean().fillna(0)
    def mare(a, b):
        denom = np.maximum(a, b)
        denom[denom == 0] = 1e-9
        return (np.abs(a - b) / denom).mean()
    return float(mare(z_real, z_synth)), float(mare(mean_real, mean_synth))

def compute_wasserstein(real_df, synth_df):
    distances = []
    for col in real_df.columns:
        distances.append(wasserstein_distance(real_df[col], synth_df[col]))
    return float(np.mean(distances))

def compute_correlation_diff(real_df, synth_df):
    corr_real = real_df.corr().fillna(0).values
    corr_synth = synth_df.corr().fillna(0).values
    return float(np.mean(np.abs(corr_real - corr_synth)))

def compute_lr_feature_importance_agreement(X_real, y_real, X_synth, y_synth, top_k=50):
    # SAFETY CHECK: Prevent crash if synthetic data collapsed to 1 class
    if len(np.unique(y_real)) < 2 or len(np.unique(y_synth)) < 2:
        return np.nan, np.nan

    clf_real = LogisticRegression(max_iter=2000, random_state=42).fit(X_real, y_real)
    clf_synth = LogisticRegression(max_iter=2000, random_state=42).fit(X_synth, y_synth)

    imp_real = np.mean(np.abs(clf_real.coef_), axis=0) if len(clf_real.classes_) > 2 else np.abs(clf_real.coef_[0])
    imp_synth = np.mean(np.abs(clf_synth.coef_), axis=0) if len(clf_synth.classes_) > 2 else np.abs(clf_synth.coef_[0])

    tau, _ = kendalltau(imp_real, imp_synth)
    tau = float(tau) if not np.isnan(tau) else 0.0

    k_actual = min(top_k, len(imp_real))
    top_real_idx = set(np.argsort(imp_real)[-k_actual:])
    top_synth_idx = set(np.argsort(imp_synth)[-k_actual:])
    overlap_score = len(top_real_idx.intersection(top_synth_idx)) / k_actual

    return float(tau), float(overlap_score)

def compute_cluster_preservation(X_real, y_real, X_synth, y_synth):
    n_clusters = len(np.unique(y_real))
    km_real = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_real)
    ari_real = adjusted_rand_score(y_real, km_real.labels_)
    km_synth = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_synth)
    ari_synth = adjusted_rand_score(y_synth, km_synth.labels_)
    return float(ari_real), float(ari_synth)

def compute_baseline_metrics(train_df, test_df):
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test  = test_df.drop(columns=['label'])
    y_test  = test_df['label']

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return float(acc), float(f1)

# ==========================================
# MAIN BENCHMARK PIPELINE
# ==========================================

def run_benchmark(full_data_path, label_column, mpspdz_path, protocols, feature_sizes, n_runs=1, prefix="", port=None):
    print(f"Loading full dataset from {full_data_path}...")
    df = pd.read_csv(full_data_path)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    from sklearn.preprocessing import LabelEncoder
    for col in df.select_dtypes(include=['object']).columns:
        if col != label_column:
            df = df.drop(columns=[col])

    if df[label_column].dtype == 'object':
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])

    y = df[label_column]
    X = df.drop(columns=[label_column])

    # Create directory for saving generated synthetic datasets
    os.makedirs("saved_synthetic_data", exist_ok=True)

    results = []
    raw_log_file = f"{prefix}benchmark_fidelity_RAW_log.csv"

    for n_features in feature_sizes:
        for current_protocol in protocols:
            print(f"\n{'='*80}")
            print(f" BENCHMARK: {n_features} FEATURES | PROTOCOL: {current_protocol} | RUNS: {n_runs}")
            print(f"{'='*80}")

            n_features_to_use = min(n_features, X.shape[1])
            chosen_cols = X.sample(n=n_features_to_use, axis=1, random_state=42).columns
            X_sub = X[chosen_cols].copy()

            df_sub = pd.concat([X_sub, y], axis=1)
            df_sub.rename(columns={label_column: 'label'}, inplace=True)

            run_metrics_list = []

            for run_id in range(n_runs):
                print(f"\n--- [Features: {n_features_to_use} | Protocol: {current_protocol}] Run {run_id + 1}/{n_runs} ---")

                # Deterministic split per run_id ensures apples-to-apples comparison
                current_seed = 42 + run_id

                train_df, test_df = train_test_split(
                    df_sub, test_size=0.2, random_state=current_seed, stratify=df_sub['label']
                )
                party1_df, party2_df = train_test_split(
                    train_df, test_size=0.5, random_state=current_seed, stratify=train_df['label']
                )

                print(f"--- Computing real-data baseline for this split ---")
                base_acc, base_f1 = compute_baseline_metrics(train_df, test_df)

                party1_path = f"{prefix}tmp_p1_feat{n_features_to_use}_{current_protocol}_run{run_id}_opt.csv"
                party2_path = f"{prefix}tmp_p2_feat{n_features_to_use}_{current_protocol}_run{run_id}_opt.csv"
                synth_out_path = f"{prefix}tmp_synthetic_feat{n_features_to_use}_{current_protocol}_run{run_id}_opt.csv"

                mpc_helper.MPC_METRICS = {
                    'binning_time': 0.0, 'marginal_time': 0.0,
                    'noise_time': 0.0, 'reveal_time': 0.0,
                    'binning_mb': 0.0, 'marginal_mb': 0.0,
                    'noise_mb': 0.0, 'reveal_mb': 0.0,
                    'binning_rounds': 0, 'marginal_rounds': 0,
                    'noise_rounds': 0, 'reveal_rounds': 0,
                    'input_prep_time': 0.0, 'compile_time': 0.0,
                    'execute_time': 0.0, 'parse_time': 0.0,
                    'generation_time': 0.0,
                    'data_sent_mb': 0.0,
                    'integer_bits': 0, 'integer_opens': 0, 'integer_triples': 0, 'vm_rounds': 0
                }

                try:
                    party1_df.to_csv(party1_path, index=False)
                    party2_df.to_csv(party2_path, index=False)

                    synth_df = run_secure_mpc_pipeline(
                        party_files=[party1_path, party2_path],
                        output_path=synth_out_path,
                        epsilon=10.0,
                        delta=1e-5,
                        marginal_protocol='ppai_bin_wo_dp_msr_opt',
                        mpspdz_path=mpspdz_path,
                        mpc_protocol=current_protocol,
                        port=port
                    )

                    # ------------------------------------------------------------
                    # NEW: Save a permanent copy of the synthetic data for this run
                    # ------------------------------------------------------------
                    perm_synth_path = f"saved_synthetic_data/{prefix}synth_feat{n_features_to_use}_{current_protocol}_run{run_id}.csv"
                    synth_df.to_csv(perm_synth_path, index=False)
                    print(f"  -> Successfully saved synthetic dataset to: {perm_synth_path}")

                    print(f"--- Evaluating Downstream Utility & Fidelity ---")
                    X_train_real = train_df.drop(columns=['label'])
                    y_train_real = train_df['label']
                    X_train_synth = synth_df.drop(columns=['label'])
                    y_train_synth = synth_df['label'].round().astype(int)
                    X_test = test_df.drop(columns=['label'])
                    y_test = test_df['label']

                    # ------------------------------------------------------------
                    # SAFETY CHECK: Downstream ML Metrics (Logistic Regression, No AUC)
                    # ------------------------------------------------------------
                    if len(np.unique(y_train_synth)) < 2:
                        print(f"  -> [Warning] PGM collapsed labels to 1 class. Skipping ML eval to prevent crash.")
                        acc = np.nan
                        f1 = np.nan
                    else:
                        clf = LogisticRegression(max_iter=2000, random_state=42)
                        clf.fit(X_train_synth, y_train_synth)
                        y_pred = clf.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')

                    # Fidelity Metrics
                    hist_intersect = compute_histogram_intersection(X_train_real, X_train_synth)
                    dcr_knn = compute_dcr(X_train_real, X_train_synth, k=1)
                    de_tpr = compute_differential_expression_tpr(X_train_real, y_train_real, X_train_synth, y_train_synth)
                    coex_tpr = compute_coexpression_tpr(X_train_real, X_train_synth, threshold=0.7)

                    mare_zero, mare_mean = compute_1d_marginals_and_zero_rates(X_train_real, X_train_synth)
                    wd_dist = compute_wasserstein(X_train_real, X_train_synth)
                    corr_diff = compute_correlation_diff(X_train_real, X_train_synth)
                    lr_rank_tau, lr_topk_overlap = compute_lr_feature_importance_agreement(X_train_real, y_train_real, X_train_synth, y_train_synth, top_k=50)
                    ari_real, ari_synth = compute_cluster_preservation(X_train_real, y_train_real, X_train_synth, y_train_synth)

                    metrics = mpc_helper.MPC_METRICS.copy()
                    metrics['run_id'] = run_id
                    metrics['protocol'] = current_protocol
                    metrics['num_features'] = n_features_to_use

                    # Baselines
                    metrics['base_accuracy'] = base_acc
                    metrics['base_f1'] = base_f1

                    # Synthetic ML Eval
                    metrics['accuracy'] = acc
                    metrics['f1_score'] = f1

                    # Statistical & Biological Fidelity
                    metrics['hist_intersection'] = hist_intersect
                    metrics['dcr_knn'] = dcr_knn
                    metrics['de_tpr'] = de_tpr
                    metrics['coex_tpr'] = coex_tpr
                    metrics['wasserstein_dist'] = wd_dist
                    metrics['mare_zero_rate'] = mare_zero
                    metrics['mare_nz_mean'] = mare_mean
                    metrics['corr_diff_mae'] = corr_diff
                    metrics['feat_rank_tau'] = lr_rank_tau
                    metrics['feat_topk_overlap'] = lr_topk_overlap
                    metrics['ari_real'] = ari_real
                    metrics['ari_synth'] = ari_synth

                    run_metrics_list.append(metrics)

                    # INCREMENTAL SAVE
                    raw_df = pd.DataFrame([metrics])
                    if not os.path.isfile(raw_log_file):
                        raw_df.to_csv(raw_log_file, index=False)
                    else:
                        raw_df.to_csv(raw_log_file, mode='a', header=False, index=False)

                except Exception as e:
                    import traceback
                    print(f"Pipeline failed for {n_features_to_use} features on Run {run_id + 1}: {e}")
                    traceback.print_exc()
                    continue
                finally:
                    for p in [party1_path, party2_path, synth_out_path]:
                        if os.path.exists(p):
                            os.remove(p)

            if not run_metrics_list:
                continue

            print(f"\n--- Averaging metrics across {len(run_metrics_list)} successful runs for {current_protocol} ({n_features_to_use} features) ---")
            avg_metrics = {
                'protocol': current_protocol,
                'num_features': n_features_to_use
            }

            keys_to_avg = [
                'base_accuracy', 'base_f1',
                'accuracy', 'f1_score',
                'hist_intersection', 'dcr_knn', 'de_tpr', 'coex_tpr',
                'wasserstein_dist', 'feat_rank_tau', 'feat_topk_overlap', 'mare_zero_rate', 'mare_nz_mean',
                'corr_diff_mae', 'ari_real', 'ari_synth',
                # Stage-level times (from MP-SPDZ timer output)
                'binning_time', 'marginal_time', 'noise_time', 'reveal_time',
                # Stage-level comm (from TimerWithComm parentheses in MP-SPDZ stderr)
                'binning_mb', 'marginal_mb', 'noise_mb', 'reveal_mb',
                'binning_rounds', 'marginal_rounds', 'noise_rounds', 'reveal_rounds',
                # Pipeline-level times
                'input_prep_time', 'compile_time', 'execute_time', 'parse_time', 'generation_time',
                # Global communication totals
                'data_sent_mb', 'integer_bits', 'integer_opens', 'integer_triples', 'vm_rounds'
            ]

            for k in keys_to_avg:
                vals = [rm[k] for rm in run_metrics_list if pd.notnull(rm[k])]
                if not vals:
                    avg_metrics[k] = np.nan
                elif isinstance(vals[0], float):
                    avg_metrics[k] = round(sum(vals) / len(vals), 4)
                else:
                    avg_metrics[k] = int(sum(vals) / len(vals))

            for t_key in ['binning_time', 'marginal_time', 'noise_time', 'reveal_time',
                          'input_prep_time', 'compile_time', 'execute_time', 'parse_time', 'generation_time']:
                avg_metrics[t_key] = round(avg_metrics[t_key], 2)
            avg_metrics['data_sent_mb'] = round(avg_metrics['data_sent_mb'], 2)

            results.append(avg_metrics)

    print("\n" + "="*80)
    print(f" FINAL BENCHMARK RESULTS")
    print("="*80)

    if not results:
        print("All runs failed. No dataframe to display.")
        return

    cols = [
        'protocol', 'num_features', 'base_accuracy', 'accuracy', 'base_f1', 'f1_score',
        'hist_intersection', 'dcr_knn', 'de_tpr', 'coex_tpr',
        'wasserstein_dist', 'feat_rank_tau', 'feat_topk_overlap', 'mare_zero_rate', 'mare_nz_mean', 'corr_diff_mae', 'ari_real', 'ari_synth',
        # Stage-level times (inside MPC execution)
        'binning_time', 'marginal_time', 'noise_time', 'reveal_time',
        # Stage-level comm (MB and rounds per stage from MP-SPDZ TimerWithComm)
        'binning_mb', 'marginal_mb', 'noise_mb', 'reveal_mb',
        'binning_rounds', 'marginal_rounds', 'noise_rounds', 'reveal_rounds',
        # Pipeline-level times
        'input_prep_time', 'compile_time', 'execute_time', 'parse_time', 'generation_time',
        # Global communication totals
        'data_sent_mb', 'integer_bits', 'integer_opens', 'integer_triples', 'vm_rounds'
    ]

    results_df = pd.DataFrame(results)[cols]
    print(results_df.to_string(index=False))

    # Save a distinct Averaged CSV for each requested protocol
    for current_protocol in protocols:
        protocol_df = results_df[results_df['protocol'] == current_protocol]
        if not protocol_df.empty:
            out_filename = f"{prefix}benchmark_fidelity_avg_{current_protocol}_opt.csv"
            protocol_df.to_csv(out_filename, index=False)
            print(f"✓ Averaged results for {current_protocol} saved to: {out_filename}")

    print(f"✓ Raw un-averaged results with all intermediate runs safely preserved in: {raw_log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper to benchmark Secure MPC Pipeline + Biological Fidelity")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--mpspdz', type=str, required=True)
    parser.add_argument('--runs', type=int, default=3, help='Number of DP estimation runs to average')
    parser.add_argument('--protocols', nargs='+', default=['ring', 'mal-rep-ring'], help='List of MP-SPDZ protocols to test')
    parser.add_argument('--features', type=int, nargs='+', default=[200, 500, 800, 1000])
    parser.add_argument('--prefix', type=str, default='', help='Prefix for all output filenames (enables concurrent runs)')
    parser.add_argument('--port', type=int, default=None,
                        help='Base port for MP-SPDZ parties (-pn flag). Set to a unique value '
                             'per concurrent run (e.g. 5000, 5100, 5200) to avoid port conflicts.')
    args = parser.parse_args()

    run_benchmark(
        full_data_path=args.data,
        label_column=args.label,
        mpspdz_path=args.mpspdz,
        protocols=args.protocols,
        feature_sizes=args.features,
        n_runs=args.runs,
        prefix=args.prefix,
        port=args.port
    )
