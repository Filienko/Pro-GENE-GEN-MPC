#!/usr/bin/env python3
"""
Compare in-the-clear ANOVA vs MPC ANOVA (MP-SPDZ)

Outputs:
- Top-K overlap metrics
- Rank correlation metrics
- Importance preservation metrics
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from scipy.stats import spearmanr, kendalltau
from utils.mpc_helper import MPCProtocolExecutor


# ============================================================================
# PLAINTEXT ANOVA
# ============================================================================

def run_clear_anova(party_files, top_k):
    print("\n" + "=" * 80)
    print("RUNNING IN-THE-CLEAR ANOVA")
    print("=" * 80)

    dfs = [pd.read_csv(f) for f in party_files]
    df = pd.concat(dfs, ignore_index=True)

    target_col = "label" if "label" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    f_values, _ = f_classif(X, y)
    f_values = np.nan_to_num(f_values)

    ranking = np.argsort(f_values)[::-1]

    print(f"Top-{top_k} Features (Clear):")
    for i in range(top_k):
        idx = ranking[i]
        print(
            f"Rank {i+1}: Feature {idx} "
            f"({X.columns[idx]}), F={f_values[idx]:.4f}"
        )

    return ranking.tolist(), f_values, X.columns.tolist()


# ============================================================================
# MPC INPUT PREPARATION
# ============================================================================

def prepare_mpc_inputs(party_files, mpspdz_path):
    player_data_dir = os.path.join(mpspdz_path, "Player-Data")
    os.makedirs(player_data_dir, exist_ok=True)

    dfs = [pd.read_csv(f) for f in party_files]
    combined = pd.concat(dfs)

    target_col = "label" if "label" in combined.columns else combined.columns[-1]

    M = combined.shape[1] - 1
    C = int(combined[target_col].max() + 1)
    N = []

    for i, df in enumerate(dfs):
        N.append(len(df))
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values.astype(int)

        y_onehot = np.zeros((len(y), C), dtype=int)
        for r, c in enumerate(y):
            y_onehot[r, c] = 1

        input_file = os.path.join(player_data_dir, f"Input-P{i}-0")
        with open(input_file, "w") as f:
            for r in range(len(df)):
                f.write(
                    " ".join(map(str, X[r])) + " " +
                    " ".join(map(str, y_onehot[r])) + "\n"
                )

    return N, M, C


# ============================================================================
# MPC ANOVA
# ============================================================================

def parse_mpc_ranking(stdout):
    print("\n--- RAW MPC OUTPUT START ---")
    print(stdout.strip())
    print("--- RAW MPC OUTPUT END ---\n")
    
    ranking = []
    for line in stdout.split("\n"):
        if "Rank" in line and "Feature" in line:
            # Safely handle the string splitting
            try:
                idx = int(line.split("Feature")[-1].strip())
                ranking.append(idx)
            except ValueError:
                continue
                
    if not ranking:
        print("⚠️ WARNING: Could not parse any rankings. Check the raw output above for MP-SPDZ errors!")
        
    return ranking


def run_mpc_anova(party_files, mpspdz_path, top_k):
    print("\n" + "=" * 80)
    print("RUNNING MPC ANOVA")
    print("=" * 80)

    mpc_source = "deg_anova.mpc"
    src = os.path.join(mpspdz_path, "Programs", "Source", mpc_source)

    if os.path.exists(mpc_source):
        shutil.copy(mpc_source, src)
        print("✓ MPC source copied")

    N, M, C = prepare_mpc_inputs(party_files, mpspdz_path)
    print(f"✓ Data prepared: N={N}, M={M}, C={C}")

    executor = MPCProtocolExecutor(
        mpspdz_path=os.path.abspath(mpspdz_path),
        protocol="ring"
    )

    args = [N[0], N[1], M, C, top_k]
    program = mpc_source[:-4]

    executor.compile_protocol(program, args=args)

    result = executor.execute_protocol(
        program,
        num_parties=2,
        args=args
    )

    ranking = parse_mpc_ranking(result.stdout)

    print("Top-K Features (MPC):")
    for i, idx in enumerate(ranking):
        print(f"Rank {i+1}: Feature {idx}")

    return ranking


# ============================================================================
# RANKING METRICS
# ============================================================================

def evaluate_rankings(clear_rank, mpc_rank, clear_scores, top_k):
    clear_topk = set(clear_rank[:top_k])
    mpc_topk = set(mpc_rank[:top_k])

    overlap = len(clear_topk & mpc_topk)
    union = len(clear_topk | mpc_topk)

    precision = overlap / top_k
    recall = overlap / top_k
    jaccard = overlap / union if union > 0 else 0

    clear_pos = {f: i for i, f in enumerate(clear_rank)}
    mpc_pos = {f: i for i, f in enumerate(mpc_rank)}

    common = list(set(clear_pos) & set(mpc_pos))
    clear_r = [clear_pos[f] for f in common]
    mpc_r = [mpc_pos[f] for f in common]

    spearman, _ = spearmanr(clear_r, mpc_r)
    kendall, _ = kendalltau(clear_r, mpc_r)

    clear_topk_score = np.mean([clear_scores[i] for i in clear_topk])
    mpc_topk_score = np.mean([clear_scores[i] for i in mpc_topk])
    ratio = mpc_topk_score / clear_topk_score if clear_topk_score > 0 else 0

    print("\n" + "=" * 80)
    print("RANKING COMPARISON METRICS")
    print("=" * 80)

    print(f"Overlap@{top_k}        : {overlap}/{top_k}")
    print(f"Precision@{top_k}     : {precision:.3f}")
    print(f"Recall@{top_k}        : {recall:.3f}")
    print(f"Jaccard@{top_k}       : {jaccard:.3f}")

    print("\nGlobal Ranking Agreement:")
    print(f"Spearman ρ            : {spearman:.3f}")
    print(f"Kendall τ             : {kendall:.3f}")

    print("\nImportance Preservation:")
    print(f"Mean clear F (clear)  : {clear_topk_score:.4f}")
    print(f"Mean clear F (MPC)    : {mpc_topk_score:.4f}")
    print(f"Relative ratio        : {ratio:.3f}")

    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--party_files", nargs=2, required=True)
    parser.add_argument("--mpspdz_path", required=True)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    clear_rank, clear_scores, feature_names = run_clear_anova(
        args.party_files, args.top_k
    )

    mpc_rank = run_mpc_anova(
        args.party_files, args.mpspdz_path, args.top_k
    )

    evaluate_rankings(
        clear_rank=clear_rank,
        mpc_rank=mpc_rank,
        clear_scores=clear_scores,
        top_k=args.top_k
    )
