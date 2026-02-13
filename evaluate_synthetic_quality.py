#!/usr/bin/env python3
"""
Evaluate synthetic data quality by training on synthetic vs real data
and testing on real test data.

- Synthetic data: label column = 'label' (encoded)
- Real data: label column = original name (encoded using saved encoder)
"""

import pandas as pd
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


def load_synthetic(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Synthetic file {path} must contain 'label'")
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    return X, y


def load_real(path, label_col, encoder):
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")

    y_raw = df[label_col]

    # Encode using saved encoder
    label_to_int = encoder["label_to_int"]

    if y_raw.dtype == object or y_raw.dtype.name == "category":
        y = y_raw.map(label_to_int)
    else:
        y = y_raw.astype(int)

    if y.isna().any():
        bad = y[y.isna()].index.tolist()[:5]
        raise ValueError(f"Unmapped labels found in {path} at rows {bad}")

    X = df.drop(columns=[label_col])

    return X, y.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_train", required=True)
    parser.add_argument("--real_train", required=True)
    parser.add_argument("--real_test", required=True)
    parser.add_argument("--real_label_col", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    args = parser.parse_args()

    # =========================
    # LOAD LABEL ENCODER
    # =========================
    with open(args.encoder) as f:
        encoder = json.load(f)

    # =========================
    # LOAD DATA
    # =========================
    X_syn, y_syn = load_synthetic(args.synthetic_train)
    X_real_tr, y_real_tr = load_real(
        args.real_train, args.real_label_col, encoder
    )
    X_real_te, y_real_te = load_real(
        args.real_test, args.real_label_col, encoder
    )

    # =========================
    # ALIGN FEATURE SPACE
    # =========================
    syn_features = X_syn.columns.tolist()

    missing = set(syn_features) - set(X_real_tr.columns)
    if missing:
        raise ValueError(f"Real data missing features: {list(missing)[:5]}")

    X_real_tr = X_real_tr[syn_features]
    X_real_te = X_real_te[syn_features]

    print(f"✓ Feature space aligned ({len(syn_features)} features)")

    # =========================
    # TRAIN MODELS
    # =========================
    clf_real = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1
    )

    clf_syn = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1
    )

    clf_real.fit(X_real_tr, y_real_tr)
    clf_syn.fit(X_syn, y_syn)

    # =========================
    # EVALUATE
    # =========================
    y_pred_real = clf_real.predict(X_real_te)
    y_pred_syn = clf_syn.predict(X_real_te)

    print("\n" + "=" * 80)
    print("EVALUATION ON REAL TEST DATA")
    print("=" * 80)

    print("\n[TRAINED ON REAL DATA]")
    print(f"Accuracy: {accuracy_score(y_real_te, y_pred_real):.4f}")
    print(f"Macro-F1: {f1_score(y_real_te, y_pred_real, average='macro'):.4f}")
    print(classification_report(y_real_te, y_pred_real))

    print("\n[TRAINED ON SYNTHETIC DATA]")
    print(f"Accuracy: {accuracy_score(y_real_te, y_pred_syn):.4f}")
    print(f"Macro-F1: {f1_score(y_real_te, y_pred_syn, average='macro'):.4f}")
    print(classification_report(y_real_te, y_pred_syn))

    print("=" * 80)


if __name__ == "__main__":
    main()
