import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Rename a column to 'label' and split CSV horizontally.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--label_col", required=True, help="Column name to rename to 'label'")
    parser.add_argument("--split", type=float, default=0.5,
                        help="Fraction of rows for party 1 (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found in CSV.")

    # Rename column to 'label'
    df = df.rename(columns={args.label_col: "label"})

    # Shuffle rows
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split
    split_idx = int(len(df) * args.split)
    df1 = df.iloc[:split_idx]
    df2 = df.iloc[split_idx:]

    # Output filenames
    base, ext = os.path.splitext(args.csv)
    out1 = f"{base}_party_1{ext}"
    out2 = f"{base}_party_2{ext}"

    # Save
    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)

    print(f"Saved {out1} ({len(df1)} rows)")
    print(f"Saved {out2} ({len(df2)} rows)")

if __name__ == "__main__":
    main()
