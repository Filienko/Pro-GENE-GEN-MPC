import pandas as pd

def combine_splits(train_path, val_path, output_path):
    # Load splits
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Concatenate
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Save
    combined_df.to_csv(output_path, index=False)

    print(f"Combined file saved to: {output_path}")
    print(f"Total rows: {len(combined_df)}")

if __name__ == "__main__":
    combine_splits(
        train_path="counts_with_who2022_train_subset.csv",
        val_path="counts_with_who2022_test_subset.csv",
        output_path="counts_with_who2022_full.csv"
    )

