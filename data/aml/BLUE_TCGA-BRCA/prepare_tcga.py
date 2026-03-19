import argparse
import pandas as pd

def prepare_dataset(expr_path, label_path, output_path=None):
    """
    Convert TCGA gene x sample matrix to sample x gene
    and merge with WHO_2022 labels.

    Parameters
    ----------
    expr_path : str
        Path to expression TSV file (genes x samples)
    label_path : str
        Path to labels CSV file
    output_path : str (optional)
        If provided, saves merged dataframe to CSV
    """


    # -----------------------
    # 1. Load expression data
    # -----------------------
    print("Loading expression matrix...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    # genes x samples  -->  samples x genes
    expr = expr.T
    expr.index.name = "sample_id"

    print(f"Expression shape after transpose: {expr.shape}")

    # -----------------------
    # 2. Load labels
    # -----------------------
    print("Loading labels...")
    labels = pd.read_csv(label_path)

    if "samplesID" not in labels.columns:
        raise ValueError("Expected column 'samplesID' in label file.")

    if "Subtype" not in labels.columns:
        raise ValueError("Expected column '' in label file.")

    labels = labels.set_index("samplesID")


    # -----------------------
    # 3. Merge
    # -----------------------
    print("Merging expression with labels...")
    df = expr.join(labels["Subtype"], how="inner")

    print(f"Final dataset shape: {df.shape}")

    # -----------------------
    # 4. Save (optional)
    # -----------------------
    if output_path:
        df.to_csv(output_path)
        print(f"Saved to {output_path}")

    return df
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare TCGA expression matrix with the labels."
    )

    parser.add_argument("--expr", required=True, help="Expression TSV file")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument("--out", required=False, help="Output CSV file")

    args = parser.parse_args()

    prepare_dataset(args.expr, args.labels, args.out)

