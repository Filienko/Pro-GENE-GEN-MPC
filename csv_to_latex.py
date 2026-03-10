import csv
import os
import sys

def csv_to_latex(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]
    num_cols = len(headers)
    col_format = "l" + "c" * (num_cols - 1)  # first col left-aligned, rest centered

    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    lines = []

    lines.append(f"% Table: {table_name}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{table_name.replace('_', ' ')}}}")
    lines.append(f"    \\label{{tab:{table_name}}}")
    lines.append(f"    \\resizebox{{\\textwidth}}{{!}}{{")
    lines.append(f"    \\begin{{tabular}}{{{col_format}}}")
    lines.append(r"        \toprule")

    # Header row
    header_str = " & ".join(f"\\textbf{{{h.replace('_', r'\_')}}}" for h in headers)
    lines.append(f"        {header_str} \\\\")
    lines.append(r"        \midrule")

    # Data rows
    for row in data_rows:
        # Pad or trim row to match header length
        row = row[:num_cols] + [''] * (num_cols - len(row))
        row_str = " & ".join(str(v).replace('_', r'\_') for v in row)
        lines.append(f"        {row_str} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def process_directory(directory):
    csv_files = sorted(f for f in os.listdir(directory) if f.endswith('.csv'))

    if not csv_files:
        print("No CSV files found in directory.")
        return

    for filename in csv_files:
        filepath = os.path.join(directory, filename)
        latex = csv_to_latex(filepath)

        out_path = os.path.join(directory, os.path.splitext(filename)[0] + ".tex")
        with open(out_path, "w") as f:
            f.write(latex)

        print(f"✓ {filename} → {os.path.basename(out_path)}")

    print(f"\nDone. {len(csv_files)} table(s) generated.")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    process_directory(directory)
