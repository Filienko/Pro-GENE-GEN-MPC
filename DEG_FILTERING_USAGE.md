# DEG Filtering Usage Guide

## Overview

The secure MPC pipeline now supports **Differentially Private DEG (Differentially Expressed Genes) Filtering**. This feature allows you to select the top-k most significant genes using secure ANOVA F-test and the Exponential Mechanism, all within the MPC framework.

## Key Features

1. **DP-Protected Feature Selection**: Uses the Exponential Mechanism to select top-k DEGs with differential privacy guarantees
2. **Secure ANOVA**: Computes F-statistics for all genes securely in MPC
3. **Integrated Workflow**: Selection, binning, and marginal computation all happen in one MPC protocol
4. **No Data Leakage**: Raw data, F-statistics, and binned data remain secret throughout

## Usage

### Basic Command (All Genes - Original Behavior)

```bash
python run_secure_mpc_pipeline.py \
    --party_files party_0.csv party_1.csv \
    --output_path synthetic_data.csv \
    --mpspdz_path /path/to/MP-SPDZ/ \
    --epsilon 1.0 \
    --delta 1e-5
```

### With DEG Filtering (Top-K Genes)

```bash
python run_secure_mpc_pipeline.py \
    --party_files party_0.csv party_1.csv \
    --output_path synthetic_data.csv \
    --mpspdz_path /path/to/MP-SPDZ/ \
    --epsilon 1.0 \
    --delta 1e-5 \
    --top_k_genes 10
```

This will:
1. Securely compute ANOVA F-statistics for all genes
2. Use the Exponential Mechanism to select the top-10 DEGs with DP protection
3. Bin only the selected 10 genes
4. Compute marginals only for the selected 10 genes
5. Generate synthetic data with 10 features (genes) + 1 label

## Setup Instructions

### 1. Copy MPC Protocol to MP-SPDZ

The DEG filtering feature requires the `integrated_dp_deg_pgm.mpc` protocol file:

```bash
# Copy the MPC protocol to your MP-SPDZ directory
cp integrated_dp_deg_pgm.mpc /path/to/MP-SPDZ/
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## How It Works

### Standard Mode (No DEG Filtering)
1. **Input**: All genes from each party
2. **Binning**: Bins all genes in MPC
3. **Marginals**: Computes marginals for all genes
4. **Output**: Synthetic data with all genes

### DEG Filtering Mode
1. **Input**: All genes from each party
2. **DEG Selection**:
   - Computes ANOVA F-statistics for all genes (secret)
   - Adds Gumbel noise using Exponential Mechanism
   - Selects top-k genes (DP-protected)
3. **Binning**: Bins only the k selected genes
4. **Marginals**: Computes marginals only for k genes
5. **Output**: Synthetic data with k genes

## Privacy Guarantees

- **Standard Mode**: (ε, δ)-DP on binning and marginals
- **DEG Filtering Mode**: (ε, δ)-DP on DEG selection, binning, and marginals
  - The Exponential Mechanism protects the selection process
  - Even though selected gene indices are revealed, they are DP-protected

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--party_files` | List[str] | Required | Paths to party data CSV files |
| `--output_path` | str | `synthetic_data_secure.csv` | Output path for synthetic data |
| `--mpspdz_path` | str | Required | Path to MP-SPDZ installation |
| `--epsilon` | float | 1.0 | Privacy parameter ε |
| `--delta` | float | 1e-5 | Privacy parameter δ |
| `--top_k_genes` | int | None | Number of top DEGs to select (enables filtering) |
| `--num_iters` | int | 10000 | Number of inference iterations |

## Example Scenarios

### Scenario 1: High-Dimensional Data (1000+ genes)
```bash
# Select top 20 most significant genes
python run_secure_mpc_pipeline.py \
    --party_files hospital_a.csv hospital_b.csv \
    --output_path synthetic_top20.csv \
    --mpspdz_path /path/to/MP-SPDZ/ \
    --top_k_genes 20 \
    --epsilon 2.0
```

### Scenario 2: Standard Analysis (All genes)
```bash
# Use all genes (no filtering)
python run_secure_mpc_pipeline.py \
    --party_files hospital_a.csv hospital_b.csv \
    --output_path synthetic_all_genes.csv \
    --mpspdz_path /path/to/MP-SPDZ/
```

## Output Format

### Standard Mode
- Columns: `gene_0, gene_1, ..., gene_N, label`
- N = total number of input genes

### DEG Filtering Mode
- Columns: `gene_0, gene_1, ..., gene_{k-1}, label`
- k = value of `--top_k_genes`
- Note: Generic names are used for selected genes

## Technical Details

### MPC Protocol Files

1. **Standard Mode**: Uses `ppai_bin_msr.mpc`
   - Location: `utils/mpc_helper.py::compute_marginals_with_binning()`

2. **DEG Filtering Mode**: Uses `integrated_dp_deg_pgm.mpc`
   - Location: `utils/mpc_helper.py::compute_marginals_with_deg_filtering()`

### Code Modifications

Files modified to support DEG filtering:
- `utils/mpc_helper.py`: Added `compute_marginals_with_deg_filtering()` and `_parse_deg_marginals_output()`
- `models/Private_PGM/model_secure_mpc.py`: Added `top_k_genes` parameter and conditional logic
- `run_secure_mpc_pipeline.py`: Added `--top_k_genes` command-line argument

## Troubleshooting

### Error: "MPC protocol file not found"
```
Solution: Copy integrated_dp_deg_pgm.mpc to your MP-SPDZ directory
cp integrated_dp_deg_pgm.mpc /path/to/MP-SPDZ/
```

### Error: "Expected X genes, got Y"
```
Solution: When using DEG filtering, the output has k genes, not all genes.
This is expected behavior.
```

### Error: "Parsed X values, expected Y"
```
Solution: Ensure your MPC protocol arguments match:
- party_sizes (auto-detected from files)
- num_genes (auto-detected from files)
- num_classes (auto-detected from data)
- k (from --top_k_genes argument)
```

## Performance Considerations

- **Standard Mode**: Time complexity depends on total genes
- **DEG Filtering Mode**:
  - Initial ANOVA: O(all genes)
  - Selection: O(all genes × k)
  - Binning/Marginals: O(k genes) - **FASTER** than standard mode when k << total genes

**Recommendation**: Use DEG filtering when:
- You have high-dimensional data (100+ genes)
- You want to focus on the most discriminative features
- You want faster downstream synthesis (fewer marginals to fit)

## Security Considerations

✅ **What is protected:**
- Raw patient data (never leaves parties)
- F-statistics for all genes (computed in MPC)
- Binned data (never revealed)
- Exact marginal counts (DP noise added)

❌ **What is revealed (DP-protected):**
- Selected gene indices (protected by Exponential Mechanism)
- Noisy marginals for selected genes
- Noisy bin means (if applicable)

## References

- Exponential Mechanism: McSherry & Talwar (2007)
- Private-PGM: McKenna et al. (2021)
- MP-SPDZ: https://github.com/data61/MP-SPDZ
