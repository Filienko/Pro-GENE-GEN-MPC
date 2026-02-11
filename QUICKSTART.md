# Quick Start Guide: Running Private-PGM with MPC

This guide shows you how to run the Private-PGM pipeline on your dataset, with or without MPC.

## Prerequisites

### For Standard Mode (No MPC)
```bash
pip install numpy pandas scipy scikit-learn
```

### For MPC Mode (Additional)
```bash
# Install MP-SPDZ
git clone https://github.com/data61/MP-SPDZ.git
cd MP-SPDZ
make -j8 tldr
export MPSPDZ_PATH=$(pwd)
```

## Running on Your Dataset

### Option 1: Standard Mode (Recommended for Testing)

```bash
python run_mpc_pipeline.py \
    --data_path data/your_data.csv \
    --output_path synthetic_data.csv \
    --epsilon 1.0 \
    --delta 1e-5
```

**Example with real data:**
```bash
# If you have the AML dataset
python run_mpc_pipeline.py \
    --data_path data/aml/processed_data.csv \
    --output_path synthetic_aml.csv \
    --epsilon 1.0 \
    --delta 1e-5
```

### Option 2: MPC Mode (For Multi-Party Scenarios)

```bash
python run_mpc_pipeline.py \
    --data_path data/your_data.csv \
    --output_path synthetic_data.csv \
    --use_mpc \
    --mpspdz_path /path/to/MP-SPDZ \
    --epsilon 1.0 \
    --delta 1e-5
```

## Data Format Requirements

Your CSV file should have:
- **Features**: Continuous or integer values (will be discretized automatically)
- **Label column**: Named 'label' with integer class labels (0, 1, 2, ...)

**Example data structure:**
```
gene_0,gene_1,gene_2,...,label
0.523,1.234,0.891,...,0
1.234,0.456,1.789,...,1
...
```

## Understanding the Pipeline

The complete workflow includes these steps:

### 1. **Data Splitting**
```
Your Data → Train (80%) + Test (20%)
```

### 2. **Discretization (Binning)**
```
Continuous values → Discrete bins [0, 1, 2, 3]
```
- **Standard mode**: Uses Python quantiles locally
- **MPC mode**: Uses `ppai_bin_opt.mpc` protocol

### 3. **Marginal Computation**
```
Discrete data → 1-way & 2-way marginals (with DP noise)
```
- **Standard mode**: Computes directly on data
- **MPC mode**: Uses `ppai_msr_noisy_final` protocol

### 4. **Model Training**
```
Noisy marginals → FactoredInference → GraphicalModel
```

### 5. **Synthesis**
```
GraphicalModel → Synthetic discrete data
```

### 6. **Inverse Binning**
```
Discrete bins → Continuous values (using bin means)
```

## MPC Protocols Overview

The repository includes these MPC protocols:

| Protocol | Purpose | When Used |
|----------|---------|-----------|
| `ppai_bin.mpc` | Basic binning | Step 2 (binning) - basic version |
| `ppai_bin_opt.mpc` | Optimized binning | Step 2 (binning) - recommended |
| `ppai_bin_test.mpc` | Test binning | Step 2 (binning) - for testing |
| `ppai_msr` | Marginals (no noise) | Step 3 (marginals) - basic version |
| `ppai_msr_noisy_final` | Marginals + noise | Step 3 (marginals) - recommended |

## Simple Python API Usage

### Standard Mode
```python
from model import Private_PGM
import pandas as pd

# Load and discretize your data first
df_discrete = ...  # Your discretized data

# Create model
model = Private_PGM(
    target_variable='label',
    enable_privacy=True,
    target_epsilon=1.0,
    target_delta=1e-5,
    use_mpc=False  # Standard mode
)

# Configure domain (4 bins per feature, N classes for label)
config = {
    'gene_0': 4,
    'gene_1': 4,
    # ... more features
    'label': 5  # Number of classes
}

# Train
model.train(df_discrete, config, num_iters=1000)

# Generate synthetic data
synthetic = model.generate(num_rows=1000)
```

### MPC Mode
```python
from model import Private_PGM

# Create model with MPC enabled
model = Private_PGM(
    target_variable='label',
    enable_privacy=True,
    target_epsilon=1.0,
    target_delta=1e-5,
    use_mpc=True,  # Enable MPC
    mpspdz_path='/home/mpcuser/MP-SPDZ/',
    mpc_protocol='ring'
)

# Train with MPC protocol
model.train(
    df_discrete,
    config,
    num_iters=1000,
    mpc_protocol_file='ppai_msr_noisy_final'
)

# Generate synthetic data
synthetic = model.generate(num_rows=1000)
```

## Common Issues

### Issue: "Data file not found"
**Solution**: Provide absolute path or check your current directory
```bash
python run_mpc_pipeline.py --data_path $(pwd)/data/your_data.csv
```

### Issue: "MP-SPDZ not found"
**Solution**: MPC mode requires MP-SPDZ installation. Either:
1. Install MP-SPDZ and set `--mpspdz_path`
2. Use standard mode (remove `--use_mpc` flag)

### Issue: "Memory error with large dataset"
**Solution**: Reduce number of features or use sampling
```python
# Sample your data first
df_sample = df.sample(n=10000, random_state=42)
```

## Performance Tips

### Standard Mode
- **Speed**: Fast (~1-5 minutes for 1000 samples, 100 features)
- **Memory**: ~2GB for typical datasets
- **Use when**: Single data owner, fast iteration needed

### MPC Mode
- **Speed**: Slower (~10-100x overhead)
- **Memory**: Higher due to secret sharing
- **Network**: Requires stable connection between parties
- **Use when**: Multiple data custodians, high privacy requirements

## Example: Complete Workflow

Here's a complete example with a synthetic dataset:

```python
import numpy as np
import pandas as pd
from run_mpc_pipeline import run_standard_pipeline

# Create synthetic dataset
np.random.seed(42)
n_samples = 1000
n_features = 50

data = {
    **{f'gene_{i}': np.random.randn(n_samples) for i in range(n_features)},
    'label': np.random.randint(0, 5, n_samples)
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('my_data.csv', index=False)

# Run pipeline
run_standard_pipeline(
    data_path='my_data.csv',
    output_path='synthetic_data.csv',
    epsilon=1.0,
    delta=1e-5
)

# Load synthetic data
synthetic_df = pd.read_csv('synthetic_data.csv')
print(f"Original: {df.shape}, Synthetic: {synthetic_df.shape}")
```

## Next Steps

1. **Test with your data**: Start with standard mode
2. **Evaluate quality**: Compare synthetic vs real data distributions
3. **Adjust privacy**: Try different epsilon/delta values
4. **Scale to MPC**: When ready, enable MPC mode

## References

- [Private-PGM Paper](https://arxiv.org/abs/1901.09136)
- [MP-SPDZ Documentation](https://mp-spdz.readthedocs.io/)
- [MPC_SDG Reference](https://github.com/sikhapentyala/MPC_SDG)

## Getting Help

If you encounter issues:
1. Check `MPC_INTEGRATION.md` for detailed documentation
2. Review `example_mpc_usage.py` for code examples
3. Open an issue on GitHub with your error message
