# MPC Integration for Private-PGM

## Overview

This document describes the integration of Multi-Party Computation (MPC) into the Private-PGM model, enabling secure computation of marginals across multiple data custodians without revealing raw data.

## What's New

### 1. MPC Support in Private_PGM Class

The `Private_PGM` class now supports two modes of operation:

- **Standard Mode (default)**: Computes marginals directly on plaintext data (existing functionality)
- **MPC Mode (new)**: Computes marginals using secure multi-party computation via MP-SPDZ

### 2. New Components

#### MPC Helper Module (`utils/mpc_helper.py`)

Three new classes for MPC integration:

- `MPCProtocolExecutor`: Handles compilation and execution of .mpc protocols with MP-SPDZ
- `HorizontalDataSplitter`: Splits data horizontally across multiple parties
- `MPCMarginalComputer`: High-level interface for computing marginals using MPC

#### Example Usage Script (`example_mpc_usage.py`)

Demonstrates how to use both standard and MPC modes.

## Architecture

### Standard Mode Flow

```
Raw Data → Compute Marginals → Add Noise → Inference → Generate Synthetic Data
```

### MPC Mode Flow

```
Raw Data → Split Between Parties → MPC Protocol → Noisy Marginals → Inference → Generate Synthetic Data
                                     ↓
                              (Data remains encrypted)
```

## Key Differences: Standard vs MPC

| Aspect | Standard Mode | MPC Mode |
|--------|---------------|----------|
| Data Access | Direct access to plaintext data | Data remains secret-shared |
| Computation | Local computation | Distributed secure computation |
| Privacy | Differential Privacy only | Differential Privacy + MPC security |
| Requirements | None | MP-SPDZ installation required |
| Performance | Faster | Slower (network overhead) |
| Use Case | Single data custodian | Multiple data custodians |

## Usage

### Standard Mode (Existing Functionality)

```python
from model import Private_PGM

model = Private_PGM(
    target_variable='label',
    enable_privacy=True,
    target_epsilon=1.0,
    target_delta=1e-5,
    use_mpc=False  # Standard mode (default)
)

model.train(data, config, num_iters=1000)
synthetic_data = model.generate(num_rows=500)
```

### MPC Mode (New Functionality)

```python
from model import Private_PGM

model = Private_PGM(
    target_variable='label',
    enable_privacy=True,
    target_epsilon=1.0,
    target_delta=1e-5,
    use_mpc=True,  # Enable MPC mode
    mpspdz_path='/home/mpcuser/MP-SPDZ/',  # Path to MP-SPDZ
    mpc_protocol='ring'  # MPC protocol to use
)

model.train(
    data,
    config,
    num_iters=1000,
    mpc_protocol_file='ppai_msr_noisy_final'  # MPC protocol for marginals
)

synthetic_data = model.generate(num_rows=500)
```

## MPC Protocols

The repository includes several MPC protocol files:

### Marginal Computation Protocols

- **`ppai_msr`**: Basic marginal computation protocol
  - Computes 1-way and 2-way marginals
  - No noise addition

- **`ppai_msr_noisy_final`**: Complete marginal computation with noise (recommended)
  - Computes 1-way and 2-way marginals
  - Adds Gaussian or Laplace noise for differential privacy
  - Optimized performance with unrolled loops

### Binning Protocols

- **`ppai_bin.mpc`**: Basic binning protocol
- **`ppai_bin_opt.mpc`**: Optimized binning
- **`ppai_bin_test.mpc`**: Test version with fold-based binning

## Requirements

### For Standard Mode
- Python 3.7+
- NumPy
- SciPy
- Pandas
- mbi library (included in Private_PGM)

### For MPC Mode (Additional)
- MP-SPDZ installation
- Network connectivity between MPC parties
- Sufficient computational resources for secure computation

## Installation

### 1. Install MP-SPDZ

```bash
# Clone MP-SPDZ repository
git clone https://github.com/data61/MP-SPDZ.git
cd MP-SPDZ

# Compile
make -j8 tldr

# Set environment variable
export MPSPDZ_PATH=/path/to/MP-SPDZ
```

### 2. Verify Installation

```bash
# Test MP-SPDZ installation
cd MP-SPDZ
./compile.py tutorial
Scripts/ring.sh tutorial
```

## API Reference

### Private_PGM.__init__()

```python
def __init__(
    target_variable,      # str: Target variable name
    enable_privacy,       # bool: Enable differential privacy
    target_epsilon,       # float: Privacy parameter epsilon
    target_delta,         # float: Privacy parameter delta
    use_mpc=False,        # bool: Enable MPC mode
    mpspdz_path=None,     # str: Path to MP-SPDZ installation
    mpc_protocol='ring'   # str: MPC protocol to use
)
```

### Private_PGM.train()

```python
def train(
    train_df,              # pd.DataFrame: Training data
    config,                # dict: Domain configuration
    cliques=None,          # list: Clique tuples for 2-way marginals
    num_iters=10000,       # int: Number of inference iterations
    mpc_protocol_file=None # str: Path to MPC protocol file (MPC mode only)
)
```

## Implementation Details

### Marginal Computation in MPC

The MPC protocol (`ppai_msr_noisy_final`) computes:

1. **1-way marginals**:
   - For each feature: counts for each bin (0, 1, 2, 3)
   - For labels: counts for each class

2. **2-way marginals**:
   - For each (feature, label) pair: joint counts (4 bins × 5 classes = 20 values)

3. **Noise addition**:
   - Gaussian noise (if target_delta > 0)
   - Laplace noise (if target_delta = 0)

### Data Splitting

Data is split horizontally between parties:
- Party 0: First 50% of samples
- Party 1: Last 50% of samples

Each party's data remains secret throughout the computation.

### Security Guarantees

- **Differential Privacy**: (ε, δ)-DP guarantee through noise addition
- **MPC Security**: Semi-honest security against individual parties
- **Combined**: Protection against both data inference and party collusion

## Performance Considerations

### Standard Mode
- Fast: O(n × d) for n samples and d features
- Memory: O(n × d)
- Suitable for: Single machine, fast iteration

### MPC Mode
- Slower: 10-100× overhead due to secure computation
- Network: Requires stable network between parties
- Memory: Higher due to secret sharing
- Suitable for: Multi-party scenarios, high privacy requirements

## Troubleshooting

### Error: "MP-SPDZ compile.py not found"
- Ensure MP-SPDZ is installed
- Set `mpspdz_path` parameter correctly
- Check MPSPDZ_PATH environment variable

### Error: "Protocol script not found"
- Verify MP-SPDZ installation is complete
- Check Scripts/ directory exists in MP-SPDZ
- Ensure protocol parameter matches available scripts

### Error: "MPC protocol file not found"
- Check `mpc_protocol_file` path is correct
- Ensure .mpc file exists in repository
- Use absolute path if relative path fails

### MPC Execution Hangs
- Check network connectivity between parties
- Verify firewall settings allow MPC traffic
- Increase timeout if computation is very large

## Examples

See `example_mpc_usage.py` for complete working examples.

## References

1. [Private-PGM Original](https://github.com/ryan112358/private-pgm)
2. [MP-SPDZ Framework](https://github.com/data61/MP-SPDZ)
3. [MPC_SDG Reference Implementation](https://github.com/sikhapentyala/MPC_SDG/blob/icml/private-pgm-master/mechanisms/aim_MPC_H.py)

## Contributing

To add new MPC protocols:

1. Create .mpc file following MP-SPDZ syntax
2. Implement marginal computation with appropriate noise
3. Ensure output format matches expected structure
4. Update documentation

## License

Same as original Private-PGM implementation (see LICENSE file).
