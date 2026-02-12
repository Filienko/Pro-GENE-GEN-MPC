# MPC-based Private PGM Implementation

This directory contains an MPC (Multi-Party Computation) implementation of Private PGM for privacy-preserving synthetic data generation. The implementation uses MP-SPDZ library for secure computation protocols.

## Overview

The MPC implementation replaces the centralized binning and marginal computation with secure multi-party computation protocols, enabling multiple parties to jointly train a generative model without revealing their individual data.

## Key Components

### 1. MPC Protocols (.mpc files)

Located in the root directory:
- **ppai_bin.mpc**: MPC protocol for secure binning of features
  - Performs quantile-based binning on secret-shared data
  - Computes bin means with differential privacy noise
  - Supports multi-party horizontal data partitioning

- **ppai_msr_final.mpc**: MPC protocol for computing marginals
  - Computes 1-way feature marginals (counts for each feature value)
  - Computes 1-way label marginals (counts for each class)
  - Computes 2-way marginals (joint counts of features and labels)
  - Adds Gaussian or Laplace noise for differential privacy

### 2. Python Implementation

- **mpc_utils.py**: Utility classes and functions for MPC
  - `HDataHolder`: Class for managing data for each MPC party
  - `split_data_horizontal()`: Splits data horizontally across parties
  - `compile_mpc_protocol()`: Compiles .mpc protocols using MP-SPDZ
  - `run_mpc_protocol()`: Executes compiled MPC protocols

- **mpc_model.py**: MPC-based Private PGM model
  - `MPC_PrivatePGM`: Main class that orchestrates MPC computation
  - Integrates with the existing FactoredInference engine
  - Supports the same interface as the original `Private_PGM`

- **main.py**: Training script (updated to support MPC)
  - Added `--use_mpc` flag to enable MPC mode
  - Added `--mpc_path` to specify MP-SPDZ installation path
  - Added `--mpc_protocol` to choose MPC protocol (ring, semi2k, mascot)

## Architecture

### Data Flow

```
┌─────────────┐     ┌─────────────┐
│   Party A   │     │   Party B   │
│   (Alice)   │     │    (Bob)    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       │   Secret Shares   │
       ├───────────────────┤
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│    MP-SPDZ MPC Computation      │
│  ┌───────────────────────────┐  │
│  │  1. Binning (ppai_bin)    │  │
│  │  2. Marginals (ppai_msr)  │  │
│  └───────────────────────────┘  │
└────────────────┬────────────────┘
                 │
                 ▼
     ┌────────────────────┐
     │  Noisy Marginals   │
     └─────────┬──────────┘
               │
               ▼
     ┌────────────────────┐
     │ FactoredInference  │
     │   (PGM Learning)   │
     └─────────┬──────────┘
               │
               ▼
     ┌────────────────────┐
     │  Synthetic Data    │
     └────────────────────┘
```

### MPC Workflow

1. **Data Partitioning**: Training data is split horizontally between two parties (Alice and Bob)
2. **Secret Sharing**: Each party secret-shares their data using MP-SPDZ
3. **MPC Binning** (Optional): Secure binning of features using quantiles
4. **MPC Marginals**: Secure computation of 1-way and 2-way marginals with DP noise
5. **Model Learning**: FactoredInference uses the noisy marginals to learn a PGM
6. **Synthetic Data Generation**: Generate synthetic data from the learned model

## Installation

### Prerequisites

1. **MP-SPDZ**: Install the MP-SPDZ library
   ```bash
   git clone https://github.com/data61/MP-SPDZ.git
   cd MP-SPDZ
   make -j 8 tldr
   ```

2. **Set environment variable** (optional):
   ```bash
   export MPC_PATH=/path/to/MP-SPDZ
   ```

3. **Python dependencies**: Already included in the project requirements

### Copying MPC Protocols

Copy the .mpc protocol files to the MP-SPDZ Programs directory:
```bash
cp ppai_bin.mpc $MPC_PATH/Programs/Source/
cp ppai_msr_final.mpc $MPC_PATH/Programs/Source/
```

## Usage

### Basic Usage with MPC

Run the training script with MPC enabled:

```bash
python main.py \
    --exp_name my_mpc_experiment \
    --dataset aml \
    --enable_privacy \
    --target_epsilon 8.0 \
    --target_delta 1e-5 \
    --use_mpc \
    --mpc_path /path/to/MP-SPDZ \
    --mpc_protocol ring \
    --num_iters 10000
```

### Parameters

**MPC-specific parameters:**
- `--use_mpc`: Enable MPC-based computation (default: False)
- `--mpc_path`: Path to MP-SPDZ installation (default: /opt/MP-SPDZ or $MPC_PATH)
- `--mpc_protocol`: MPC protocol to use (default: ring)
  - `ring`: Ring-based protocol (fastest)
  - `semi2k`: Semi-honest 2-party protocol
  - `mascot`: MASCOT protocol with authentication

**Privacy parameters:**
- `--enable_privacy`: Enable differential privacy
- `--target_epsilon`: Privacy budget (epsilon)
- `--target_delta`: Privacy parameter (delta)

**Other parameters:**
- `--dataset`: Dataset name (aml, mouse, bulk_aml)
- `--num_iters`: Number of iterations for PGM learning
- `--num_samples_ratio`: Ratio of synthetic samples to generate

### Example Commands

1. **MPC with default settings:**
   ```bash
   python main.py --exp_name test_mpc --dataset aml --use_mpc
   ```

2. **MPC with custom MP-SPDZ path:**
   ```bash
   python main.py \
       --exp_name test_mpc \
       --dataset aml \
       --use_mpc \
       --mpc_path ~/MP-SPDZ \
       --enable_privacy \
       --target_epsilon 5.0
   ```

3. **Compare MPC vs. centralized:**
   ```bash
   # Centralized version
   python main.py --exp_name centralized --dataset aml --enable_privacy

   # MPC version
   python main.py --exp_name mpc --dataset aml --enable_privacy --use_mpc
   ```

## Implementation Details

### Current Implementation Status

The current implementation provides:
1. ✅ Complete MPC infrastructure (HDataHolder, compilation, execution)
2. ✅ Integration with existing Private PGM pipeline
3. ✅ Command-line interface for MPC mode
4. ⚠️ MPC protocols are ready but marginals computation is using traditional approach

### Next Steps for Full MPC Integration

To fully enable MPC computation, uncomment the MPC protocol calls in `mpc_model.py`:

```python
# In train() method of MPC_PrivatePGM class:

# Option 1: Use MPC for binning
binned_data, bin_means = self._run_binning_mpc(
    num_genes, num_patients_per_party, num_classes
)

# Option 2: Use MPC for marginals computation
marginals_1way, marginals_1way_labels, marginals_2way = self._run_marginals_mpc(
    total, num_genes, num_classes
)
measurements = self._create_measurements_from_mpc(
    data, marginals_1way, marginals_1way_labels, marginals_2way, sigma
)
```

### Extending the Implementation

**To add new MPC protocols:**

1. Create a new .mpc file in the root directory
2. Add compilation and execution methods in `mpc_model.py`
3. Update the workflow in the `train()` method

**To support more than 2 parties:**

1. Update `n_parties` in `MPC_PrivatePGM.__init__()`
2. Modify data partitioning in `split_data_horizontal()`
3. Update MPC protocol compilation arguments

## Reference

This implementation is based on the MPC approach from:
- Repository: https://github.com/sikhapentyala/MPC_SDG
- File: `mechanisms/aim_MPC_H.py`

The implementation uses a similar architecture with:
- HDataHolder for managing party data
- MP-SPDZ compilation and execution workflow
- Integration with existing PGM learning infrastructure

## Troubleshooting

### Common Issues

1. **MP-SPDZ not found:**
   - Set the `MPC_PATH` environment variable
   - Or use `--mpc_path` argument

2. **Compilation errors:**
   - Ensure .mpc files are in `$MPC_PATH/Programs/Source/`
   - Check that MP-SPDZ is properly installed

3. **Runtime errors:**
   - Check that the protocol name matches the .mpc file
   - Verify that input data dimensions match protocol arguments

4. **Memory issues:**
   - Use smaller datasets for testing
   - Reduce number of iterations (`--num_iters`)
   - Consider using the `ring` protocol (fastest)

## Performance Considerations

- **Protocol choice**: `ring` is fastest, `mascot` provides authentication
- **Network**: MPC requires network communication between parties
- **Computation**: MPC computation is slower than centralized (expect 10-100x overhead)
- **Privacy-Utility tradeoff**: Lower epsilon = more privacy but potentially lower utility

## Security Notes

- The current implementation simulates both parties on the same machine
- For production use, deploy parties on separate machines
- Ensure secure network channels between parties
- Consider using authenticated protocols (e.g., mascot) for malicious security
- Audit MPC protocols for privacy guarantees

## License

This MPC implementation follows the same license as the main project.
See `LICENSE` file in the `models/Private_PGM/` directory.
