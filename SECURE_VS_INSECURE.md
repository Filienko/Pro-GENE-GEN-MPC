# Secure vs Insecure Implementation Comparison

## Overview

This document compares the two implementations and explains when to use each.

## ⚠️ CRITICAL: Which Version Should You Use?

| Scenario | Use This Script | Security Level |
|----------|----------------|----------------|
| **Production with sensitive data** | `run_secure_mpc_pipeline.py` | 🔒 **SECURE** - Full MPC + DP |
| **Multiple data custodians** | `run_secure_mpc_pipeline.py` | 🔒 **SECURE** - Full MPC + DP |
| **Healthcare, financial data** | `run_secure_mpc_pipeline.py` | 🔒 **SECURE** - Full MPC + DP |
| **Testing with fake data** | `run_mpc_pipeline.py` | ⚠️ **INSECURE** - DP only |
| **Development and debugging** | `run_mpc_pipeline.py` | ⚠️ **INSECURE** - DP only |
| **Single data owner** | `run_mpc_pipeline.py` | ⚠️ **INSECURE** - DP only |

## Detailed Comparison

### File Comparison Matrix

| File | Security | MPC Binning | MPC Marginals | Data Leakage | Use Case |
|------|----------|-------------|---------------|--------------|----------|
| `run_secure_mpc_pipeline.py` | ✅ Secure | ✅ Yes | ✅ Yes | ❌ None | **PRODUCTION** |
| `run_mpc_pipeline.py` | ⚠️ Insecure | ❌ No | ⚠️ Optional | ✅ Yes | Testing only |
| `models/Private_PGM/model_secure_mpc.py` | ✅ Secure | ✅ Yes | ✅ Yes | ❌ None | **PRODUCTION** |
| `models/Private_PGM/model.py` | ⚠️ Mixed | ❌ No | ⚠️ Optional | ⚠️ Partial | Legacy |

## Security Analysis

### ✅ Secure Implementation (`run_secure_mpc_pipeline.py`)

**Data Flow:**
```
Party A: raw_data_A.csv ─┐
                          ├─> [MPC Binning] ─> [MPC Marginals] ─> Noisy Stats ─> Synthesis
Party B: raw_data_B.csv ─┘         ↑                  ↑              ↑
                              Protected!         Protected!      Public (DP)
```

**Security Properties:**
- ✅ Raw data never leaves parties
- ✅ Binning done entirely in MPC
- ✅ Marginals computed entirely in MPC
- ✅ Only DP-protected noisy statistics revealed
- ✅ Full (ε,δ)-DP guarantee
- ✅ Semi-honest MPC security

**What's Protected:**
1. **Individual values**: Never revealed
2. **Quantiles**: Computed in MPC
3. **Bin assignments**: Computed in MPC
4. **Marginal counts**: Protected by DP noise
5. **Bin means**: Protected by DP noise

**Privacy Budget Allocation:**
```
Total: (ε, δ)
├─ Binning: (ε/2, δ/2)
│   └─ Noisy bin means
└─ Marginals: (ε/2, δ/2)
    ├─ 1-way: (ε/4, δ/4)
    └─ 2-way: (ε/4, δ/4)
```

**Usage:**
```bash
python run_secure_mpc_pipeline.py \
    --party_files hospital_a.csv hospital_b.csv \
    --output_path synthetic_secure.csv \
    --mpspdz_path /home/mpcuser/MP-SPDZ/ \
    --epsilon 1.0 \
    --delta 1e-5
```

### ⚠️ Insecure Implementation (`run_mpc_pipeline.py`)

**Data Flow:**
```
Party A: raw_data_A.csv ─┐
                          ├─> Combine ─> [Python Binning] ─> [Optional MPC Marginals] ─> Synthesis
Party B: raw_data_B.csv ─┘       ↑              ↑
                           LEAKED!        LEAKED!
```

**Security Issues:**
- ❌ Raw data combined before processing
- ❌ Binning done in Python (reveals raw values)
- ❌ Quantiles computed locally (reveals distribution)
- ❌ Bin means computed without DP noise
- ⚠️ Marginals can optionally use MPC

**What's Leaked:**
1. **Individual values**: Fully revealed during binning
2. **Quantiles**: Exact quantile values revealed
3. **Bin assignments**: All bin assignments revealed
4. **Bin means**: Exact (non-noisy) means revealed
5. **Data distribution**: Full distribution revealed

**Why This Is Dangerous:**
```python
# This code reveals raw data:
df_train, statistic_dict, mean_dict, quantile_dict = discretize_data_local(df)
#                          ↑             ↑              ↑
#                      LEAKED!       LEAKED!        LEAKED!
```

Even if marginals are computed with MPC later, the data has already been leaked!

**When It's Acceptable:**
- Testing with synthetic/fake data
- Development and debugging
- Data is already public
- Single data owner (no multi-party scenario)

## Protocol Usage Comparison

### Secure Implementation

| Step | Protocol | Security | Output |
|------|----------|----------|--------|
| 1. Input | N/A | Secret shares | Each party keeps their data |
| 2. Binning | `ppai_bin_opt.mpc` | ✅ MPC + DP | Noisy bin means (DP) |
| 3. Marginals | `ppai_msr_noisy_final` | ✅ MPC + DP | Noisy marginals (DP) |
| 4. Inference | Public | N/A | Works on noisy stats |
| 5. Synthesis | Public | N/A | Synthetic data |

### Insecure Implementation

| Step | Protocol | Security | Output |
|------|----------|----------|--------|
| 1. Input | N/A | ❌ None | Combined raw data |
| 2. Binning | Python `np.quantile` | ❌ None | Exact bin means |
| 3. Marginals | Optional MPC | ⚠️ Partial | Noisy marginals (DP) |
| 4. Inference | Public | N/A | Works on mixed stats |
| 5. Synthesis | Public | N/A | Synthetic data |

## Code Examples

### ✅ Secure Usage

```python
from models.Private_PGM.model_secure_mpc import SecureMPCPrivatePGM

# Initialize secure model
model = SecureMPCPrivatePGM(
    target_variable='label',
    enable_privacy=True,
    target_epsilon=1.0,
    target_delta=1e-5,
    mpspdz_path='/home/mpcuser/MP-SPDZ/',
    num_parties=2
)

# Train from separate party files (data never combined!)
model.train_from_party_files(
    party_data_files=['party_0.csv', 'party_1.csv'],
    config=config,
    bin_protocol='ppai_bin_opt',
    marginal_protocol='ppai_msr_noisy_final'
)

# Generate secure synthetic data
synthetic = model.generate_continuous(num_rows=1000)
```

### ⚠️ Insecure Usage (Testing Only)

```python
from model import Private_PGM

# Load ALL data (reveals raw data!)
df = pd.read_csv('combined_data.csv')  # ⚠️ LEAKS DATA

# Discretize locally (reveals raw data!)
df_discrete, stats, means, quantiles = discretize_data_local(df)  # ⚠️ LEAKS DATA

# Train (marginals can optionally use MPC)
model = Private_PGM(
    target_variable='label',
    enable_privacy=True,
    target_epsilon=1.0,
    target_delta=1e-5,
    use_mpc=True  # MPC for marginals only (too late!)
)

model.train(df_discrete, config)
synthetic = model.generate(num_rows=1000)
```

## Migration Guide

### Migrating from Insecure to Secure

**Before (Insecure):**
```bash
python run_mpc_pipeline.py \
    --data_path combined_data.csv \
    --use_mpc
```

**After (Secure):**
```bash
# Step 1: Split your data into party files
# party_0.csv: First party's data
# party_1.csv: Second party's data

# Step 2: Run secure pipeline
python run_secure_mpc_pipeline.py \
    --party_files party_0.csv party_1.csv \
    --mpspdz_path /home/mpcuser/MP-SPDZ/ \
    --epsilon 1.0 \
    --delta 1e-5
```

### Creating Party Files

If you have a single dataset and want to simulate multiple parties:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv('full_data.csv')

# Split into party datasets
df_party_0, df_party_1 = train_test_split(
    df, test_size=0.5, random_state=42
)

# Save separately
df_party_0.to_csv('party_0.csv', index=False)
df_party_1.to_csv('party_1.csv', index=False)

# Now each party can keep their file private
```

## Compliance Considerations

### HIPAA / GDPR / Sensitive Data

For sensitive data requiring regulatory compliance:

**✅ MUST USE:** `run_secure_mpc_pipeline.py`

**Rationale:**
- Ensures raw data never leaves custodian systems
- Provides cryptographic security guarantees
- Maintains (ε,δ)-differential privacy
- Supports multi-party collaboration without data sharing

### Non-Sensitive / Public Data

For public datasets or development:

**✅ CAN USE:** `run_mpc_pipeline.py`

**Rationale:**
- Faster execution (no MPC overhead)
- Easier debugging
- Data is already public

## Performance Comparison

| Metric | Secure MPC | Insecure |
|--------|-----------|----------|
| **Execution Time** | Slower (10-100x) | Faster (baseline) |
| **Memory Usage** | Higher (secret sharing) | Lower |
| **Network** | Required between parties | Not required |
| **Setup** | Requires MP-SPDZ | Python only |
| **Security** | Full MPC + DP | DP only (if enabled) |

## Frequently Asked Questions

### Q: When should I use the secure version?

**A:** Always use the secure version (`run_secure_mpc_pipeline.py`) when:
- Working with real sensitive data
- Multiple data custodians are involved
- Regulatory compliance is required
- Production deployment

### Q: Can I use the insecure version for testing?

**A:** Yes, but:
- Only with synthetic/fake data
- Never with real sensitive data
- Clearly mark as "testing only"
- Migrate to secure version before production

### Q: What if I only have one data owner?

**A:** If there's only one data owner and no multi-party requirement:
- The "insecure" version may be acceptable
- But the secure version is still recommended for best practices
- Consider future multi-party scenarios

### Q: Do I need MP-SPDZ for the insecure version?

**A:** No, the insecure version can run without MP-SPDZ. However, it doesn't provide the security guarantees needed for sensitive data.

### Q: Can I mix the two approaches?

**A:** No. Either:
- Use secure end-to-end (binning + marginals in MPC)
- Or accept that you're using the insecure approach

Mixing is dangerous because data leakage at any step compromises the entire pipeline.

## Conclusion

| Priority | Recommendation |
|----------|---------------|
| **High Priority** | Use `run_secure_mpc_pipeline.py` for all sensitive data |
| **Medium Priority** | Migrate existing code to secure version |
| **Low Priority** | Keep insecure version for testing only |

**Remember:** Security is not optional when dealing with sensitive data. Always use the secure implementation for production deployments.
