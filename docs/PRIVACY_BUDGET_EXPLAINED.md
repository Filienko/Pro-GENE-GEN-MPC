# Privacy Budget Allocation Explained

## Why Split ε/2 for Binning and ε/2 for Marginals?

### TL;DR

The 50/50 split is **simple and safe**, but **not necessarily optimal**. You can customize it based on your needs.

---

## Differential Privacy Composition

### The Basic Rule

When you run multiple DP mechanisms on the same dataset, privacy costs **add up**:

```
Mechanism 1: (ε₁, δ₁)-DP
Mechanism 2: (ε₂, δ₂)-DP
───────────────────────────
Combined:    (ε₁+ε₂, δ₁+δ₂)-DP
```

### Why This Matters

Our pipeline has **two DP mechanisms**:

1. **Binning**: Compute noisy bin means
   - Input: Raw continuous data
   - Output: Noisy mean for each bin
   - Privacy cost: (ε_bin, δ_bin)

2. **Marginals**: Compute noisy marginal counts
   - Input: Binned discrete data
   - Output: Noisy 1-way and 2-way marginals
   - Privacy cost: (ε_marg, δ_marg)

**Total privacy cost:** (ε_bin + ε_marg, δ_bin + δ_marg)

If you want total budget of (ε, δ), you need:
```
ε_bin + ε_marg ≤ ε
δ_bin + δ_marg ≤ δ
```

---

## Allocation Strategies

### Strategy 1: Equal Split (Current Default)

```python
ε_binning = ε / 2     # 0.5 if ε=1.0
ε_marginals = ε / 2   # 0.5 if ε=1.0
```

**Pros:**
- ✅ Simple and intuitive
- ✅ Conservative (safe)
- ✅ Easy to explain
- ✅ No tuning needed

**Cons:**
- ⚠️ May not be optimal for utility
- ⚠️ Ignores relative importance of each step
- ⚠️ Ignores sensitivity differences

**When to use:** Default choice for most applications

---

### Strategy 2: Sensitivity-Based Allocation

**Idea:** Allocate budget proportionally to the **number of queries** in each step.

**Analysis:**
- Binning: Computes means for `num_features × 4 bins` values
  - Example: 1000 features × 4 bins = 4,000 values
- Marginals: Computes many more marginal queries
  - 1-way: 1000 features × 4 bins = 4,000 values
  - 2-way: 1000 features × 4 × 5 classes = 20,000 values
  - Total: ~24,000 values

**Better allocation:**
```python
# Marginals need more budget (more queries)
ε_binning = 0.3 * ε    # 30% of budget
ε_marginals = 0.7 * ε  # 70% of budget
```

**Effect on noise:**
- Binning gets MORE noise (less accurate means)
- Marginals get LESS noise (more accurate marginals)
- **Result:** Better model quality (marginals are more important)

**When to use:** When you want to optimize for synthetic data quality

---

### Strategy 3: Utility-Driven Allocation

**Idea:** Give more budget to the **more important** step for your use case.

**Scenario A: Inverse binning is critical**
```python
# You need accurate continuous values from synthetic data
ε_binning = 0.6 * ε    # More budget for accurate bin means
ε_marginals = 0.4 * ε  # Less budget for marginals
```

**Scenario B: Model accuracy is critical**
```python
# You need accurate statistical properties
ε_binning = 0.2 * ε    # Less budget for bin means
ε_marginals = 0.8 * ε  # More budget for accurate marginals
```

**When to use:** When you have specific quality requirements

---

### Strategy 4: Advanced Composition (RDP)

**Current implementation already uses moments accountant** (a form of RDP) **within** each step, but uses basic composition **between** steps.

**Idea:** Use Rényi DP composition for tighter bounds.

With RDP, the composition is:
```
RDP(α): ε(α)₁ + ε(α)₂
Then convert to (ε, δ)-DP: ε = min_α [ε(α) + log(1/δ)/(α-1)]
```

**Benefit:** Can get **lower noise for same privacy** or **stronger privacy for same noise**.

**Implementation status:**
- ✅ Partially implemented (`moments_calibration` uses RDP)
- ⚠️ Could extend to full RDP composition between binning and marginals

**When to use:** Research applications requiring optimal privacy-utility tradeoff

---

## Concrete Example

### Setup
- Total budget: (ε=1.0, δ=1e-5)
- Dataset: 1000 features, 5 classes, 10,000 samples

### Comparison Table

| Strategy | ε_bin | ε_marg | σ_bin | σ_marg | Use Case |
|----------|-------|--------|-------|--------|----------|
| Equal (50/50) | 0.5 | 0.5 | 1.35 | 1.35 | Default, balanced |
| Sensitivity (30/70) | 0.3 | 0.7 | 1.85 | 1.00 | Better model quality |
| Utility-Driven (20/80) | 0.2 | 0.8 | 2.15 | 0.88 | Best marginals |
| Inverse-Focused (60/40) | 0.6 | 0.4 | 1.10 | 1.70 | Better continuous values |

**Key insight:** Higher σ = more noise = less accuracy

---

## How to Customize Allocation

### Option 1: Modify Initialization

Edit `models/Private_PGM/model_secure_mpc.py`:

```python
# Current (equal split)
self.epsilon_binning = target_epsilon / 2
self.epsilon_marginals = target_epsilon / 2

# Custom allocation (30/70)
self.epsilon_binning = target_epsilon * 0.3
self.epsilon_marginals = target_epsilon * 0.7
```

### Option 2: Pass as Parameters (Better)

Modify the `__init__` to accept allocation parameter:

```python
def __init__(self, ..., budget_allocation_ratio=0.5):
    """
    Args:
        budget_allocation_ratio: Fraction of budget for binning (default: 0.5)
    """
    self.epsilon_binning = target_epsilon * budget_allocation_ratio
    self.epsilon_marginals = target_epsilon * (1 - budget_allocation_ratio)
```

Then use:
```python
# Equal split (default)
model = SecureMPCPrivatePGM(..., budget_allocation_ratio=0.5)

# Sensitivity-based
model = SecureMPCPrivatePGM(..., budget_allocation_ratio=0.3)

# Marginal-focused
model = SecureMPCPrivatePGM(..., budget_allocation_ratio=0.2)
```

---

## Advanced: Within-Step Budget Allocation

The marginal computation step itself has **two sub-steps**:
1. 1-way marginals
2. 2-way marginals

**Current:** Also split 50/50 (in the original Private_PGM code)

**Could optimize:**
```python
# 1-way marginals: simpler, need less budget
ε_1way = ε_marginals * 0.3

# 2-way marginals: more complex, need more budget
ε_2way = ε_marginals * 0.7
```

---

## Recommendations

### For Most Users (Default)
```
✅ Use 50/50 split
✅ Simple, safe, well-understood
✅ Good baseline
```

### For Better Utility
```
✅ Use 30/70 split (binning/marginals)
✅ Prioritizes model quality
✅ Better synthetic data accuracy
```

### For Research
```
✅ Implement full RDP composition
✅ Optimize allocation via grid search
✅ Measure utility empirically
```

---

## Implementation Roadmap

### Phase 1: Configurable Allocation (Easy)
- [ ] Add `budget_allocation_ratio` parameter
- [ ] Support custom splits via command-line
- [ ] Document recommended values

### Phase 2: Smart Defaults (Medium)
- [ ] Auto-compute based on dataset size
- [ ] Adjust based on number of features
- [ ] Heuristics for optimal allocation

### Phase 3: Full RDP Composition (Advanced)
- [ ] Extend RDP accountant to multi-step
- [ ] Optimize allocation with RDP bounds
- [ ] Adaptive allocation during execution

---

## Further Reading

1. **Basic Composition:**
   - Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"
   - Section 3.3: "Composition Theorems"

2. **Advanced Composition:**
   - Dwork, Rothblum, Vadhan, "Boosting and Differential Privacy"
   - Theorem 3.20

3. **RDP Composition:**
   - Mironov, "Rényi Differential Privacy"
   - Balle, Barthe, Gaboardi, "Privacy Profiles and Amplification by Subsampling"

4. **Moments Accountant:**
   - Abadi et al., "Deep Learning with Differential Privacy"
   - Used in TensorFlow Privacy

---

## Summary

**Question:** Why ε/2 for binning and ε/2 for marginals?

**Answer:**
- It's a **safe default** that ensures total privacy budget is not exceeded
- It's **not optimal** - you can do better with custom allocation
- **Recommended:** Use 30/70 split (binning/marginals) for better utility
- **Advanced:** Implement full RDP composition for optimal bounds

**Action:** You can easily customize this in the code or we can add it as a parameter!
