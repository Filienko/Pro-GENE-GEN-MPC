"""
Privacy Budget Allocation Analysis for Secure MPC Private-PGM

This document analyzes different strategies for allocating privacy budget
between binning and marginal computation steps.
"""

import numpy as np
from utils.rdp_accountant import compute_rdp, get_privacy_spent


def analyze_budget_allocation():
    """
    Compare different privacy budget allocation strategies
    """

    print("="*80)
    print("PRIVACY BUDGET ALLOCATION ANALYSIS")
    print("="*80)

    # Total budget
    total_epsilon = 1.0
    total_delta = 1e-5

    print(f"\nTotal Privacy Budget: (ε={total_epsilon}, δ={total_delta})")
    print("\n" + "-"*80)

    # Strategy 1: Equal Split (Current)
    print("\nSTRATEGY 1: Equal Split (50/50) [CURRENT]")
    print("-"*80)
    eps_bin_1 = total_epsilon / 2
    eps_marg_1 = total_epsilon / 2
    print(f"Binning:   ε={eps_bin_1}, δ={total_delta/2}")
    print(f"Marginals: ε={eps_marg_1}, δ={total_delta/2}")
    print(f"Total:     ε={eps_bin_1 + eps_marg_1}, δ={total_delta/2 + total_delta/2}")
    print(f"✓ Simple, conservative, easy to understand")
    print(f"⚠ May not be optimal for utility")

    # Strategy 2: Sensitivity-Based
    print("\n\nSTRATEGY 2: Sensitivity-Based Allocation")
    print("-"*80)
    # Binning: Computes means for ~1000 features × 4 bins = 4000 values
    # Marginals: Computes many more marginal queries
    # Allocate based on relative query counts
    eps_bin_2 = 0.3 * total_epsilon  # Less budget (fewer queries)
    eps_marg_2 = 0.7 * total_epsilon  # More budget (more queries)
    print(f"Binning:   ε={eps_bin_2}, δ={0.3*total_delta}")
    print(f"Marginals: ε={eps_marg_2}, δ={0.7*total_delta}")
    print(f"Total:     ε={eps_bin_2 + eps_marg_2}, δ={0.3*total_delta + 0.7*total_delta}")
    print(f"✓ Optimized for number of queries")
    print(f"✓ Better utility for marginals")
    print(f"⚠ Requires careful sensitivity analysis")

    # Strategy 3: Utility-Driven
    print("\n\nSTRATEGY 3: Utility-Driven Allocation")
    print("-"*80)
    # If marginals are more important for model quality
    eps_bin_3 = 0.2 * total_epsilon
    eps_marg_3 = 0.8 * total_epsilon
    print(f"Binning:   ε={eps_bin_3}, δ={0.2*total_delta}")
    print(f"Marginals: ε={eps_marg_3}, δ={0.8*total_delta}")
    print(f"Total:     ε={eps_bin_3 + eps_marg_3}, δ={0.2*total_delta + 0.8*total_delta}")
    print(f"✓ Maximizes utility for critical component")
    print(f"✓ Better synthetic data quality")
    print(f"⚠ May degrade bin mean quality")

    # Strategy 4: Advanced Composition (RDP)
    print("\n\nSTRATEGY 4: RDP Composition [ADVANCED]")
    print("-"*80)
    print("Using Rényi Differential Privacy for tighter composition")

    # With RDP composition, we can often get better bounds
    # This requires using the RDP accountant throughout
    orders = range(2, 100)

    # Simulate: With RDP, same noise could give better (ε, δ)
    # Or same (ε, δ) could allow less noise (better utility)

    print("✓ Tighter composition bounds")
    print("✓ Better utility for same privacy")
    print("✓ Already partially implemented (moments_calibration)")
    print("⚠ More complex to implement end-to-end")

    # Strategy 5: Adaptive Allocation
    print("\n\nSTRATEGY 5: Adaptive Allocation [FUTURE]")
    print("-"*80)
    print("Allocate budget dynamically based on data characteristics")
    print(f"Example: If data is already discrete → 0 for binning, {total_epsilon} for marginals")
    print("✓ Maximally efficient budget usage")
    print("✓ Best possible utility")
    print("⚠ Requires runtime decisions")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
    Current Status: Using Strategy 1 (Equal Split)

    For Most Users:
    - Keep Strategy 1 (50/50 split)
    - Simple, safe, and easy to explain

    For Advanced Users:
    - Consider Strategy 2 (Sensitivity-Based) for better utility
    - Requires profiling your specific dataset

    For Researchers:
    - Implement Strategy 4 (RDP Composition) for optimal bounds
    - Requires deeper integration with RDP accountant

    Future Work:
    - Implement adaptive allocation based on data properties
    - Add utility optimization with privacy constraints
    - Support user-specified budget allocation
    """)


def compare_noise_levels():
    """
    Compare noise levels under different allocations
    """
    print("\n" + "="*80)
    print("NOISE LEVEL COMPARISON")
    print("="*80)

    total_epsilon = 1.0
    total_delta = 1e-5

    # For Gaussian mechanism with moments accountant
    from scipy import optimize
    orders = range(2, 4096)

    def calibrate_sigma(eps, delta):
        def obj(sigma):
            rdp = compute_rdp(1.0, sigma, 1, orders)
            privacy = get_privacy_spent(orders, rdp, delta=delta)
            return privacy[0] - eps + 1e-8

        low, high = 1.0, 1.0
        while obj(low) < 0:
            low /= 2.0
        while obj(high) > 0:
            high *= 2.0
        return optimize.bisect(obj, low, high)

    print("\nFor total budget (ε=1.0, δ=1e-5):")
    print("-"*80)

    # Strategy 1: 50/50
    sigma_bin_1 = calibrate_sigma(0.5, total_delta/2)
    sigma_marg_1 = calibrate_sigma(0.5, total_delta/2)
    print(f"\nStrategy 1 (50/50):")
    print(f"  Binning sigma:   {sigma_bin_1:.4f}")
    print(f"  Marginals sigma: {sigma_marg_1:.4f}")

    # Strategy 2: 30/70
    sigma_bin_2 = calibrate_sigma(0.3, total_delta*0.3)
    sigma_marg_2 = calibrate_sigma(0.7, total_delta*0.7)
    print(f"\nStrategy 2 (30/70):")
    print(f"  Binning sigma:   {sigma_bin_2:.4f} ({(sigma_bin_2/sigma_bin_1 - 1)*100:+.1f}%)")
    print(f"  Marginals sigma: {sigma_marg_2:.4f} ({(sigma_marg_2/sigma_marg_1 - 1)*100:+.1f}%)")

    # Strategy 3: 20/80
    sigma_bin_3 = calibrate_sigma(0.2, total_delta*0.2)
    sigma_marg_3 = calibrate_sigma(0.8, total_delta*0.8)
    print(f"\nStrategy 3 (20/80):")
    print(f"  Binning sigma:   {sigma_bin_3:.4f} ({(sigma_bin_3/sigma_bin_1 - 1)*100:+.1f}%)")
    print(f"  Marginals sigma: {sigma_marg_3:.4f} ({(sigma_marg_3/sigma_marg_1 - 1)*100:+.1f}%)")

    print("\nInterpretation:")
    print("- Higher sigma = more noise = less accuracy")
    print("- Lower sigma = less noise = better accuracy")
    print("- Trade-off between binning and marginal accuracy")


if __name__ == "__main__":
    analyze_budget_allocation()
    compare_noise_levels()
