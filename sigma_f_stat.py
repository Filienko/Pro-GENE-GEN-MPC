import math

def calculate_f_stat_noise(epsilon_topk, delta, k, data_range, min_variance, N):
    """
    Calculates the Gaussian noise scale (sigma) for Top-K F-statistic selection.
    
    Parameters:
    - epsilon_topk: Total privacy budget allocated for selecting the top k genes.
    - delta: The DP delta parameter (e.g., 1e-5).
    - k: Number of genes to select.
    - data_range: The maximum possible difference between two gene expressions (max - min).
    - min_variance: A clipped minimum value for the within-group variance (to prevent infinite sensitivity).
    - N: Total number of patients.
    """
    
    # 1. Estimate Global Sensitivity of the F-statistic (Delta F)
    # This is a simplified upper bound. A single patient changing their data 
    # affects the numerator by at most ~ (data_range^2) and the denominator 
    # is clamped by min_variance.
    sensitivity_f = (data_range ** 2) / min_variance
    
    # 2. Allocate budget per step (Basic Composition)
    # Since we select k items sequentially, each step uses a fraction of the budget.
    eps_per_step = epsilon_topk / k
    
    # 3. Calculate Gaussian Noise Scale (sigma)
    # Standard formula for Gaussian DP: sigma = (Delta * sqrt(2 * ln(1.25 / delta))) / epsilon
    sigma = (sensitivity_f * math.sqrt(2 * math.log(1.25 / delta))) / eps_per_step
    
    # 4. Scale for MP-SPDZ
    # Your MPC script divides the input by 10,000 to convert to sfix.
    mpc_sigma_arg = int(sigma * 10000)
    
    print(f"Calculated Sigma: {sigma:.4f}")
    print(f"Value to pass to MP-SPDZ args: {mpc_sigma_arg}")
    
    return mpc_sigma_arg

# --- Example Usage ---
# Assuming epsilon=1.0 for the top-k selection, delta=1e-5, k=10, 
# expression values clamped to a range of 5.0, minimum variance clamped to 1.0, and 100 patients.
mpc_arg_7 = calculate_f_stat_noise(
    epsilon_topk=1.0, 
    delta=1e-5, 
    k=10, 
    data_range=5.0, 
    min_variance=1.0, 
    N=100
)

