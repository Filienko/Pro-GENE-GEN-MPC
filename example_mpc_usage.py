"""
Example: Using Private-PGM with MPC for Secure Multi-Party Computation

This script demonstrates how to use the updated Private_PGM class with MPC
to securely compute marginals across multiple data custodians.
"""

import pandas as pd
import numpy as np
from model import Private_PGM

# Example 1: Standard mode (without MPC) - existing functionality
def example_standard_mode():
    """
    Use Private-PGM in standard mode (existing functionality)
    """
    print("\n" + "=" * 80)
    print("Example 1: Standard Mode (No MPC)")
    print("=" * 80)

    # Load your data
    # Replace with actual data loading
    # data = pd.read_csv('your_data.csv')

    # Create dummy data for demonstration
    n_samples = 1000
    n_features = 10
    n_classes = 5

    data = pd.DataFrame(
        np.random.randint(0, 4, size=(n_samples, n_features)),
        columns=[f'gene_{i}' for i in range(n_features)]
    )
    data['label'] = np.random.randint(0, n_classes, size=n_samples)

    # Define domain configuration
    config = {f'gene_{i}': 4 for i in range(n_features)}
    config['label'] = n_classes

    # Initialize Private PGM in standard mode
    model = Private_PGM(
        target_variable='label',
        enable_privacy=True,
        target_epsilon=1.0,
        target_delta=1e-5,
        use_mpc=False  # Standard mode
    )

    # Train the model
    print("Training model in standard mode...")
    model.train(data, config, num_iters=1000)

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = model.generate(num_rows=500)
    print(f"Generated synthetic data shape: {synthetic_data.shape}")

    return model


# Example 2: MPC mode - NEW functionality
def example_mpc_mode():
    """
    Use Private-PGM with MPC for secure multi-party computation
    """
    print("\n" + "=" * 80)
    print("Example 2: MPC Mode (Secure Multi-Party Computation)")
    print("=" * 80)

    # Load your data
    # In a real MPC scenario, each party would have their own data
    # and it would never be combined in plaintext

    # Create dummy data for demonstration
    n_samples = 1000
    n_features = 10
    n_classes = 5

    data = pd.DataFrame(
        np.random.randint(0, 4, size=(n_samples, n_features)),
        columns=[f'gene_{i}' for i in range(n_features)]
    )
    data['label'] = np.random.randint(0, n_classes, size=n_samples)

    # Define domain configuration
    config = {f'gene_{i}': 4 for i in range(n_features)}
    config['label'] = n_classes

    # Initialize Private PGM in MPC mode
    model = Private_PGM(
        target_variable='label',
        enable_privacy=True,
        target_epsilon=1.0,
        target_delta=1e-5,
        use_mpc=True,  # Enable MPC mode
        mpspdz_path='/home/mpcuser/MP-SPDZ/',  # Path to MP-SPDZ installation
        mpc_protocol='ring'  # MPC protocol to use
    )

    # Train the model with MPC
    # The marginals will be computed securely using the MPC protocol
    print("Training model with MPC...")
    print("Note: This requires MP-SPDZ to be installed and configured")

    try:
        model.train(
            data,
            config,
            num_iters=1000,
            mpc_protocol_file='ppai_msr_noisy_final'  # MPC protocol for marginal computation
        )

        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = model.generate(num_rows=500)
        print(f"Generated synthetic data shape: {synthetic_data.shape}")

        return model

    except Exception as e:
        print(f"MPC execution failed: {e}")
        print("Make sure MP-SPDZ is properly installed and configured")
        return None


# Example 3: Custom MPC protocol
def example_custom_mpc_protocol():
    """
    Use Private-PGM with a custom MPC protocol file
    """
    print("\n" + "=" * 80)
    print("Example 3: Custom MPC Protocol")
    print("=" * 80)

    # Create dummy data
    n_samples = 1000
    n_features = 10
    n_classes = 5

    data = pd.DataFrame(
        np.random.randint(0, 4, size=(n_samples, n_features)),
        columns=[f'gene_{i}' for i in range(n_features)]
    )
    data['label'] = np.random.randint(0, n_classes, size=n_samples)

    config = {f'gene_{i}': 4 for i in range(n_features)}
    config['label'] = n_classes

    # Initialize with MPC
    model = Private_PGM(
        target_variable='label',
        enable_privacy=True,
        target_epsilon=1.0,
        target_delta=1e-5,
        use_mpc=True,
        mpspdz_path='/home/mpcuser/MP-SPDZ/',
        mpc_protocol='ring'
    )

    # Use a custom MPC protocol file
    custom_protocol = 'ppai_msr'  # or any other .mpc file you have

    print(f"Using custom MPC protocol: {custom_protocol}")

    try:
        model.train(
            data,
            config,
            num_iters=1000,
            mpc_protocol_file=custom_protocol
        )

        synthetic_data = model.generate(num_rows=500)
        print(f"Generated synthetic data shape: {synthetic_data.shape}")

        return model

    except Exception as e:
        print(f"Training failed: {e}")
        return None


if __name__ == "__main__":
    print("Private-PGM with MPC Examples")
    print("=" * 80)

    # Run standard mode example
    model_standard = example_standard_mode()

    # Run MPC mode example
    # Uncomment when MP-SPDZ is installed and configured
    # model_mpc = example_mpc_mode()

    # Run custom protocol example
    # Uncomment when MP-SPDZ is installed and configured
    # model_custom = example_custom_mpc_protocol()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
