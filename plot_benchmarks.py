import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the expected suffixes and their legend labels
FILE_MAPPINGS = {
    'epsilon': {
        '_epsilon_avg_mal-rep-ring.csv': '3 PC active',
        '_epsilon_avg_ring.csv': '3 PC passive',
        '_pgm_epsilon_avg.csv': 'in-the-clear version'
    },
    'fidelity': {
        '_fidelity_avg_mal-rep-ring_opt.csv': '3 PC active',
        '_fidelity_avg_ring_opt.csv': '3 PC passive', 
        '_pgm_fidelity_avg.csv': 'in-the-clear version'
    }
}

def parse_directory():
    """Scans the directory and groups files by their prefix (dataset name)."""
    files_by_prefix = {}
    for f in os.listdir('.'):
        if not f.endswith('.csv'):
            continue
            
        matched_suffix = None
        category = None
        
        # Determine which mapping category this file belongs to
        for cat, suffixes in FILE_MAPPINGS.items():
            for suffix in suffixes:
                if f.endswith(suffix):
                    matched_suffix = suffix
                    category = cat
                    break
            if matched_suffix:
                break
                
        if matched_suffix:
            prefix = f[:-len(matched_suffix)]
            if prefix not in files_by_prefix:
                files_by_prefix[prefix] = {'epsilon': {}, 'fidelity': {}}
            files_by_prefix[prefix][category][matched_suffix] = f
            
    return files_by_prefix

def filter_data(df):
    """Filters out any rows where epsilon is 100."""
    if 'epsilon' in df.columns:
        df = df[df['epsilon'] != 100.0]
    return df

def plot_epsilon_vs_accuracy(prefix, files_dict):
    """Plots Accuracy vs Epsilon."""
    plt.figure(figsize=(8, 6))
    plotted_anything = False
    min_x = float('inf')
    
    for suffix, label in FILE_MAPPINGS['epsilon'].items():
        if suffix in files_dict:
            file_path = files_dict[suffix]
            df = pd.read_csv(file_path)
            
            # Filter out epsilon 100
            df = filter_data(df)
            
            if 'epsilon' in df.columns and 'accuracy' in df.columns and not df.empty:
                # Sort to ensure lines draw properly from left to right
                df = df.sort_values(by='epsilon')
                plt.plot(df['epsilon'], df['accuracy'], marker='o', label=label)
                
                # Track the minimum x-value for the axis limit
                min_x = min(min_x, df['epsilon'].min())
                plotted_anything = True
            else:
                print(f"Warning: Missing data in {file_path}")

    if plotted_anything:
        plt.xlabel('Epsilon')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Epsilon ({prefix})')
        
        # Set X-axis to start at the first available data point
        if min_x != float('inf'):
            plt.xlim(left=min_x)
            
        # Set Y-axis to start at 0
        plt.ylim(bottom=0)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        out_file = f"{prefix}_epsilon_vs_accuracy.png"
        plt.savefig(out_file)
        print(f"Saved: {out_file}")
    plt.close()

def plot_fidelity_metrics(prefix, files_dict):
    """Plots Execution Time and Compile Time vs Num of Genes."""
    
    # 1. Execution Time Graphs (Separated by protocol)
    for suffix, label in FILE_MAPPINGS['fidelity'].items():
        if suffix in files_dict:
            file_path = files_dict[suffix]
            df = pd.read_csv(file_path)
            
            # Filter out epsilon 100
            df = filter_data(df)
            
            if 'num_features' in df.columns and 'execute_time' in df.columns and not df.empty:
                plt.figure(figsize=(8, 6))
                df = df.sort_values(by='num_features')
                plt.plot(df['num_features'], df['execute_time'], marker='o', label=label, color='tab:blue')
                
                plt.xlabel('Number of Genes (num_features)')
                plt.ylabel('Execution Time (s)')
                plt.title(f'Execution Time vs Number of Genes ({prefix} - {label})')
                
                # Set X-axis to start at the first available data point
                plt.xlim(left=df['num_features'].min())
                
                # Set Y-axis to start at 0
                plt.ylim(bottom=0)
                
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Format label for a clean filename
                safe_label = label.replace(" ", "_").replace("-", "_")
                out_file = f"{prefix}_execution_time_vs_genes_{safe_label}.png"
                plt.savefig(out_file)
                print(f"Saved: {out_file}")
                plt.close()
            else:
                print(f"Info: Skipping execution time for '{label}' ({file_path}) - columns missing.")

    # 2. Compile Time Graph (Still combined)
    plt.figure(figsize=(8, 6))
    plotted_comp = False
    min_x = float('inf')
    
    for suffix, label in FILE_MAPPINGS['fidelity'].items():
        if suffix in files_dict:
            file_path = files_dict[suffix]
            df = pd.read_csv(file_path)
            
            # Filter out epsilon 100
            df = filter_data(df)
            
            if 'num_features' in df.columns and 'compile_time' in df.columns and not df.empty:
                df = df.sort_values(by='num_features')
                plt.plot(df['num_features'], df['compile_time'], marker='o', label=label)
                
                # Track the minimum x-value for the axis limit
                min_x = min(min_x, df['num_features'].min())
                plotted_comp = True
            else:
                print(f"Info: Skipping compile time for '{label}' ({file_path}) - columns missing.")

    if plotted_comp:
        plt.xlabel('Number of Genes (num_features)')
        plt.ylabel('Compile Time (s)')
        plt.title(f'Compile Time vs Number of Genes ({prefix})')
        
        # Set X-axis to start at the first available data point
        if min_x != float('inf'):
            plt.xlim(left=min_x)
            
        # Set Y-axis to start at 0
        plt.ylim(bottom=0)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        out_file = f"{prefix}_compile_time_vs_genes.png"
        plt.savefig(out_file)
        print(f"Saved: {out_file}")
    plt.close()

def main():
    # Group files by their variable prefix
    files_by_prefix = parse_directory()
    
    if not files_by_prefix:
        print("No matching CSV files found in the current directory.")
        return

    # Iterate through each group and generate plots
    for prefix, categories in files_by_prefix.items():
        print(f"\nProcessing prefix group: '{prefix}'")
        plot_epsilon_vs_accuracy(prefix, categories['epsilon'])
        plot_fidelity_metrics(prefix, categories['fidelity'])

if __name__ == "__main__":
    main()
