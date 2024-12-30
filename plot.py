import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re

def read_data_file(file_path):
    """Read and process a single data file."""
    print(f'Reading data file: {file_path}')
    df = pd.read_csv(file_path, delim_whitespace=True)
    # Forward fill the objective values
    full_range = pd.DataFrame({'evaluations': range(1, df['evaluations'].max() + 1)})
    df = pd.merge(full_range, df, on='evaluations', how='left')
    df['raw_y'] = df['raw_y'].fillna(method='ffill')
    
    # Check if the last entry is an improvement, and remove it if not
    if len(df) > 1 and df['raw_y'].iloc[-1] >= df['raw_y'].iloc[-2]:
        df = df.iloc[:-1]
    
    return df


def get_data_files():
    """Get all relevant data files for the first run of each algorithm and dimension."""
    dimensions = [2, 10, 40, 100]
    functions = [1, 8, 12, 15, 21]
    function_names = ['1_Sphere', '8_Rosenbrock', '12_BentCigar', '15_RastriginRotated', '21_Gallagher101']
    algorithms = ['baxus', 'pca_only', 'turbo_only', 'turbo_pca']

    files = []
    for dim in dimensions:
        for algo in algorithms:
            for function_number in functions:
                for function_name in function_names:
                    # Construct the file path
                    file_path = Path(f'./data-{algo}-f{function_number}-dim{dim}/ioh_data/data_f{function_name}/IOHprofiler_f{function_number}_DIM{dim}.dat')
                    print(file_path)
                    if file_path.exists():
                        files.append(file_path)
    return files

def get_algorithm_name(file_path):
    """Extract algorithm name from file path and return a friendly name."""
    path_str = str(file_path)
    if 'turbo_pca' in path_str:
        return 'TuRBO-PCA'
    elif 'turbo_only' in path_str:
        return 'TuRBO'
    elif 'pca_only' in path_str:
        return 'PCA-BO'
    elif 'baxus' in path_str:
        return 'BAXUS'
    return 'Unknown'

def get_dimension(file_path):
    """Extract dimension from file path."""
    dim_match = re.search(r'dim(\d+)', str(file_path))
    return int(dim_match.group(1)) if dim_match else None

def get_function_number(file_path):
    """Extract function number from file path."""
    func_match = re.search(r'_f(\d+)_', str(file_path))
    return int(func_match.group(1)) if func_match else None

def create_visualizations():
    """Create and save visualizations for each dimension."""
    # Colors for different algorithms
    colors = {
        'BAXUS': 'blue',
        'PCA-BO': 'red',
        'TuRBO': 'green',
        'TuRBO-PCA': 'purple'
    }

    # Dimensions and functions
    dimensions = [2, 10, 40, 100]
    functions = [1, 8, 12, 15, 21]

    # Get all data files
    data_files = get_data_files()

    for dim in dimensions:
        # Initialize figure for the current dimension
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=False)
        fig.suptitle(f'{dim}D Comparison of HDBO Algorithms', y=1.05, size=16)

        for func_idx, func in enumerate(functions):
            ax = axes[func_idx]
            ax.set_title(f'Function f{func}')
            ax.set_xlabel('Evaluations')
            if func_idx == 0:
                ax.set_ylabel('Objective Value')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="-", alpha=0.2)

            # Plot data for each algorithm
            for file_path in data_files:
                file_dim = get_dimension(file_path)
                file_func = get_function_number(file_path)

                if file_dim == dim and file_func == func:
                    algo = get_algorithm_name(file_path)
                    df = read_data_file(file_path)
                    ax.plot(df['evaluations'], df['raw_y'], 
                            label=algo, color=colors[algo])

            if func_idx == len(functions) - 1:  # Add legend only to the last subplot
                ax.legend()

        plt.tight_layout()
        plt.savefig(f'dim{dim}results.png')
        plt.close(fig)

# Example usage
if __name__ == "__main__":
    create_visualizations()
