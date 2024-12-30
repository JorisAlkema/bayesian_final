#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import re

def find_dat_files(root_dir):
    pattern = os.path.join(root_dir, "data-*-f*-dim*", "ioh_data-*", "*", "*.dat")
    dat_files = glob.glob(pattern, recursive=True)
    
    parsed_files = []
    run_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for file_path in dat_files:
        parts = file_path.split(os.sep)
        data_folder = parts[-4]
        ioh_data_folder = parts[-3]
        run_folder = parts[-2]
        dat_file = parts[-1]

        match = re.match(r"data-(?P<method>.+)-f(?P<fid>\d+)-dim(?P<dim>\d+)", data_folder)
        if not match:
            print(f"Warning: Data folder name '{data_folder}' does not match expected pattern. Skipping file {file_path}.")
            continue
        method = match.group('method')
        fid = int(match.group('fid'))
        dim = int(match.group('dim'))

        instance_match = re.match(r"ioh_data(?:-(?P<instance>\d+))?", ioh_data_folder)
        instance = int(instance_match.group('instance')) if instance_match.group('instance') else 0

        run_match = re.match(r"data_f(?P<fid>\d+)_(?P<function_name>.+)", run_folder)
        if not run_match:
            print(f"Warning: Run folder name '{run_folder}' does not match expected pattern. Skipping file {file_path}.")
            continue
        run_fid = int(run_match.group('fid'))
        function_name = run_match.group('function_name')

        run_counter[dim][fid][method] += 1
        repetition = run_counter[dim][fid][method]

        expected_dat_filename = f"IOHprofiler_f{fid}_DIM{dim}.dat"
        parsed_files.append((method, fid, dim, instance, repetition, file_path))
    return parsed_files

def read_dat_file(file_path, problematic_files):
    try:
        df = pd.read_csv(file_path, sep='\s+', comment='#')
        if df.empty:
            print(f"Warning: {file_path} is empty or contains only comments. Skipping this file.")
            problematic_files.append(file_path)
            return None, None
        if df.shape[1] < 2:
            print(f"Warning: {file_path} does not have at least two columns. Skipping this file.")
            problematic_files.append(file_path)
            return None, None
        evals = df.iloc[:, 0].values
        raw_y = df.iloc[:, 1].values
        return evals, raw_y
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} has no data to parse. Skipping this file.")
        problematic_files.append(file_path)
        return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        problematic_files.append(file_path)
        return None, None

def compute_best_so_far(raw_y):
    return np.minimum.accumulate(raw_y)

def aggregate_runs(parsed_files, budget_multiplier=10):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    max_evals_per_dim = defaultdict(int)
    problematic_files = []

    for entry in parsed_files:
        _, fid, dim, _, _, _ = entry
        max_evals_per_dim[dim] = max(max_evals_per_dim[dim], budget_multiplier * dim)

    for method, fid, dim, instance, repetition, file_path in tqdm(parsed_files, desc="Aggregating runs"):
        evals, raw_y = read_dat_file(file_path, problematic_files)
        if evals is None or raw_y is None:
            continue  # Skip files that couldn't be read properly

        best_f = compute_best_so_far(raw_y)

        max_evals = budget_multiplier * dim

        if len(best_f) < max_evals:
            padding = max_evals - len(best_f)
            best_f = np.concatenate([best_f, np.full(padding, best_f[-1])])
        elif len(best_f) > max_evals:
            best_f = best_f[:max_evals]

        data[dim][fid][method].append(best_f)
    
    if problematic_files:
        print("\nList of problematic files:")
        for pf in problematic_files:
            print(pf)
    
    return data, max_evals_per_dim

def compute_average_best_f(data):
    avg_data = defaultdict(lambda: defaultdict(dict))
    
    for dim in data:
        for fid in data[dim]:
            for method in data[dim][fid]:
                runs = data[dim][fid][method]
                if not runs:
                    continue  # Skip if there are no runs
                runs_array = np.array(runs)
                avg_best_f = np.mean(runs_array, axis=0)
                avg_data[dim][fid][method] = avg_best_f
    
    return avg_data

def plot_aggregated_performance(avg_data, max_evals_per_dim, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for dim in sorted(avg_data.keys()):
        fids = sorted(avg_data[dim].keys())
        num_fids = len(fids)
        
        cols = 3
        rows = int(np.ceil(num_fids / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)
        fig.suptitle(f"Performance Comparison for Dimension {dim}", fontsize=16)
        
        for idx, fid in enumerate(fids):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]
            
            methods = sorted(avg_data[dim][fid].keys())
            for method in methods:
                avg_best_f = avg_data[dim][fid][method]
                evals = np.arange(1, len(avg_best_f) + 1)
                ax.plot(evals, avg_best_f, label=method)
            
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Best f so far")
            ax.set_xscale('log')
            ax.set_yscale('log')  # Logarithmic scale for y-axis
            ax.set_title(f"Function f{fid}")
            ax.legend()
            ax.grid(True, which="both", ls="--", linewidth=0.5)
        
        total_subplots = rows * cols
        if num_fids < total_subplots:
            for idx in range(num_fids, total_subplots):
                row = idx // cols
                col = idx % cols
                fig.delaxes(axes[row][col])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_filename = f"aggregated_performance_dim{dim}"
        save_path = os.path.join(output_dir, f"{plot_filename}.png")
        plt.savefig(save_path)
        
        plt.close()

def main():
    root_dir = '.'
    output_dir = 'aggregated_plots'
    
    parsed_files = find_dat_files(root_dir)
    
    if not parsed_files:
        print("No .dat files found. Please check the directory structure and file paths.")
        return
    
    data, max_evals_per_dim = aggregate_runs(parsed_files)
    
    if not data:
        print("No valid data to aggregate. Exiting.")
        return
    
    avg_data = compute_average_best_f(data)
    
    if not avg_data:
        print("No average data computed. Exiting.")
        return
    
    plot_aggregated_performance(avg_data, max_evals_per_dim, output_dir)
    print(f"Plots have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
