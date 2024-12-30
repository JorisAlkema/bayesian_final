###this code was generated with some help from chatgpt - as we did not find it critical to write this code from scratch
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import re

def find_dat_files():
    pattern = os.path.join("data-*-f*-dim*", "ioh_data-*/data_f*_*", "IOHprofiler_f*_DIM*.dat")
    dat_files = glob.glob(pattern, recursive=True)

    parsed_files = []
    run_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) 

    for file_path in dat_files:
        parts = file_path.split(os.sep)
        try:
            data_folder = parts[-4]
            ioh_data_folder = parts[-3]
            run_folder = parts[-2]
            dat_file = parts[-1]
        except IndexError:
            continue

        match = re.match(r"data-(?P<method>.+)-f(?P<fid>\d+)-dim(?P<dim>\d+)", data_folder)
        if not match:
            continue
        method = match.group('method')
        fid = int(match.group('fid'))
        dim = int(match.group('dim'))

        instance_match = re.match(r"ioh_data(?:-(?P<instance>\d+))?", ioh_data_folder)
        if not instance_match:
            continue
        instance = int(instance_match.group('instance')) if instance_match.group('instance') else 1

        run_match = re.match(r"data_f(?P<fid_run>\d+)_.*", run_folder)
        if not run_match:
            continue
        fid_run = int(run_match.group('fid_run'))

        if fid_run != fid:
            continue

        run_counter[dim][fid][method] += 1
        repetition = run_counter[dim][fid][method]

        parsed_files.append((method, fid, dim, instance, repetition, file_path))

    return parsed_files

def read_dat_file(file_path, problematic_files):
    try:
        df = pd.read_csv(file_path, sep='\s+', comment='#')
        if df.empty or df.shape[1] < 2:
            problematic_files.append(file_path)
            return None, None
        evals = df.iloc[:, 0].values
        raw_y = df.iloc[:, 1].values
        return evals, raw_y
    except pd.errors.EmptyDataError:
        problematic_files.append(file_path)
        return None, None
    except Exception:
        problematic_files.append(file_path)
        return None, None

def compute_best_so_far(raw_y):
    return np.minimum.accumulate(raw_y)

def map_best_f_to_common_evals(evals, best_f, common_evals):
    if len(evals) == 0:
        return np.full_like(common_evals, np.inf, dtype=np.float64)

    sorted_idx = np.argsort(evals)
    evals_sorted = evals[sorted_idx]
    best_f_sorted = best_f[sorted_idx]

    mapped_best_f = np.full_like(common_evals, np.inf, dtype=np.float64)

    current_best = np.inf
    run_idx = 0
    for i, ce in enumerate(common_evals):
        while run_idx < len(evals_sorted) and evals_sorted[run_idx] <= ce:
            current_best = min(current_best, best_f_sorted[run_idx])
            run_idx += 1
        mapped_best_f[i] = current_best if current_best != np.inf else np.nan

    mask = np.isnan(mapped_best_f)
    if np.any(mask):
        if np.any(~np.isnan(mapped_best_f)):
            first_valid = np.nanmin(mapped_best_f[~mask])
            mapped_best_f = np.where(mask, first_valid, mapped_best_f)
        else:
            mapped_best_f = np.where(mask, np.inf, mapped_best_f)

    return mapped_best_f

def aggregate_runs(parsed_files, budget_multiplier=10):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    max_evals_per_dim = defaultdict(int)
    problematic_files = []

    for entry in parsed_files:
        _, fid, dim, _, _, _ = entry
        max_evals_per_dim[dim] = max(max_evals_per_dim[dim], budget_multiplier * dim)

    common_evals_per_dim = {dim: np.arange(1, max_evals_per_dim[dim] + 1) for dim in max_evals_per_dim}

    for method, fid, dim, _, _, file_path in tqdm(parsed_files, desc="Aggregating runs"):
        evals, raw_y = read_dat_file(file_path, problematic_files)
        if evals is None or raw_y is None:
            continue

        sorted_idx = np.argsort(evals)
        evals_sorted = evals[sorted_idx]
        raw_y_sorted = raw_y[sorted_idx]

        best_f = compute_best_so_far(raw_y_sorted)
        common_evals = common_evals_per_dim[dim]
        mapped_best_f = map_best_f_to_common_evals(evals_sorted, best_f, common_evals)

        data[dim][fid][method].append(mapped_best_f)

    return data, common_evals_per_dim

def compute_average_best_f(data):
    avg_data = defaultdict(lambda: defaultdict(dict))

    for dim in data:
        for fid in data[dim]:
            for method in data[dim][fid]:
                runs = data[dim][fid][method]
                if not runs:
                    continue
                runs_array = np.array(runs)
                avg_best_f = np.mean(runs_array, axis=0)
                avg_data[dim][fid][method] = avg_best_f

    return avg_data

def plot_aggregated_performance(avg_data, common_evals_per_dim, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for dim in sorted(avg_data.keys()):
        fids = sorted(avg_data[dim].keys())
        num_fids = len(fids)

        cols = 3
        rows = int(np.ceil(num_fids / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)
        fig.suptitle(f"Performance Comparison for Dimension {dim}", fontsize=16)

        common_evals = common_evals_per_dim[dim]
        all_methods = set()

        # First pass to collect all methods for the legend
        for fid in fids:
            methods = sorted(avg_data[dim][fid].keys())
            all_methods.update(methods)
        
        all_methods = sorted(all_methods)
        method_colors = {method: None for method in all_methods}  # To store colors for consistency

        for idx, fid in enumerate(fids):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            methods = sorted(avg_data[dim][fid].keys())
            for method in methods:
                avg_best_f = avg_data[dim][fid][method]
                line, = ax.plot(common_evals, avg_best_f, label=method)
                if method_colors[method] is None:
                    method_colors[method] = line.get_color()

            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Best f so far")
            ax.set_xscale('linear')  
            ax.set_yscale('log')     
            ax.set_title(f"Function f{fid}")
            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

        # Remove unused subplots
        total_subplots = rows * cols
        if num_fids < total_subplots:
            for idx in range(num_fids, total_subplots):
                row = idx // cols
                col = idx % cols
                fig.delaxes(axes[row][col])

        # Create a centralized legend
        handles = [plt.Line2D([0], [0], color=method_colors[method], label=method) for method in all_methods]
        fig.legend(handles=handles, loc='lower center', ncol=len(all_methods), fontsize='small')

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        plot_filename = f"aggregated_performance_dim{dim}"
        save_path = os.path.join(output_dir, f"{plot_filename}.png")
        plt.savefig(save_path)

        plt.close()

def main():
    output_dir = 'aggregated_plots'

    parsed_files = find_dat_files()

    data, common_evals_per_dim = aggregate_runs(parsed_files)

    avg_data = compute_average_best_f(data)

    plot_aggregated_performance(avg_data, common_evals_per_dim, output_dir)

if __name__ == "__main__":
    main()
