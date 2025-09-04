import pandas as pd
import numpy as np
import torch
import pickle
import os
import json
import gc
from torch.distributions import Bernoulli
from torch.optim import LBFGS
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
from util import get_HELM_model_benchmark,get_everything_benchmark_raw,get_official_provider_model_benchmark
bundles.iclr2024()
print("running experiment")

def visualize_response_matrix(results, value, filename):
    # Extract the groups labels in the order of the columns
    group_values = results.columns.get_level_values("scenario")

    # Identify the boundaries where the group changes
    boundaries = []
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            boundaries.append(i - 0.5)  # using 0.5 to place the line between columns

    # Visualize the results with a matrix: red is 0, white is -1 and blue is 1
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Calculate midpoints for each group label
    groups_list = list(group_values)
    group_names = []
    group_midpoints = []
    current_group = groups_list[0]
    start_index = 0
    for i, grp in enumerate(groups_list):
        if grp != current_group:
            midpoint = (start_index + i - 1) / 2.0
            group_names.append(current_group)
            group_midpoints.append(midpoint)
            current_group = grp
            start_index = i
    # Add the last group
    midpoint = (start_index + len(groups_list) - 1) / 2.0
    group_names.append(current_group)
    group_midpoints.append(midpoint)

    # Define the minimum spacing between labels (e.g., 100 units)
    min_spacing = 100
    last_label_pos = -float("inf")
    # Plot the matrix
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(value, aspect="auto", cmap=cmap, norm=norm)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="black", linewidth=0.25, linestyle="--", alpha=0.5)
        
        # Add group labels above the matrix, only if they're spaced enough apart
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                ax.text(pos, -5, name, ha='center', va='bottom', rotation=90, fontsize=3)
                last_label_pos = pos

        # Add model labels on the y-axis
        ax.set_yticks(range(len(results.index)))
        ax.set_yticklabels(results.index, fontsize=3)

        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles

def visualize_response_matrix_universal(results, value, filename):
    """
    Visualize response matrix that works with both MultiIndex columns and tuple columns.
    
    Args:
        results: DataFrame with either MultiIndex columns or tuple columns
        value: The values to visualize (usually same as results)
        filename: Output filename for the plot
    """
    
    # Extract scenario information from columns
    if isinstance(results.columns, pd.MultiIndex):
        # Original case: MultiIndex with 'scenario' level
        if "scenario" in results.columns.names:
            group_values = results.columns.get_level_values("scenario")
        else:
            # Fallback: use first level of MultiIndex
            group_values = results.columns.get_level_values(0)
    else:
        # New case: tuple columns where first element is scenario
        if isinstance(results.columns[0], tuple):
            group_values = [col[0] for col in results.columns]
        else:
            # Fallback: use column names directly
            group_values = results.columns.tolist()
    
    # Convert to list for easier processing
    group_values = list(group_values)
    
    # Identify the boundaries where the group changes
    boundaries = []
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            boundaries.append(i - 0.5)  # using 0.5 to place the line between columns

    # Visualize the results with a matrix: red is 0, white is -1 and blue is 1
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Calculate midpoints for each group label
    group_names = []
    group_midpoints = []
    current_group = group_values[0]
    start_index = 0
    
    for i, grp in enumerate(group_values):
        if grp != current_group:
            midpoint = (start_index + i - 1) / 2.0
            group_names.append(current_group)
            group_midpoints.append(midpoint)
            current_group = grp
            start_index = i
    
    # Add the last group
    midpoint = (start_index + len(group_values) - 1) / 2.0
    group_names.append(current_group)
    group_midpoints.append(midpoint)

    # Define the minimum spacing between labels (e.g., 100 units)
    min_spacing = 100
    last_label_pos = -float("inf")
    
    # Convert data to numeric format for plotting
    # Handle boolean data: True -> 1, False -> 0
    # Handle other data types as needed
    if hasattr(value, 'dtypes'):
        # DataFrame case
        plot_data = value.copy()
        for col in plot_data.columns:
            if plot_data[col].dtype == 'object' or plot_data[col].dtype == 'bool':
                # Try to convert booleans first
                if plot_data[col].dtype == 'bool':
                    plot_data[col] = plot_data[col].astype(int)
                else:
                    # For other object types, try to convert to numeric
                    try:
                        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
                    except:
                        # If conversion fails, treat as boolean-like
                        plot_data[col] = plot_data[col].astype(bool).astype(int)
        plot_data = plot_data.values
    else:
        # Array case
        plot_data = np.array(value)
        if plot_data.dtype == 'object' or plot_data.dtype == 'bool':
            if plot_data.dtype == 'bool':
                plot_data = plot_data.astype(int)
            else:
                # Try to convert to numeric, fallback to boolean conversion
                try:
                    plot_data = plot_data.astype(float)
                except (ValueError, TypeError):
                    plot_data = np.array(value, dtype=bool).astype(int)
    
    # Plot the matrix
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(plot_data, aspect="auto", cmap=cmap, norm=norm)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="black", linewidth=0.25, linestyle="--", alpha=0.5)
        
        # Add group labels above the matrix, only if they're spaced enough apart
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                ax.text(pos, -5, name, ha='center', va='bottom', rotation=90, fontsize=3)
                last_label_pos = pos

        # Add model labels on the y-axis
        ax.set_yticks(range(len(results.index)))
        ax.set_yticklabels(results.index, fontsize=3)

        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()


# Example usage:
# For your original dataset:
# visualize_response_matrix_universal(res, res, "results/plot/tmp.png")

# For your new tuple-column dataset:
# visualize_response_matrix_universal(new_res, new_res, "results/plot/new_tmp.png")

# Example usage:
# For your original dataset:
# visualize_response_matrix_universal(res, res, "results/plot/tmp.png")

# For your new tuple-column dataset:
# visualize_response_matrix_universal(new_res, new_res, "results/plot/new_tmp.png")

# res = get_HELM_model_benchmark()
# visualize_response_matrix_universal(res, res, "results/plot/HELM_data_2.png")

res = get_everything_benchmark_raw()
res_numeric = res.astype(float)
# breakpoint()
visualize_response_matrix_universal(res_numeric, res_numeric, "results/plot/everything_data_no_arc.png")

# res = get_official_provider_model_benchmark()

# data_clean = res.drop('model_name', axis=1)

# # Find binary columns
# binary_cols = []
# for col in data_clean.columns:
#     unique_vals = data_clean[col].dropna().unique()
#     if all(val in [0.0, 1.0] for val in unique_vals):
#         binary_cols.append(col)
# # Keep only binary columns
# data_binary = data_clean[binary_cols]

# # Convert ALL columns to float64 (including boolean columns)
# data_binary = data_binary.astype('float64')
# res_numeric = data_binary.astype(float)
# # breakpoint()
# visualize_response_matrix_universal(res_numeric, res_numeric, "results/plot/official_provider_data.png")


