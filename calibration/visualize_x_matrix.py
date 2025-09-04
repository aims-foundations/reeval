import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
import warnings
from util import get_time_diff_matrix_for_benchmark
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
import warnings
warnings.filterwarnings('ignore')

def sort_dataframe_by_mean_values_within_scenario(df, scenarios):
    """
    Sort DataFrame by:
    1. Column means within each scenario (highest mean left within each scenario group)
    2. Row means across all scenarios (highest mean at top)
    
    Args:
        df: DataFrame with continuous values
        scenarios: List of scenario names for each column
    
    Returns:
        Sorted DataFrame, sorted scenarios list
    """
    # Convert to DataFrame if it's a tensor
    if isinstance(df, torch.Tensor):
        df = pd.DataFrame(df.numpy())
    
    # Ensure scenarios is a list
    if isinstance(scenarios, torch.Tensor):
        scenarios = scenarios.tolist()
    elif hasattr(scenarios, 'tolist'):
        scenarios = scenarios.tolist()
    
    # Group columns by scenario and sort within each group
    scenario_groups = {}
    for i, scenario in enumerate(scenarios):
        if scenario not in scenario_groups:
            scenario_groups[scenario] = []
        scenario_groups[scenario].append(i)
    
    # Sort columns within each scenario by mean (descending = highest mean first)
    sorted_columns = []
    sorted_scenarios = []
    for scenario in scenario_groups:
        scenario_col_indices = scenario_groups[scenario]
        scenario_df = df.iloc[:, scenario_col_indices]
        
        # Calculate mean for each column in this scenario
        column_means = scenario_df.mean(axis=0)
        
        # Sort by mean (descending = highest mean first)
        sorted_indices = column_means.sort_values(ascending=False).index
        
        # Convert back to original column indices

        base = scenario_col_indices[0]
        original_indices = [scenario_col_indices[idx-base] for idx in sorted_indices]
        sorted_columns.extend(original_indices)
        sorted_scenarios.extend([scenario] * len(original_indices))
    
    # Sort rows by mean (descending = highest mean at top)
    row_means = df.mean(axis=1)
    sorted_rows = row_means.sort_values(ascending=False).index
    
    # Apply both sorts
    df_sorted = df.iloc[sorted_rows, sorted_columns]
    
    return df_sorted, sorted_scenarios, sorted_rows

def visualize_continuous_matrix(df, scenarios, model_names, filename, title="Model Performance Matrix"):
    """
    Visualize continuous response matrix with gradient coloring.
    
    Args:
        df: DataFrame with continuous values
        scenarios: List of scenario names for each column
        model_names: List of model names for rows
        filename: Output filename for the plot
        title: Title for the plot
    """
    
    # Convert data to numpy array for plotting
    plot_data = df.values if hasattr(df, 'values') else np.array(df)
    
    # Create colormap for continuous values
    # Use a gradient from blue (low) to red (high)
    cmap = plt.cm.viridis  # You can change this to other colormaps like 'plasma', 'inferno', 'RdYlBu', etc.
    
    # Calculate value range for normalization
    vmin = np.nanmin(plot_data)
    vmax = np.nanmax(plot_data)
    
    # Identify the boundaries where the scenario changes
    boundaries = []
    for i in range(1, len(scenarios)):
        if scenarios[i] != scenarios[i - 1]:
            boundaries.append(i - 0.5)

    # Calculate midpoints for each group label
    group_names = []
    group_midpoints = []
    current_group = scenarios[0]
    start_index = 0
    
    for i, grp in enumerate(scenarios):
        if grp != current_group:
            midpoint = (start_index + i - 1) / 2.0
            group_names.append(current_group)
            group_midpoints.append(midpoint)
            current_group = grp
            start_index = i
    
    # Add the last group
    midpoint = (start_index + len(scenarios) - 1) / 2.0
    group_names.append(current_group)
    group_midpoints.append(midpoint)

    # Define the minimum spacing between labels
    min_spacing = max(50, len(scenarios) // 20)  # Adaptive spacing
    last_label_pos = -float("inf")
    
    # Plot the matrix
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(max(20, len(scenarios)//100), max(10, len(model_names)//50)))
        
        # Create the heatmap
        im = ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="white", linewidth=0.5, linestyle="-", alpha=0.8)
        
        # Add group labels above the matrix
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                ax.text(pos, -len(model_names)*0.02, name, ha='center', va='bottom', 
                       rotation=45, fontsize=max(6, min(10, 100//len(group_names))))
                last_label_pos = pos

        # Add model labels on the y-axis (sample every nth model if too many)
        if len(model_names) > 100:
            step = len(model_names) // 50  # Show ~50 labels max
            tick_indices = range(0, len(model_names), step)
            tick_labels = [model_names[i] if i < len(model_names) else "" for i in tick_indices]
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels, fontsize=6)
        else:
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=max(3, 12-len(model_names)//50))

        # Remove x-axis ticks (too many columns)
        ax.set_xticks([])
        
        # Add title
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=8)
        
        # Add value range info in the plot
        ax.text(0.02, 0.98, f'Range: [{vmin:.1f}, {vmax:.1f}]', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # plt.tight_layout()
        # plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor='white')
        # plt.close()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95)
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        
    print(f"Visualization saved to: {filename}")
    print(f"Data shape: {plot_data.shape}")
    print(f"Value range: [{vmin:.2f}, {vmax:.2f}]")
    print(f"Number of scenarios: {len(set(scenarios))}")
    print(f"Number of models: {len(model_names)}")

def visualize_meta_orig_x(meta, output_filename="meta_orig_x_visualization.png"):
    """
    Main function to visualize meta['orig_x'] data.
    
    Args:
        meta: Dictionary containing 'orig_x', 'model_names', and 'cat1'
        output_filename: Name for the output file
    """
    
    # Extract data
    orig_x = meta['orig_x']
    model_names = meta['model_names']
    cat1 = meta['cat1']
    
    # Convert tensor to numpy if needed
    if isinstance(orig_x, torch.Tensor):
        orig_x_np = orig_x.detach().cpu().numpy()
    else:
        orig_x_np = np.array(orig_x)
    
    # Convert to DataFrame
    df = pd.DataFrame(orig_x_np)
    
    # Convert model names to list if needed
    if hasattr(model_names, 'tolist'):
        model_names = model_names.tolist()
    elif isinstance(model_names, torch.Tensor):
        model_names = model_names.detach().cpu().numpy().tolist()
    
    print("Original data shape:", df.shape)
    print("Sample of original data:")
    print(df.iloc[:5, :5])
    
    # Sort the data
    print("Sorting data by mean values...")
    df_sorted, scenarios_sorted, sorted_row_indices = sort_dataframe_by_mean_values_within_scenario(df, cat1)
    
    # Get sorted model names
    model_names_sorted = [model_names[i] for i in sorted_row_indices]
    df_sorted = df
    
    print("Data sorted successfully!")
    print("Sample of sorted data:")
    print(df_sorted.iloc[:5, :5])
    
    # Create visualization
    print("Creating visualization...")
    visualize_continuous_matrix(
        df_sorted, 
        scenarios_sorted, 
        model_names_sorted, 
        output_filename,
        title="Model Performance Matrix (Sorted by Mean Values)"
    )

# Example usage:
# visualize_meta_orig_x(meta, "meta_orig_x_heatmap.png")

# If you want to customize the colormap, you can modify the function:
def visualize_meta_orig_x_custom_colormap(meta, output_filename="meta_orig_x_custom.png", colormap='RdYlBu_r'):
    """
    Visualize with custom colormap options.
    Popular colormaps for continuous data:
    - 'viridis': Purple to yellow (default)
    - 'plasma': Purple to pink/yellow  
    - 'inferno': Black to yellow
    - 'RdYlBu_r': Red-Yellow-Blue (reversed)
    - 'coolwarm': Blue to red
    - 'seismic': Blue-white-red
    """
    
    # Extract and process data (same as above)
    orig_x = meta['orig_x']
    model_names = meta['model_names']
    cat1 = meta['cat1']
    
    if isinstance(orig_x, torch.Tensor):
        orig_x_np = orig_x.detach().cpu().numpy()
    else:
        orig_x_np = np.array(orig_x)
    
    df = pd.DataFrame(orig_x_np)
    
    if hasattr(model_names, 'tolist'):
        model_names = model_names.tolist()
    elif isinstance(model_names, torch.Tensor):
        model_names = model_names.detach().cpu().numpy().tolist()
    
    # Sort data
    df_sorted, scenarios_sorted, sorted_row_indices = sort_dataframe_by_mean_values_within_scenario(df, cat1)
    model_names_sorted = [model_names[i] for i in sorted_row_indices]
    
    # Modify visualization function to use custom colormap
    plot_data = df_sorted.values
    cmap = plt.cm.get_cmap(colormap)
    vmin, vmax = np.nanmin(plot_data), np.nanmax(plot_data)
    
    # Get boundaries and group info (same logic as before)
    boundaries = []
    for i in range(1, len(scenarios_sorted)):
        if scenarios_sorted[i] != scenarios_sorted[i - 1]:
            boundaries.append(i - 0.5)
    
    # Create plot
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(max(20, len(scenarios_sorted)//100), max(10, len(model_names)//50)))
        
        im = ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        
        for b in boundaries:
            ax.axvline(x=b, color="white", linewidth=0.5, linestyle="-", alpha=0.8)
        
        # Simplified labeling for large datasets
        ax.set_xticks([])
        if len(model_names_sorted) > 100:
            step = len(model_names_sorted) // 50
            tick_indices = range(0, len(model_names_sorted), step)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([model_names_sorted[i] for i in tick_indices], fontsize=6)
        else:
            ax.set_yticks(range(len(model_names_sorted)))
            ax.set_yticklabels(model_names_sorted, fontsize=max(3, 12-len(model_names_sorted)//50))
        
        ax.set_title(f"Model Performance Matrix ({colormap} colormap)", fontsize=14, pad=20)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
    
    print(f"Custom visualization saved to: {output_filename}")
meta = get_time_diff_matrix_for_benchmark(0)

visualize_meta_orig_x(meta)