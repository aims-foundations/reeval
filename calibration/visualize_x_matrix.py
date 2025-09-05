import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
import warnings
from util import get_time_diff_matrix_for_benchmark
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

def smart_sample_matrix(df, scenarios, model_names, max_rows=1000, max_cols=5000):
    """
    Intelligently sample the matrix to reduce size while preserving key patterns.
    Uses random sampling to preserve original distribution.
    
    Args:
        df: DataFrame to sample
        scenarios: List of scenario names
        model_names: List of model names  
        max_rows: Maximum number of rows (models) to keep
        max_cols: Maximum number of columns (questions/scenarios) to keep
    
    Returns:
        Sampled dataframe, sampled scenarios, sampled model names
    """
    
    # Sample rows (models) if needed - USE RANDOM SAMPLING
    if len(model_names) > max_rows:
        print(f"Sampling {max_rows} models from {len(model_names)} total models...")
        
        # Use purely random sampling to preserve distribution
        np.random.seed(42)  # For reproducibility
        sampled_row_indices = sorted(np.random.choice(
            range(len(model_names)), 
            size=max_rows, 
            replace=False
        ).tolist())
        
        df = df.iloc[sampled_row_indices, :]
        model_names = [model_names[i] for i in sampled_row_indices]
        
        print(f"Random sampling: kept {max_rows} models (preserves original distribution)")
    
    # Sample columns (scenarios) if needed - USE RANDOM SAMPLING
    if len(scenarios) > max_cols:
        print(f"Sampling {max_cols} columns from {len(scenarios)} total columns...")
        
        # Use purely random sampling to preserve distribution
        np.random.seed(42)  # For reproducibility
        sampled_col_indices = sorted(np.random.choice(
            range(len(scenarios)), 
            size=max_cols, 
            replace=False
        ).tolist())
        
        df = df.iloc[:, sampled_col_indices]
        scenarios = [scenarios[i] for i in sampled_col_indices]
        
        print(f"Random sampling: kept {max_cols} columns (preserves original distribution)")
    
    return df, scenarios, model_names

def visualize_continuous_matrix(df, scenarios, model_names, filename, title="Model Performance Matrix", 
                               original_counts=None):
    """
    Enhanced visualization with better labeling for models, scenarios, and legend.
    
    Args:
        df: DataFrame with continuous values
        scenarios: List of scenario names for each column
        model_names: List of model names for rows
        filename: Output filename for the plot
        title: Title for the plot
        original_counts: Dict with original data size info for sampling warning
    """
    
    # Convert data to numpy array for plotting
    plot_data = df.values if hasattr(df, 'values') else np.array(df)
    
    # Create colormap for continuous values
    cmap = plt.cm.viridis
    
    # Calculate value range for normalization
    vmin = np.nanmin(plot_data)
    vmax = np.nanmax(plot_data)
    
    # Identify the boundaries where the scenario changes
    boundaries = []
    for i in range(1, len(scenarios)):
        if scenarios[i] != scenarios[i - 1]:
            boundaries.append(i - 0.5)

    # Calculate midpoints and boundaries for each scenario group
    scenario_info = []
    current_scenario = scenarios[0]
    start_index = 0
    
    for i, scenario in enumerate(scenarios + [None]):  # Add None to handle last group
        if scenario != current_scenario or scenario is None:
            end_index = i if scenario is not None else len(scenarios)
            midpoint = (start_index + end_index - 1) / 2.0
            scenario_info.append({
                'name': current_scenario,
                'start': start_index,
                'end': end_index - 1,
                'midpoint': midpoint,
                'width': end_index - start_index
            })
            
            if scenario is not None:
                current_scenario = scenario
                start_index = i

    # Calculate adaptive figure size
    base_width = max(16, len(scenarios) * 0.1)
    base_height = max(10, len(model_names) * 0.08)
    
    # Plot the matrix
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(base_width, base_height))
        
        # Create the heatmap
        im = ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Add vertical lines at scenario boundaries
        for b in boundaries:
            ax.axvline(x=b, color="white", linewidth=1.5, linestyle="-", alpha=0.9)
        
        # Enhanced scenario labeling with better spacing and rotation
        scenario_font_size = max(6, min(12, 120 // len(scenario_info)))
        
        # Create scenario labels with better positioning
        for info in scenario_info:
            # Only show label if scenario has reasonable width
            if info['width'] >= 2:  # Only label scenarios with at least 2 columns
                ax.text(info['midpoint'], -len(model_names)*0.03, info['name'], 
                       ha='center', va='bottom', rotation=45, 
                       fontsize=scenario_font_size, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        # Enhanced model name labeling
        model_font_size = max(4, min(10, 100 // len(model_names)))
        
        if len(model_names) > 150:
            # For very large datasets, show every nth model
            step = max(1, len(model_names) // 50)
            tick_indices = range(0, len(model_names), step)
            tick_labels = [model_names[i] for i in tick_indices]
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels, fontsize=model_font_size)
        elif len(model_names) > 50:
            # For medium datasets, show every other model
            step = 2
            tick_indices = range(0, len(model_names), step)
            tick_labels = [model_names[i] for i in tick_indices]
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels, fontsize=model_font_size)
        else:
            # For small datasets, show all models
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=model_font_size)

        # Remove x-axis column ticks (individual columns)
        ax.set_xticks([])
        
        # Enhanced axis labels
        ax.set_xlabel('Scenarios (grouped and sorted by mean performance)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Models (sorted by overall mean performance)', fontsize=12, fontweight='bold')
        
        # Add title with more context
        ax.set_title(f'{title}\n(Higher values = better performance)', fontsize=14, fontweight='bold', pad=30)
        
        # Enhanced colorbar with better labeling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Time Difference (Days)\nLower = Better Performance', 
                      rotation=270, labelpad=25, fontsize=11, fontweight='bold')
        
        # Create meaningful colorbar ticks
        n_ticks = 8
        tick_values = np.linspace(vmin, vmax, n_ticks)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.1f}' for val in tick_values], fontsize=9)
        
        # Add sampling info if data was sampled
        stats_y_pos = 0.98
#         if original_counts:
#             sampling_info = f"""SAMPLED DATA WARNING:
# • Showing {len(model_names)} of {original_counts['models']} models
# • Showing {len(scenarios)} of {original_counts['columns']} total columns
# • Sampling preserves best/worst performers + random middle"""
            
#             ax.text(0.02, 0.98, sampling_info, transform=ax.transAxes, fontsize=9, 
#                    verticalalignment='top', fontfamily='monospace',
#                    bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.9, edgecolor='red'))
#             stats_y_pos = 0.75  # Move stats box down
        
        # Add statistics text box
#         stats_text = f'''Dataset Statistics:
# • Models: {len(model_names)}
# • Scenarios: {len(set(scenarios))}
# • Value Range: [{vmin:.1f}, {vmax:.1f}] days
# • Mean: {np.nanmean(plot_data):.1f} days
# • Std: {np.nanstd(plot_data):.1f} days'''
        
#         ax.text(0.02, stats_y_pos, stats_text, transform=ax.transAxes, fontsize=9, 
#                verticalalignment='top', fontfamily='monospace',
#                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Add scenario summary in top right
        scenario_counts = pd.Series(scenarios).value_counts()
        scenario_summary = "Scenario Distribution:\n" + "\n".join([f"• {name}: {count} cols" 
                                                                  for name, count in scenario_counts.head(5).items()])
        if len(scenario_counts) > 5:
            scenario_summary += f"\n• ... and {len(scenario_counts) - 5} more"
            
        ax.text(0.98, 0.98, scenario_summary, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='gray'))
        
        # Improve layout
        plt.subplots_adjust(left=0.12, bottom=0.08, right=0.88, top=0.92)
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
        plt.close()
        
    # Enhanced output information
    print(f"Visualization saved to: {filename}")
    print(f"Data shape: {plot_data.shape} (rows: models, cols: scenarios)")
    print(f"Value range: [{vmin:.2f}, {vmax:.2f}] days")
    print(f"Number of unique scenarios: {len(set(scenarios))}")
    print(f"Number of models: {len(model_names)}")
    print(f"Best performing model: {model_names[0]} (mean: {np.nanmean(plot_data[0,:]):.2f})")
    print(f"Worst performing model: {model_names[-1]} (mean: {np.nanmean(plot_data[-1,:]):.2f})")

def visualize_meta_orig_x(meta, output_filename="results/plot/meta_orig_x_visualization.png", 
                         max_rows=1000, max_cols=5000):
    """
    Enhanced main function to visualize meta['orig_x'] data with intelligent sampling for large datasets.
    
    Args:
        meta: Dictionary containing 'orig_x', 'model_names', and 'cat1'
        output_filename: Name for the output file
        max_rows: Maximum number of rows (models) to display
        max_cols: Maximum number of columns (questions/scenarios) to display
    """
    
    # Extract data
    orig_x = meta['orig_x']
    model_names = meta['model_names']
    cat1 = meta['cat1']
    
    # Store original counts for sampling warning
    original_counts = {
        'models': len(model_names),
        'columns': len(cat1)
    }
    
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
    
    print("=" * 60)
    print("ENHANCED MODEL PERFORMANCE MATRIX VISUALIZATION")
    print("=" * 60)
    print(f"Original data shape: {df.shape}")
    print(f"Sample of original data (first 5x5):")
    print(df.iloc[:5, :5])
    print(f"Data type: {type(orig_x_np[0,0])}")
    print(f"Scenarios: {len(set(cat1))} unique scenarios")
    print(f"Models: {len(model_names)} total models")
    
    # Sort the data first
    print("\n" + "-" * 40)
    print("SORTING DATA BY PERFORMANCE...")
    print("-" * 40)
    df_sorted, scenarios_sorted, sorted_row_indices = sort_dataframe_by_mean_values_within_scenario(df, cat1)
    
    # Get sorted model names
    model_names_sorted = [model_names[i] for i in sorted_row_indices]
    
    print("Data sorted successfully!")
    print(f"Best performing model (lowest mean): {model_names_sorted[0]}")
    print(f"Worst performing model (highest mean): {model_names_sorted[-1]}")
    
    # Smart sampling for large datasets
    print("\n" + "-" * 40)
    print("SMART SAMPLING FOR VISUALIZATION...")
    print("-" * 40)
    
    df_sampled, scenarios_sampled, model_names_sampled = smart_sample_matrix(
        df_sorted, scenarios_sorted, model_names_sorted, max_rows, max_cols
    )
    
    print(f"Final visualization size: {df_sampled.shape}")
    print(f"Sample of sampled data (first 5x5):")
    print(df_sampled.iloc[:5, :5])
    
    # Determine if sampling occurred
    sampling_occurred = (len(model_names_sampled) < original_counts['models'] or 
                        len(scenarios_sampled) < original_counts['columns'])
    
    # Create enhanced visualization
    print("\n" + "-" * 40)
    print("CREATING ENHANCED VISUALIZATION...")
    print("-" * 40)
    visualize_continuous_matrix(
        df_sampled, 
        scenarios_sampled, 
        model_names_sampled, 
        output_filename,
        title="Model Performance Matrix: Time Difference Analysis" + (" (Sampled)" if sampling_occurred else ""),
        original_counts=original_counts if sampling_occurred else None
    )
    print("=" * 60)

# Enhanced version with custom colormap and additional features
def visualize_meta_orig_x_advanced(meta, output_filename="meta_orig_x_advanced.png", 
                                 colormap='RdYlBu_r', show_values=False, max_rows=1000, max_cols=5000):
    """
    Advanced visualization with custom colormap and optional value display.
    
    Args:
        meta: Dictionary containing the data
        output_filename: Output filename
        colormap: Matplotlib colormap name
        show_values: Whether to display actual values in cells (for small matrices)
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
    """
    
    # Extract and process data
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
    
    # Sort data first
    df_sorted, scenarios_sorted, sorted_row_indices = sort_dataframe_by_mean_values_within_scenario(df, cat1)
    model_names_sorted = [model_names[i] for i in sorted_row_indices]
    
    # Apply sampling
    df_sampled, scenarios_sampled, model_names_sampled = smart_sample_matrix(
        df_sorted, scenarios_sorted, model_names_sorted, max_rows, max_cols
    )
    
    plot_data = df_sampled.values
    cmap = plt.cm.get_cmap(colormap)
    vmin, vmax = np.nanmin(plot_data), np.nanmax(plot_data)
    
    # Create enhanced plot with custom features
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        fig_width = max(20, len(scenarios_sampled)*0.15)
        fig_height = max(12, len(model_names_sampled)*0.1)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        im = ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add scenario boundaries
        boundaries = []
        for i in range(1, len(scenarios_sampled)):
            if scenarios_sampled[i] != scenarios_sampled[i - 1]:
                boundaries.append(i - 0.5)
                ax.axvline(x=i - 0.5, color="white", linewidth=2, linestyle="-", alpha=1.0)
        
        # Enhanced labeling system
        if len(model_names_sampled) <= 30 and len(scenarios_sampled) <= 50 and show_values:
            # Show actual values in cells for small matrices
            for i in range(len(model_names_sampled)):
                for j in range(len(scenarios_sampled)):
                    value = plot_data[i, j]
                    color = 'white' if (value - vmin) / (vmax - vmin) < 0.5 else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                           color=color, fontsize=8, fontweight='bold')
        
        # Model name labeling with smart sampling
        ax.set_yticks(range(len(model_names_sampled)))
        ax.set_yticklabels(model_names_sampled, fontsize=max(6, 60//len(model_names_sampled)))
        
        # Scenario labeling at top
        current_scenario = scenarios_sampled[0]
        start_pos = 0
        
        for i, scenario in enumerate(scenarios_sampled + [None]):
            if scenario != current_scenario or scenario is None:
                end_pos = i if scenario is not None else len(scenarios_sampled)
                mid_pos = (start_pos + end_pos - 1) / 2
                
                if end_pos - start_pos >= 3:  # Only label scenarios with enough width
                    ax.text(mid_pos, -len(model_names_sampled)*0.04, current_scenario,
                           ha='center', va='bottom', rotation=45, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
                
                if scenario is not None:
                    current_scenario = scenario
                    start_pos = i
        
        # Enhanced title and labels
        ax.set_title(f'Advanced Model Performance Analysis\n({colormap} colormap)', 
                    fontsize=16, fontweight='bold', pad=40)
        ax.set_xlabel('Scenarios →', fontsize=14, fontweight='bold')
        ax.set_ylabel('← Models (Best to Worst)', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        
        # Premium colorbar with custom styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=40, pad=0.02)
        cbar.set_label('Performance Score (Days)\n← Better    Worse →', 
                      rotation=270, labelpad=30, fontsize=12, fontweight='bold')
        
        # Custom colorbar ticks with percentiles
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        percentile_values = np.percentile(plot_data, percentiles)
        cbar.set_ticks(percentile_values)
        cbar.set_ticklabels([f'P{p}\n{v:.1f}' for p, v in zip(percentiles, percentile_values)], 
                           fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
    
    print(f"Advanced visualization saved to: {output_filename}")
    return fig

# Usage examples:
if __name__ == "__main__":
    # Get your data
    meta = get_time_diff_matrix_for_benchmark(0)
    
    # Create enhanced basic visualization with intelligent sampling (question-preserving defaults)
    visualize_meta_orig_x(meta, "enhanced_matrix_visualization.png", 
                         max_rows=200, max_cols=1000)  # Preserve more questions/scenarios
    
    # Create advanced visualization with custom colormap
    # visualize_meta_orig_x_advanced(meta, "advanced_matrix_viz.png", 
    #                               colormap='coolwarm', show_values=False,
    #                               max_rows=800, max_cols=6000)