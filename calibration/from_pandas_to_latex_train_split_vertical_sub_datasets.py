import pandas as pd
import numpy as np

def create_multilevel_latex_table(df, selected_factors=[0, 1, 2, 4, 8, 15, 30, 50], split_type='train_dist'):
    """
    Create a LaTeX table with vertically stacked metrics:
    - Each metric (AUC, Correlation, log_p) gets its own sub-table
    - Each sub-table has: Dataset -> Masking Method rows, Factor columns
    
    Args:
        df: DataFrame with columns ['metric', 'dataset', 'masking_method', 'K_fit', 'split', 'mean', 'ci95_low', 'ci95_high']
        selected_factors: List of factors to include in the table
        split_type: Either 'train_dist' or 'test_dist' to specify which split to show
    """
    
    def format_mean_ci(row, is_best=False):
        """Format mean ± CI for display, with optional bolding for best values"""
        if pd.isna(row['mean']) or pd.isna(row['ci95_low']) or pd.isna(row['ci95_high']):
            if not pd.isna(row['mean']):
                row['mean'] *= 100
                # formatted = f"{row['mean']:.1f} $\\pm$ --"
                formatted = f"{row['mean']:.1f}" 
            else:
                formatted = '--'
        else:
            row['mean'] *= 100
            row['ci95_high'] *= 100
            row['ci95_low'] *= 100
            
            mean = row['mean']
            ci_half_width = (row['ci95_high'] - row['ci95_low']) / 2
            formatted = f"{mean:.1f} $\\pm$ {ci_half_width:.2f}"
        
        if is_best and formatted != '--':
            formatted = f"\\textbf{{{formatted}}}"
        
        return formatted
    
    # Filter data for selected factors and split type
    df_filtered = df[(df['K_fit'].isin(selected_factors)) & (df['split'] == split_type)].copy()
    
    # Automatically detect datasets and masking methods from the DataFrame
    datasets = sorted(df_filtered['dataset'].unique())
    all_masking_methods = ['random_mask', 'random_row', 'date', 'size']  # All datasets have all four methods
    
    metrics = ['auc', 'corr', 'log_p']
    metric_names = {'auc': 'AUC', 'corr': 'Correlation', 'log_p': 'Log Probability'}
    
    # Calculate number of columns
    num_factor_cols = len(selected_factors)
    
    lines = []
    
    # Determine split display name for caption
    split_display = "Train" if split_type == "train_dist" else "Test"
    
    # Table header
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{Performance comparison across datasets and masking methods for AUC, Correlation, and Log Probability metrics ({split_display} Split)}}")
    lines.append("\\label{tab:multilevel_results_" + split_type + "}")
    
    # Column specification
    col_spec = "ll" + "c" * num_factor_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Create header row for factors
    header = ["Dataset", "Masking"]
    for factor in selected_factors:
        header.append(f"K={factor}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    
    # Generate sub-tables for each metric
    for metric_idx, metric in enumerate(metrics):
        # Add metric section header
        metric_header = f"\\multicolumn{{{len(header)}}}{{c}}{{\\textbf{{{metric_names[metric]}}}}}"
        lines.append(metric_header + " \\\\")
        lines.append("\\midrule")
        
        # Data rows for this metric
        row_count = 0
        
        for dataset_idx, dataset in enumerate(datasets):
            dataset_start_row = row_count
            masking_methods = all_masking_methods  # All datasets have all four methods
            
            for masking_idx, masking_method in enumerate(masking_methods):
                row_data = []
                
                # Dataset column (multirow on first occurrence)
                if row_count == dataset_start_row:
                    dataset_rows = len(masking_methods)
                    # Clean up dataset name for display
                    dataset_display = dataset.replace('_', '\\_')
                    row_data.append(f"\\multirow{{{dataset_rows}}}{{*}}{{{dataset_display}}}")
                else:
                    row_data.append("")
                
                # Masking method column
                masking_display = masking_method.replace('_', '\\_')
                row_data.append(masking_display)
                
                # Collect all data points for this row and metric to find the best
                metric_data = []
                for factor in selected_factors:
                    data_point = df_filtered[
                        (df_filtered['dataset'] == dataset) &
                        (df_filtered['masking_method'] == masking_method) &
                        (df_filtered['metric'] == metric) &
                        (df_filtered['K_fit'] == factor)
                    ]
                    
                    if len(data_point) > 0:
                        metric_data.append((factor, data_point.iloc[0]))
                    else:
                        metric_data.append((factor, None))
                
                # Find the best value (highest mean) for this metric in this row
                best_factor = None
                best_mean = -np.inf
                for factor, data in metric_data:
                    if data is not None and not pd.isna(data['mean']):
                        if data['mean'] > best_mean:
                            best_mean = data['mean']
                            best_factor = factor
                
                # Format all values for this metric, marking the best one
                for factor, data in metric_data:
                    if data is not None:
                        is_best = (factor == best_factor and best_factor is not None)
                        formatted_value = format_mean_ci(data, is_best)
                    else:
                        formatted_value = "--"
                    
                    row_data.append(formatted_value)
                
                lines.append(" & ".join(row_data) + " \\\\")
                row_count += 1
            
            # Add midrule between datasets within a metric (except for last dataset)
            if dataset_idx < len(datasets) - 1:
                lines.append("\\midrule")
        
        # Add spacing between metrics (except for last metric)
        if metric_idx < len(metrics) - 1:
            lines.append("\\midrule")
            lines.append("\\addlinespace[0.5em]")  # Add some vertical space between metric sections
    
    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def save_latex_table(df, filename="results/multilevel_table.tex", selected_factors=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256], split_type='train_dist'):
    """
    Generate and save the LaTeX table to a file
    
    Args:
        df: DataFrame with the data
        filename: Output filename for the LaTeX table
        selected_factors: List of factors to include in the table
        split_type: Either 'train_dist' or 'test_dist' to specify which split to show
    """
    latex_table = create_multilevel_latex_table(df, selected_factors, split_type)

    with open(filename, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {filename}")
    return latex_table

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("/lfs/mercury2/0/sttruong/reeval/results/summary/partial_results_auc_corr_sub_dataset.csv")
    
    # Generate table for train split
    latex_table_train = create_multilevel_latex_table(df, split_type='train_dist')
    print("TRAIN SPLIT TABLE:")
    print(latex_table_train)
    print("\n" + "="*50 + "\n")
    
    # Generate table for test split
    latex_table_test = create_multilevel_latex_table(df, split_type='test_dist')
    print("TEST SPLIT TABLE:")
    print(latex_table_test)
    
    # Save both tables to separate files
    save_latex_table(df, "results/tables/partial_results_auc_corr_table_train_sub_dataset.tex", split_type='train_dist')
    save_latex_table(df, "results/tables/partial_results_auc_corr_table_test_sub_dataset.tex", split_type='test_dist')