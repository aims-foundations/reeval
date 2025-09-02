import pandas as pd
import numpy as np

def create_multilevel_latex_table(df, selected_factors=[0, 1, 2, 4, 8, 15, 30, 50], split_type='train_dist'):
    """
    Create a LaTeX table with multi-level structure:
    - Top level: AUC and Correlation metrics
    - Second level: Factor values
    - Row levels: Dataset -> Masking Method
    
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
                formatted = f"{row['mean']:.1f} $\\pm$ --"
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
    
    # Define masking methods per dataset
    datasets = ['HELM', 'official_provider', 'everything']
    dataset_masking_methods = {
        'HELM': ['random_mask', 'random_row'],  # HELM only has these two
        'official_provider': ['random_mask', 'random_row', 'date', 'size'],
        'everything': ['random_mask', 'random_row', 'date', 'size']
    }
    metrics = ['auc','corr']
    
    # Calculate number of columns
    num_factor_cols = len(selected_factors)
    total_cols = num_factor_cols * 2  # 2 metrics
    
    lines = []
    
    # Determine split display name for caption
    split_display = "Train" if split_type == "train_dist" else "Test"
    
    # Table header
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{Performance comparison across datasets and masking methods for AUC and Correlation metrics ({split_display} Split)}}")
    lines.append("\\label{tab:multilevel_results_" + split_type + "}")
    
    # Column specification
    col_spec = "ll" + "c" * total_cols  # Removed one 'l' since we don't have split column anymore
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Create multi-level header
    # First header row: Metrics
    header1 = ["Dataset", "Masking"]  # Removed "Split"
    auc_multicolumn = "\\multicolumn{" + str(num_factor_cols) + "}{c}{AUC}"
    corr_multicolumn = "\\multicolumn{" + str(num_factor_cols) + "}{c}{Correlation}"
    header1.extend([auc_multicolumn, corr_multicolumn])
    lines.append(" & ".join(header1) + " \\\\")
    
    # Add cmidrule for metric separation
    auc_range = f"{3}-{2+num_factor_cols}"  # Adjusted ranges since we removed one column
    corr_range = f"{3+num_factor_cols}-{2+total_cols}"
    cmidrule_auc = "\\cmidrule(lr){" + auc_range + "}"
    cmidrule_corr = "\\cmidrule(lr){" + corr_range + "}"
    lines.append(cmidrule_auc + " " + cmidrule_corr)
    
    # Second header row: Factors
    header2 = ["", ""]  # Reduced from 3 empty strings to 2
    for metric in metrics:
        for factor in selected_factors:
            header2.append(f"K={factor}")
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")
    
    # Data rows
    row_count = 0
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_start_row = row_count
        masking_methods = dataset_masking_methods[dataset]  # Get masking methods for this dataset
        
        for masking_idx, masking_method in enumerate(masking_methods):
            row_data = []
            
            # Dataset column (multirow on first occurrence)
            if row_count == dataset_start_row:
                dataset_rows = len(masking_methods)  # Reduced since we don't multiply by splits anymore
                row_data.append(f"\\multirow{{{dataset_rows}}}{{*}}{{{dataset.replace('_',' ')}}}")
            else:
                row_data.append("")
            
            # Masking method column
            masking_display = masking_method.replace('_', '\\_')
            row_data.append(masking_display)
            
            # Data columns for each metric and factor
            for metric in metrics:
                # First, collect all data points for this row and metric to find the best
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
                
                # Now format all values for this metric, marking the best one
                for factor, data in metric_data:
                    if data is not None:
                        is_best = (factor == best_factor and best_factor is not None)
                        formatted_value = format_mean_ci(data, is_best)
                    else:
                        formatted_value = "--"
                    
                    row_data.append(formatted_value)
            
            lines.append(" & ".join(row_data) + " \\\\")
            row_count += 1
        
        # Add midrule between datasets (except for last one)
        if dataset_idx < len(datasets) - 1:
            lines.append("\\midrule")
    
    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def save_latex_table(df, filename="results/multilevel_table.tex", selected_factors=[0, 1, 2, 4, 8, 15, 30, 50], split_type='train_dist'):
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
    df = pd.read_csv("results/summary/partial_results_auc_corr_all.csv")
    
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
    save_latex_table(df, "results/tables/partial_results_auc_corr_table_train.tex", split_type='train_dist')
    save_latex_table(df, "results/tables/partial_results_auc_corr_table_test.tex", split_type='test_dist')