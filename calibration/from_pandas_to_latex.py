import pandas as pd
import numpy as np

def create_multilevel_latex_table(df, selected_factors=[1, 2, 4, 8, 15, 30, 50]):
    """
    Create a LaTeX table with multi-level structure:
    - Top level: AUC and Correlation metrics
    - Second level: Factor values
    - Row levels: Dataset -> Masking Method -> Train/Test
    
    Args:
        df: DataFrame with columns ['metric', 'dataset', 'masking_method', 'K_fit', 'split', 'mean', 'ci95_low', 'ci95_high']
        selected_factors: List of factors to include in the table
    """
    
    def format_mean_ci(row):
        """Format mean ± CI for display"""
        if pd.isna(row['mean']) or pd.isna(row['ci95_low']) or pd.isna(row['ci95_high']):
            if not pd.isna(row['mean']):
                return f"{row['mean']:.3f} $\\pm$ --"
            
            
            return '--'
        
        mean = row['mean']
        ci_half_width = (row['ci95_high'] - row['ci95_low']) / 2
        
        return f"{mean:.3f} $\\pm$ {ci_half_width:.3f}"
    
    # Filter data for selected factors
    df_filtered = df[df['K_fit'].isin(selected_factors)].copy()
    
    # Ensure proper ordering
    datasets = ['HELM', 'official_provider', 'everything']
    masking_methods = ['random_mask', 'random_row', 'date', 'size']
    splits = ['train_dist', 'test_dist']
    metrics = ['auc', 'corr']
    
    # Calculate number of columns
    num_factor_cols = len(selected_factors)
    total_cols = num_factor_cols * 2  # 2 metrics
    
    lines = []
    
    # Table header
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\caption{Performance comparison across datasets, masking methods, and factors for AUC and Correlation metrics}")
    lines.append("\\label{tab:multilevel_results}")
    
    # Column specification
    col_spec = "lll" + "c" * total_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Create multi-level header
    # First header row: Metrics
    header1 = ["Dataset", "Masking", "Split"]
    auc_multicolumn = "\\multicolumn{" + str(num_factor_cols) + "}{c}{AUC}"
    corr_multicolumn = "\\multicolumn{" + str(num_factor_cols) + "}{c}{Correlation}"
    header1.extend([auc_multicolumn, corr_multicolumn])
    lines.append(" & ".join(header1) + " \\\\")
    
    # Add cmidrule for metric separation
    auc_range = f"{4}-{3+num_factor_cols}"
    corr_range = f"{4+num_factor_cols}-{3+total_cols}"
    cmidrule_auc = "\\cmidrule(lr){" + auc_range + "}"
    cmidrule_corr = "\\cmidrule(lr){" + corr_range + "}"
    lines.append(cmidrule_auc + " " + cmidrule_corr)
    
    # Second header row: Factors
    header2 = ["", "", ""]
    for metric in metrics:
        for factor in selected_factors:
            header2.append(f"K={factor}")
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")
    
    # Data rows
    row_count = 0
    total_dataset_rows = len(datasets) * len(masking_methods) * len(splits)
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_start_row = row_count
        
        for masking_idx, masking_method in enumerate(masking_methods):
            masking_start_row = row_count
            
            for split_idx, split in enumerate(splits):
                row_data = []
                
                # Dataset column (multirow on first occurrence)
                if row_count == dataset_start_row:
                    dataset_rows = len(masking_methods) * len(splits)
                    row_data.append(f"\\multirow{{{dataset_rows}}}{{*}}{{{dataset}}}")
                else:
                    row_data.append("")
                
                # Masking method column (multirow on first occurrence for each dataset)
                if row_count == masking_start_row:
                    masking_rows = len(splits)
                    masking_display = masking_method.replace('_', '\\_')
                    row_data.append(f"\\multirow{{{masking_rows}}}{{*}}{{{masking_display}}}")
                else:
                    row_data.append("")
                
                # Split column
                split_display = "Train" if split == "train_dist" else "Test"
                row_data.append(split_display)
                
                # Data columns for each metric and factor
                for metric in metrics:
                    for factor in selected_factors:
                        # Find the corresponding data point
                        data_point = df_filtered[
                            (df_filtered['dataset'] == dataset) &
                            (df_filtered['masking_method'] == masking_method) &
                            (df_filtered['split'] == split) &
                            (df_filtered['metric'] == metric) &
                            (df_filtered['K_fit'] == factor)
                        ]
                        
                        if len(data_point) > 0:
                            formatted_value = format_mean_ci(data_point.iloc[0])
                        else:
                            formatted_value = "--"
                        
                        row_data.append(formatted_value)
                
                lines.append(" & ".join(row_data) + " \\\\")
                row_count += 1
            
            # Add horizontal line between masking methods (except for last one in dataset)
            if masking_idx < len(masking_methods) - 1:
                cdashline_range = "2-" + str(3 + total_cols)
                lines.append("\\cdashline{" + cdashline_range + "}")
        
        # Add midrule between datasets (except for last one)
        if dataset_idx < len(datasets) - 1:
            lines.append("\\midrule")
    
    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def save_latex_table(df, filename="results/multilevel_table.tex", selected_factors=[1, 2, 4, 8, 15, 30, 50]):
    """
    Generate and save the LaTeX table to a file
    """
    latex_table = create_multilevel_latex_table(df, selected_factors)
    
    with open(filename, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {filename}")
    return latex_table

# Example usage:
# Assuming your DataFrame is loaded as 'df'
# latex_table = create_multilevel_latex_table(df)
# print(latex_table)

# Or to save to file:
# save_latex_table(df, "my_results_table.tex")
# Example usage:
# Assuming your DataFrame is loaded as 'df'


df = pd.read_csv("results/summary/partial_results_auc_corr_all.csv")

latex_table = create_multilevel_latex_table(df)
print(latex_table)

# Or to save to file:
save_latex_table(df, "results/tables/partial_results_auc_corr_table.tex")