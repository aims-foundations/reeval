import torch
import pandas as pd
import numpy as np
data_list = []
factors = [0, 1, 2, 4, 8, 16, 32]
i = 0
benchmarks = ['openllm_math','ifeval', 'musr', 'bbh', 'gpqa', 'mmlu_pro']  # 'arc_challenge', 
for K_fit in factors:
    for train_idx in range(6):
        for test_idx in range(6):
            train_benchmark = benchmarks[train_idx]
            test_benchmark = benchmarks[test_idx]
            config_name = f"everything3_train{train_idx}_test{test_idx}_k{K_fit}_i{i}"
            try:
                train_auc = torch.load(f"results/pred_dataset/train_auc_{config_name}.pt")
                test_auc = torch.load(f"results/pred_dataset/test_auc_{config_name}.pt")

                r_train = torch.load(f"results/pred_dataset/train_corr_{config_name}.pt",weights_only=False)
                r_test = torch.load(f"results/pred_dataset/test_corr_{config_name}.pt",weights_only=False)

                run_data = {
                    "train_benchmark":train_benchmark,
                    "test_benchmark":test_benchmark,
                    "test_auc":test_auc,
                    "r_test":r_test,
                    "K_fit":K_fit,

                }
            except:
                run_data = {
                    "train_benchmark":train_benchmark,
                    "test_benchmark":test_benchmark,
                    "test_auc":None,
                    "r_test":None,
                    "K_fit":K_fit,

                }
            data_list.append(run_data)
            
df = pd.DataFrame(data_list)


def create_benchmark_cross_table(df, selected_factors=[0, 1, 2, 4, 8, 16, 32]):
    """
    Create a LaTeX table with 7x7 benchmark matrix structure:
    - Rows: Train benchmarks
    - Columns: Test benchmarks  
    - Each cell contains K_fit factors with AUC and Correlation values
    
    Args:
        df: DataFrame with columns ['train_benchmark', 'test_benchmark', 'test_auc', 'r_test', 'K_fit']
        selected_factors: List of K_fit factors to include
    """
    
    # Get unique benchmarks (should be 7)
    benchmarks = ['openllm_math', 'ifeval', 'musr', 'bbh', 'gpqa', 'mmlu_pro'] # 'arc_challenge',
    
    def find_best_values_in_cell(train_bench, test_bench, selected_factors):
        """Find the best AUC and correlation values for a given train-test pair"""
        cell_data = df[
            (df['train_benchmark'] == train_bench) &
            (df['test_benchmark'] == test_bench) &
            (df['K_fit'].isin(selected_factors))
        ]
        
        if len(cell_data) == 0:
            return None, None
        
        # Find best AUC
        valid_auc_data = cell_data.dropna(subset=['test_auc'])
        best_auc_factor = None
        if len(valid_auc_data) > 0:
            valid_auc_data = valid_auc_data.copy()
            valid_auc_data['auc_float'] = valid_auc_data['test_auc'].apply(
                lambda x: x.item() if isinstance(x, torch.Tensor) else x
            )
            best_auc_idx = valid_auc_data['auc_float'].idxmax()
            best_auc_factor = valid_auc_data.loc[best_auc_idx, 'K_fit']
        
        # Find best correlation
        valid_corr_data = cell_data.dropna(subset=['r_test'])
        best_corr_factor = None
        if len(valid_corr_data) > 0:
            best_corr_idx = valid_corr_data['r_test'].idxmax()
            best_corr_factor = valid_corr_data.loc[best_corr_idx, 'K_fit']
        
        return best_auc_factor, best_corr_factor
    
    def format_auc_corr(auc_val, corr_val, k_factor, best_auc_factor, best_corr_factor):
        """Format AUC and correlation values with bolding for best values"""
        if auc_val is None or pd.isna(auc_val):
            auc_str = "--"
        else:
            auc_val *= 100
            
            if isinstance(auc_val, torch.Tensor):
                auc_val = auc_val.item()
            auc_str = f"{auc_val:.1f}"
            # Bold if this is the best AUC factor
            if k_factor == best_auc_factor:
                auc_str = f"\\textbf{{{auc_str}}}"
        
        if corr_val is None or pd.isna(corr_val):
            corr_str = "--"
        else:
            corr_val *= 100
            corr_str = f"{corr_val:.1f}"
            # Bold if this is the best correlation factor
            if k_factor == best_corr_factor:
                corr_str = f"\\textbf{{{corr_str}}}"
        
        return f"{auc_str} / {corr_str}"
    
    lines = []
    
    # Table header
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\tiny")  # Use tiny font for large table
    lines.append("\\caption{Cross-benchmark performance matrix showing AUC / Correlation for different K-fit values}")
    lines.append("\\label{tab:benchmark_cross_matrix}")
    
    # Column specification - one column for train benchmark, then 7 columns for test benchmarks
    col_spec = "l" + "c" * len(benchmarks)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header row with test benchmark names
    header = ["Train $\\backslash$ Test"]
    for test_bench in benchmarks:
        # Clean benchmark names for display
        clean_name = test_bench.replace('_', '\\_')
        header.append(f"\\rotatebox{{45}}{{{clean_name}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    
    # Data rows - one major row per train benchmark
    for train_bench in benchmarks:
        # Clean train benchmark name
        clean_train_name = train_bench.replace('_', '\\_')
        
        # Pre-compute best values for all test benchmarks for this train benchmark
        best_values_cache = {}
        for test_bench in benchmarks:
            best_auc_factor, best_corr_factor = find_best_values_in_cell(
                train_bench, test_bench, selected_factors
            )
            best_values_cache[test_bench] = (best_auc_factor, best_corr_factor)
        
        # Create sub-table for this train benchmark
        lines.append(f"\\multirow{{{len(selected_factors)}}}{{*}}{{{clean_train_name}}}")
        
        for factor_idx, factor in enumerate(selected_factors):
            row_data = []
            
            # First column: empty for multirow, or K factor value
            if factor_idx == 0:
                row_data.append("")  # Empty because of multirow
            else:
                row_data.append("")  # Empty for subsequent rows
            
            # Add K factor indicator in a separate mini-column or as part of first test result
            factor_label = f"K={factor}"
            
            # Data columns for each test benchmark
            first_col = True
            for test_bench in benchmarks:
                # Get best values for this cell
                best_auc_factor, best_corr_factor = best_values_cache[test_bench]
                
                # Find the data point
                data_point = df[
                    (df['train_benchmark'] == train_bench) &
                    (df['test_benchmark'] == test_bench) &
                    (df['K_fit'] == factor)
                ]
                
                if len(data_point) > 0:
                    auc_val = data_point.iloc[0]['test_auc']
                    corr_val = data_point.iloc[0]['r_test']
                    formatted_value = format_auc_corr(
                        auc_val, corr_val, factor, best_auc_factor, best_corr_factor
                    )
                else:
                    formatted_value = "-- / --"
                
                # For the first column, prepend the K factor
                if first_col:
                    formatted_value = f"{factor_label}: {formatted_value}"
                    first_col = False
                
                row_data.append(formatted_value)
            
            lines.append(" & ".join(row_data) + " \\\\")
        
        # Add space between train benchmarks (except for last one)
        if train_bench != benchmarks[-1]:
            lines.append("\\midrule")
    
    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")
    lines.append("% Note: Values are formatted as AUC / Correlation")
    lines.append("% Bold values indicate the highest AUC or Correlation within each cell")
    lines.append("% -- indicates missing or None values")

    return "\n".join(lines)


def save_latex_tables(df, filename_full="results/tables/benchmark_cross_table_full_no_arc_3.tex", 
                    #  filename_compact="benchmark_cross_table_compact.tex",
                     selected_factors=[0, 1, 2, 4, 8, 16, 32]):
    """
    Generate and save both LaTeX table versions
    """
    
    # Generate full table
    latex_table_full = create_benchmark_cross_table(df, selected_factors)
    with open(filename_full, 'w') as f:
        f.write(latex_table_full)
    print(f"Full LaTeX table saved to {filename_full}")
    
    # # Generate compact table
    # latex_table_compact = create_compact_benchmark_table(df, selected_factors)
    # with open(filename_compact, 'w') as f:
    #     f.write(latex_table_compact)
    # print(f"Compact LaTeX table saved to {filename_compact}")
    
    # return latex_table_full, latex_table_compact

save_latex_tables(df)