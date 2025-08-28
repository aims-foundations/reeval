import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import pdb
from functools import reduce
def load_jbl_file(file_path):
    """
    Load a .jbl file and return its contents with basic info.
    
    Args:
        file_path (str): Path to the .jbl file
        
    Returns:
        data: The loaded data
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the file
    data = joblib.load(file_path)
    
    # Print basic info about the loaded data
    print(f"Loaded: {file_path.name}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        if data.size > 0:
            print(f"Range: [{data.min()}, {data.max()}]")
            print(f"Mean: {data.mean():.4f}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"First few items: {data[:3]}")
    elif hasattr(data, '__len__'):
        print(f"Length: {len(data)}")
    
    print()
    return data

def load_all_jbl_files(directory_path):
    """
    Load all .jbl files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing .jbl files
        
    Returns:
        dict: Dictionary with filename as key and loaded data as value
    """
    directory = Path(directory_path)
    
    # Get all .jbl files
    jbl_files = list(directory.glob("*.jbl"))
    
    print(f"Found {len(jbl_files)} .jbl files in {directory}")
    print("Files:", [f.name for f in jbl_files])
    print("=" * 50)
    
    loaded_data = {}
    
    for file_path in sorted(jbl_files):  # Sort for consistent order
        try:
            data = load_jbl_file(file_path)
            loaded_data[file_path.name] = data
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            print()
    
    print(f"Successfully loaded {len(loaded_data)} files")
    return loaded_data

# Example usage:
# scores_dir = "/lfs/skampere2/0/sttruong/model_benchmark/benchmark-prediction/data/scores"
# all_data = load_all_jbl_files(scores_dir)
print("=================")
# load_jbl_file("/lfs/skampere2/0/sttruong/model_benchmark/benchmark-prediction/data/scores/openllm_math_models.jbl")
# print(load_jbl_file("/lfs/skampere2/0/sttruong/model_benchmark/benchmark-prediction/data/scores/openllm_math.jbl"))
def find_duplicates_simple(string_list):
    """
    Simple way to find duplicates using Counter
    """
    from collections import Counter
    
    counts = Counter(string_list)
    duplicates = {item: count for item, count in counts.items() if count > 1}
    
    print(f"Total items: {len(string_list)}")
    print(f"Unique items: {len(set(string_list))}")
    print(f"Duplicate items: {len(duplicates)}")
    print(f"Duplicates: {duplicates}")
    
    return duplicates

def create_model_scores_dataframe(scores_array, model_names):
    """
    Create a pandas DataFrame with model names and their individual scores.
    
    Args:
        scores_array (np.ndarray): Shape (num_models, num_samples) with binary scores
        model_names (list): List of model names
        benchmark_name (str): Name of the benchmark for column naming
        
    Returns:
        pd.DataFrame: DataFrame with model names as first column and scores as subsequent columns
    """
    
    # Verify dimensions match
    assert len(model_names) == scores_array.shape[0], \
        f"Mismatch: {len(model_names)} models vs {scores_array.shape[0]} score rows"
    
    # Create column names: model_name, sample_0, sample_1, sample_2, ...
    columns = ['model_name'] + [f'sample_{i}' for i in range(scores_array.shape[1])]
    
    # Create data: each row is [model_name, score_0, score_1, score_2, ...]
    data = []
    for i, model_name in enumerate(model_names):
        row = [model_name] + list(scores_array[i])
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df = df.drop_duplicates()
    # pdb.set_trace()
    return df


def get_score_pandas(file_path):
    scores_array = load_jbl_file(file_path)
    model_path = file_path[:-4] + "_models.jbl"
    print("model_path",model_path)
    model_names = load_jbl_file(model_path)
    

    return create_model_scores_dataframe(scores_array,model_names)
    
# print(get_score_pandas("/lfs/skampere2/0/sttruong/model_benchmark/benchmark-prediction/data/scores/openllm_math.jbl"))


def load_all_benchmark_files(directory_path):
    """
    Load all .jbl files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing .jbl files
        
    Returns:
        dict: Dictionary with filename as key and loaded data as value
    """
    directory = Path(directory_path)
    
    # Get all .jbl files
    jbl_files = list(directory.glob("*.jbl"))
    
    print(f"Found {len(jbl_files)} .jbl files in {directory}")
    print("Files:", [f.name for f in jbl_files])
    print("=" * 50)
    
    loaded_data = {}
    open_bench = [
                "ifeval",
                "openllm_math",
                "mmlu_pro",
                "arc_challenge",
                "bbh",
                "gpqa",
                "musr"
            ]

    for file_path in sorted(jbl_files):  # Sort for consistent order
        if file_path.name.endswith("_models.jbl"):
            continue
        
        if not any(b in file_path.name for b in open_bench):
            print(f"Skipping {file_path.name} as it is not an openllm benchmark file")
            continue
        
        try:

            data = get_score_pandas(str(file_path))
            loaded_data[file_path.name] = data
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            print()
    
    print(f"Successfully loaded {len(loaded_data)} files")
    return loaded_data

# load_all_benchmark_files("/lfs/skampere2/0/sttruong/model_benchmark/benchmark-prediction/data/scores")


def simple_horizontal_join(benchmark_dict):
    """
    Simple horizontal join using iterative left outer joins.
    
    Args:
        benchmark_dict (dict): Output from your load_all_benchmark_files() function
        
    Returns:
        pd.DataFrame: Joined DataFrame with all benchmarks
    """
    
    if not benchmark_dict:
        return pd.DataFrame()
    
    result_df = None
    
    for i, (benchmark_name, df) in enumerate(benchmark_dict.items()):
        # Clean benchmark name for column prefixes
        clean_name = benchmark_name.replace('.jbl', '')
        
        # Create copy and rename columns (except model_name)
        df_processed = df.copy()
        
        # Rename sample columns to include benchmark name
        column_mapping = {}
        for col in df_processed.columns:
            if col.startswith('sample_'):
                sample_num = col.split('_')[1]
                column_mapping[col] = f'{clean_name}_q{sample_num}'
        
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Add summary statistics
        score_cols = [col for col in df_processed.columns if col.startswith(clean_name + '_q')]
        df_processed[f'{clean_name}_accuracy'] = df_processed[score_cols].mean(axis=1)
        df_processed[f'{clean_name}_total_correct'] = df_processed[score_cols].sum(axis=1)
        
        # First dataset becomes the base
        if result_df is None:
            result_df = df_processed
            print(f"Base dataset: {benchmark_name} - Shape: {result_df.shape}")
        else:
            # Left outer join with subsequent datasets
            result_df = pd.merge(result_df, df_processed, on='model_name', how='outer')
            # pdb.set_trace()
            print(f"Added {benchmark_name} - New shape: {result_df.shape}")
    
    # Sort by model name
    result_df = result_df.sort_values('model_name').reset_index(drop=True)
    
    print(f"\nFinal result:")
    print(f"Shape: {result_df.shape}")
    print(f"Unique models: {result_df['model_name'].nunique()}")
    
    return result_df

# Quick usage example:
def quick_join_example():
    """
    Quick example of how to use with your existing code
    """
    
    # Load your benchmark data
    scores_dir = "/lfs/skampere2/0/sttruong/model_benchmark/benchmark-prediction/data/scores"
    benchmark_data = load_all_benchmark_files(scores_dir)
    
    # Join horizontally
    joined_df = simple_horizontal_join(benchmark_data)
    print(joined_df['model_name'])
    # pdb.set_trace()
    # Save results
    joined_df.to_csv('all_benchmarks_joined.csv', index=False)
    joined_df.to_pickle('all_benchmarks_joined.pkl')
    
    # Show sample
    print("\nFirst 5 models and their accuracy across benchmarks:")
    accuracy_cols = [col for col in joined_df.columns if col.endswith('_accuracy')]
    print(joined_df[['model_name'] + accuracy_cols].head())
    
    return joined_df

# You can run this directly:
joined_data = quick_join_example()