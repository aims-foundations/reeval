import pandas as pd
import numpy as np
import torch
import pickle
from huggingface_hub import snapshot_download

import warnings
warnings.filterwarnings("ignore")


def uploaded_before(df: pd.DataFrame, cutoff_date: str):
    """
    Compare Upload To Hub Date against cutoff_date.

    Args:
        df: DataFrame with column 'Upload To Hub Date'
        cutoff_date: str or datetime, e.g. '2024-08-01'

    Returns:
        pandas.Series of bools: True if upload date < cutoff_date,
                               False if >= cutoff_date or NaN.
    """
    # ensure datetime
    cutoff = pd.to_datetime(cutoff_date)
    upload_dates = pd.to_datetime(df["Upload To Hub Date"], errors="coerce")

    mask = (upload_dates < cutoff) & (~upload_dates.isna())
    return mask

def size_smaller(df: pd.DataFrame, cutoff_size: float):
    """
    Compare Upload To Hub Date against cutoff_date.

    Args:
        df: DataFrame with column 'Upload To Hub Date'
        cutoff_date: str or datetime, e.g. '2024-08-01'

    Returns:
        pandas.Series of bools: True if upload date < cutoff_date,
                               False if >= cutoff_date or NaN.
    """
    # ensure datetime

    model_size = df['#Params (B)']

    mask = (model_size < cutoff_size) & (~model_size.isna())
    return mask

def keep_random_trues(series, keep_count=500):
    """
    Randomly keep only 'keep_count' True values in a boolean Series,
    setting the rest to False.
    
    Parameters:
    series: pandas Series of boolean values
    keep_count: number of True values to keep (default 500)
    
    Returns:
    pandas Series with only keep_count True values randomly selected
    """
    # Make a copy to avoid modifying the original
    result = series.copy()
    
    # Get indices where values are True
    true_indices = series[series == True].index
    
    # If we have fewer trues than requested, return as is
    if len(true_indices) <= keep_count:
        return result
    
    # Randomly sample which True indices to keep
    indices_to_keep = np.random.choice(true_indices, size=keep_count, replace=False)
    
    # Set all True values to False first
    result[series == True] = False
    
    # Set the randomly selected indices back to True
    result[indices_to_keep] = True
    
    return result

def get_all_model_meta_info():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_csv(f"{local_path}/data/openllm_all_model_info_full.csv")

    return df

def get_everything_benchmark_raw():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    with open(f"{local_path}/data/benchmark_data_open_llm_full_no_arc.pkl", "rb") as f:
        results = pickle.load(f)

    return results



def get_official_provider_model_benchmark():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_csv(f"{local_path}/data/official_provider_benchmark.csv")

    return df

def get_HELM_model_benchmark():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    with open(f"{local_path}/data/HELM_benchmark.pkl", "rb") as f:
        results = pickle.load(f)
    
    return results

def random_mask(data_idtor, pct = 0.8):
    
    train_idtor = torch.bernoulli(data_idtor * pct).int()
    test_idtor = data_idtor - train_idtor
    return train_idtor, test_idtor
    

def model_mask(data_idtor):

    train_row_idtor = torch.bernoulli(data_idtor.max(axis=1).values * 0.8).bool()
    train_idtor = torch.zeros_like(data_idtor).int()
    train_idtor[train_row_idtor, :] = data_idtor[train_row_idtor, :]
    
    train_idtor[~train_row_idtor, :],_ = random_mask(data_idtor[~train_row_idtor, :],0.1)
    test_idtor = data_idtor - train_idtor
    
    return train_idtor, test_idtor


def row_mask(data_idtor,custom_train_row):
    
    train_row_idtor = custom_train_row
    
    train_idtor = torch.zeros_like(data_idtor).int()
    data_idtor = data_idtor.int()
    train_idtor[train_row_idtor, :] = data_idtor[train_row_idtor, :]

    train_idtor[~train_row_idtor, :],_ = random_mask(data_idtor[~train_row_idtor, :],0.1)
    test_idtor = data_idtor - train_idtor
    
    return train_idtor, test_idtor

def col_mask(data_idtor,custom_col_row):
    
    train_col_idtor = custom_col_row
    
    train_idtor = torch.zeros_like(data_idtor).int()
    data_idtor = data_idtor.int()
    train_idtor[:, train_col_idtor] = data_idtor[:, train_col_idtor]

    train_idtor[:, ~train_col_idtor],_ = random_mask(data_idtor[:, ~train_col_idtor],0.1)
    test_idtor = data_idtor - train_idtor
    
    return train_idtor, test_idtor


def get_mask_and_data(data_withnan, is_random_row=False, custom_train_row = None, custom_train_col=None):
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    trial = 0
    valid_condition = False
    while not valid_condition:
        if custom_train_col is not None:
            assert custom_train_row is None
            
            train_idtor, test_idtor = col_mask(data_idtor, custom_train_col)


        elif custom_train_row is not None:

            train_idtor, test_idtor = row_mask(data_idtor, custom_train_row)
        elif is_random_row:
            train_idtor, test_idtor = model_mask(data_idtor.int())
        else:
            train_idtor, test_idtor = random_mask(data_idtor)
        
        valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
        print(f"trial {trial} valid condition: {valid_condition}")
        trial += 1
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool()

def keep_90_pct(sel):
    return torch.bernoulli(sel.float() * 0.9).bool()

def random_drop(train_sel):
    # select 90% that I will keep
    train_sel_t = torch.as_tensor(train_sel.values, dtype=torch.bool)
    keep_sel = keep_90_pct(train_sel_t) + keep_90_pct(~train_sel_t)
    return keep_sel, train_sel_t[keep_sel]

def get_helm_benchmark(seed, filter_method = 'random_mask'):
    # with open(f"/lfs/skampere1/0/sttruong/reeval/data/resmat.pkl", "rb") as f:
    results = get_HELM_model_benchmark()
        
    print("loaded")
    all_items = list(results.columns)
    cat1 = [i[1] for i in all_items]
    cat2 = [i[2] for i in all_items]
    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.values, dtype=torch.float)
    if filter_method == 'random_mask':
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, False)
    elif filter_method == 'random_row':
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, True)
    else:
        assert False
    

    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,cat2,model_names)


def attatch_meta(model_names_df, model_meta_info):

    model_meta_info = model_meta_info.rename(columns={"fullname": "model_name"})
    model_meta_info = model_meta_info.drop_duplicates(subset=["model_name"], keep="last")
    merged = pd.merge(
        model_names_df,
        model_meta_info,
        on="model_name",
        how="left"
    )
    return merged




def get_official_provider_benchmark(seed, filter_method = 'random_mask'):
    torch.manual_seed(seed)
    
    all_benchmark_data = get_official_provider_model_benchmark()
    

    model_names = all_benchmark_data['model_name'].tolist()
    model_meta_info = attatch_meta(all_benchmark_data[['model_name']], get_all_model_meta_info())
    is_random_row = None
    sel_train_row = None

    if filter_method == 'date':
        sel_train_row = uploaded_before(model_meta_info, "2024-10-01")
        keep_sel, sel_train_row = random_drop(sel_train_row)
    elif filter_method == 'size':
        sel_train_row = size_smaller(model_meta_info, 30)
        keep_sel, sel_train_row = random_drop(sel_train_row)
    elif filter_method == 'random_row':
        is_random_row = True
    elif filter_method == 'random_mask':
        is_random_row = False
    else:
        assert False

    data_clean = all_benchmark_data.drop('model_name', axis=1)

    # Find binary columns
    binary_cols = []
    for col in data_clean.columns:
        unique_vals = data_clean[col].dropna().unique()
        if all(val in [0.0, 1.0] for val in unique_vals):
            binary_cols.append(col)
    # Keep only binary columns
    data_binary = data_clean[binary_cols]

    # Convert ALL columns to float64 (including boolean columns)
    data_binary = data_binary.astype('float64')

    # Now convert to tensor
    data_withnan = torch.tensor(data_binary.values, dtype=torch.float32)
    if filter_method in ['date','size']:
        data_withnan = data_withnan[keep_sel]

    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_random_row=is_random_row,custom_train_row=sel_train_row)
    # print(sel_train_row.shape)
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (None,None,model_names)


def get_everything_benchmark(seed, filter_method = 'date'):
    torch.manual_seed(seed)

    dataset_dates = {
        "IFEVAL": "2023-11-15",
        "BBH": "2022-10-17",
        "MATH": "2021-11-08",
        "GPQA": "2023-11-20",
        "MUSR": "2024-03-23",
        "MMLU": "2024-11-06",
        "arc_challenge": "2024-11-17"
    }
    results = get_everything_benchmark_raw()
    results_model_name_df = results.reset_index().rename(columns={"index": "model_name"})
    
    model_meta_info = attatch_meta(results_model_name_df[['model_name']], get_all_model_meta_info())
    is_random_row = None
    sel_train_row = None

    if filter_method == 'date':
        sel_train_row = uploaded_before(model_meta_info, "2025-02-26")
        keep_sel, sel_train_row = random_drop(sel_train_row)
    elif filter_method == 'size':
        sel_train_row = size_smaller(model_meta_info, 14)
        keep_sel, sel_train_row = random_drop(sel_train_row)
    elif filter_method == 'random_row':
        is_random_row = True
    elif filter_method == 'random_mask':
        is_random_row = False
    else:
        assert False

    all_items = list(results.columns)
    cat1 = [i[0] for i in all_items]

    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.astype("boolean").astype(float).to_numpy())
    if filter_method in ['date','size']:
        data_withnan = data_withnan[keep_sel]
    
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_random_row=is_random_row, custom_train_row=sel_train_row)
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,None,model_names)


def process_everything_500_sub_data(results):
    benchmarks = ['openllm_math',  'ifeval', 'musr', 'bbh', 'gpqa', 'mmlu_pro']
    col_scenarios = pd.Series([i[0] for i in results.columns])
    keep_col_sel = pd.Series([False for i in results.columns])

    for benchmark in benchmarks:
        train_col_idtor = col_scenarios == benchmark
        train_col_idtor = keep_random_trues(train_col_idtor)
        keep_col_sel = keep_col_sel | train_col_idtor
    results = results.loc[:, keep_col_sel.values]   # keep only relevant columns
    return results
    

def get_everything_benchmark_500_sub_data(seed, filter_method = 'date'):
    torch.manual_seed(seed)

    results = get_everything_benchmark_raw()
    results = process_everything_500_sub_data(results)
    
    
    
    
    results_model_name_df = results.reset_index().rename(columns={"index": "model_name"})
    
    model_meta_info = attatch_meta(results_model_name_df[['model_name']], get_all_model_meta_info())
    is_random_row = None
    sel_train_row = None

    if filter_method == 'date':
        sel_train_row = uploaded_before(model_meta_info, "2025-02-26")
        keep_sel, sel_train_row = random_drop(sel_train_row)
    elif filter_method == 'size':
        sel_train_row = size_smaller(model_meta_info, 14)
        keep_sel, sel_train_row = random_drop(sel_train_row)
    elif filter_method == 'random_row':
        is_random_row = True
    elif filter_method == 'random_mask':
        is_random_row = False
    else:
        assert False

    all_items = list(results.columns)
    cat1 = [i[0] for i in all_items]

    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.astype("boolean").astype(float).to_numpy())
    if filter_method in ['date','size']:
        data_withnan = data_withnan[keep_sel]
    
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_random_row=is_random_row, custom_train_row=sel_train_row)
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,None,model_names)






def create_time_difference_matrix(results, model_meta_info, dataset_dates):
    """
    Create a matrix showing time difference between model upload date and dataset date.
    Positive values mean model was uploaded after dataset creation.
    Negative values mean model was uploaded before dataset creation.
    
    Args:
        results: DataFrame with models as index and (dataset, question) as columns
        model_meta_info: DataFrame with model_name and Upload To Hub Date
        dataset_dates: Dict mapping dataset names to their creation dates
    
    Returns:
        time_diff_matrix: DataFrame with same shape as results, but values are time differences in days
    """
    
    # Convert dataset dates to datetime
    dataset_dates_dt = {}
    for dataset, date_str in dataset_dates.items():
        dataset_dates_dt[dataset] = pd.to_datetime(date_str)
    
    # Convert model upload dates to datetime
    model_meta_info = model_meta_info.copy()
    model_meta_info['Upload To Hub Date'] = pd.to_datetime(model_meta_info['Upload To Hub Date'])
    
    # Create a mapping from model name to upload date
    model_date_mapping = dict(zip(model_meta_info['model_name'], model_meta_info['Upload To Hub Date']))
    
    # STEP 1: Create column series mapping each column to its dataset date (do this once)
    print("Creating column-to-dataset mapping...")
    col_to_dataset_date = pd.Series(index=results.columns, dtype='datetime64[ns]')
    
    for col in results.columns:
        # Extract dataset name from column (assuming format like (dataset, question))
        if isinstance(col, tuple):
            dataset_name = col[0]  # First element of tuple
        else:
            # If column is a string, you might need to parse it differently
            dataset_name = col.split('_')[0] if '_' in col else col
        
        # Map dataset names to match the keys in dataset_dates
        dataset_key = None
        for key in dataset_dates_dt.keys():
            if key.lower() in dataset_name.lower() or dataset_name.lower() in key.lower():
                dataset_key = key
                break
        
        if dataset_key and dataset_key in dataset_dates_dt:
            col_to_dataset_date[col] = dataset_dates_dt[dataset_key]
        else:
            col_to_dataset_date[col] = pd.NaT  # Not a Time
    
    # STEP 2: For each model, create a series of time differences and assign to matrix
    print("Creating time difference matrix...")
    time_diff_matrix = pd.DataFrame(
        index=results.index,
        columns=results.columns,
        dtype=float
    )

    for model_name in results.index:
        if model_name in model_date_mapping:
            model_date = model_date_mapping[model_name]
            
            # Calculate time differences for all columns at once using vectorized operations
            model_date_series = pd.Series(model_date, index=results.columns)
            time_diffs = (model_date_series - col_to_dataset_date).dt.days

            time_diff_matrix.loc[model_name] = time_diffs
        else:

            # If model not found in meta info, set entire row to NaN
            time_diff_matrix.loc[model_name] = np.nan

    return time_diff_matrix

# Usage example with your existing variables:
def get_time_diff_matrix_for_benchmark(seed):
    """
    Modified version of your function that also returns the time difference matrix
    """
    torch.manual_seed(seed)

    dataset_dates = {
        "IFEVAL": "2023-11-15",
        "BBH": "2022-10-17", 
        "MATH": "2021-11-08",
        "GPQA": "2023-11-20",
        "MUSR": "2024-03-23",
        "MMLU": "2024-11-06",
    }

    results = get_everything_benchmark_raw()
    results_model_name_df = results.reset_index().rename(columns={"index": "model_name"})

    model_meta_info = attatch_meta(results_model_name_df[['model_name']], get_all_model_meta_info())
    
    # Create time difference matrix
    time_diff_matrix = create_time_difference_matrix(results, model_meta_info, dataset_dates)
    indicator_time_matrix = (time_diff_matrix >= 0).astype(int)

    # Continue with your existing logic...
    is_random_row = None
    sel_train_row = None
    
    all_items = list(results.columns)
    cat1 = [i[0] for i in all_items]
    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.astype("boolean").astype(float).to_numpy())

    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan)
    
    data_idtor = data_idtor.bool()

    time_diff_matrix = torch.tensor(time_diff_matrix.values, dtype=torch.float64)
    indicator_time_matrix = torch.tensor(indicator_time_matrix.values, dtype=torch.float64)
    
    rmv_row_mask = torch.isnan(time_diff_matrix).any(dim=1)
    # keep value rows
    # time_diff_matrix_withneg1 = time_diff_matrix.nan_to_num(nan=-1.0)
    # time_diff_matrix_idtor = (time_diff_matrix_withneg1 != -1).to(float)
    # data_idtor = time_diff_matrix_idtor & time_diff_matrix_idtor

    time_diff_matrix = time_diff_matrix[~rmv_row_mask]
    indicator_time_matrix = indicator_time_matrix[~rmv_row_mask]
    data_idtor = data_idtor[~rmv_row_mask]
    data_with0 = data_with0[~rmv_row_mask]
    
    idx_i, idx_j = torch.meshgrid(
        torch.arange(data_idtor.shape[0]), 
        torch.arange(data_idtor.shape[1]), 
        indexing='ij'
    )

    
    

    return {
        "x":time_diff_matrix[data_idtor].unsqueeze(1),
        'm':indicator_time_matrix[data_idtor],
        'y':data_with0[data_idtor],
        'idx_i':idx_i[data_idtor],
        'idx_j':idx_j[data_idtor],
        "orig_x":time_diff_matrix,
        "model_names":pd.Series(model_names)[~rmv_row_mask.numpy()],
        "cat1":cat1,
        
    }
    
    # return data_withneg1, data_with0, data_idtor.bool(), (cat1, None, model_names), time_diff_matrix, indicator_time_matrix



def get_everything_benchmark_1_to_2(seed,train_dataset_id,test_dataset_id):
    torch.manual_seed(seed)
    benchmarks = ['openllm_math', 'ifeval', 'musr', 'bbh', 'gpqa', 'mmlu_pro']

    train_benchmark = benchmarks[train_dataset_id]
    test_benchmark = benchmarks[test_dataset_id]
    
    results = get_everything_benchmark_raw()
    results = process_everything_500_sub_data(results)
    col_scenarios = pd.Series([i[0] for i in results.columns])
    train_col_idtor = col_scenarios == train_benchmark
    test_col_idtor = col_scenarios == test_benchmark
    keep_mask = train_col_idtor | test_col_idtor
    
    train_col_idtor = train_col_idtor[keep_mask].to_numpy()
    test_col_idtor = test_col_idtor[keep_mask].to_numpy()
    results = results.loc[:, keep_mask.values]   # keep only relevant columns
    
    # ---- Remove rows where all entries are NaN ----
    non_nan_mask = ~results.isna().all(axis=1)
    results = results.loc[non_nan_mask]
        
    model_names = list(results.index)
    all_items = list(results.columns)
    cat1 = [i[0] for i in all_items]
    data_withnan = torch.tensor(results.astype("boolean").astype(float).to_numpy())

    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, custom_train_col=train_col_idtor)
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,None,model_names)




# print("loading")
# pd.read_csv("/lfs/skampere1/0/sttruong/reeval/calibration/all_benchmarks_joined.csv")
# print("loading")
# get_new_benchmark(0)
# print("loading")
# load_old_benchmark(0)

def get_everything_data(seed, is_adap_testing=False):
    torch.manual_seed(seed)
    print("loading pd")
    results = pd.read_parquet("/lfs/skampere1/0/sttruong/reeval/data/benchmark_data_open_llm.parquet")
    print("loaded pd")

    print("loaded")
    all_items = list(results.columns)
    cat1 = [i[0] for i in all_items]

    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.values, dtype=torch.float)
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_adap_testing)
    

    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,None,model_names)



def get_everything_data_sk2(seed, is_adap_testing=False):
    torch.manual_seed(seed)
    print("loading pd")
    results = pd.read_parquet("/lfs/skampere2/0/sttruong/reeval/data/benchmark_data_open_llm.parquet")
    print(results.shape)
    print("loaded pd")

    print("loaded")
    all_items = list(results.columns)
    cat1 = [i[0] for i in all_items]

    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.astype("boolean").astype(float).to_numpy())
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_adap_testing)
    

    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,None,model_names)

# df = get_all_model_meta_info()


# # your dataset date mapping
# dataset_dates = {
#     "IFEVAL": "2023-11-15",
#     "BBH": "2022-10-17",
#     "MATH": "2021-11-08",
#     "GPQA": "2023-11-20",
#     "MUSR": "2024-03-23",
#     "MMLU": "2024-11-06",
#     "arc_challenge": "2024-11-17"
# }

# # convert to datetime
# dataset_dates = {k: pd.to_datetime(v) for k,v in dataset_dates.items()}

# # ensure Upload To Hub Date is datetime
# df["Upload To Hub Date"] = pd.to_datetime(df["Upload To Hub Date"], errors="coerce")

# # build result matrix
# result = pd.DataFrame(index=df["fullname"])


# # # merge the upload date into result
# # result = result.drop(columns=["Upload To Hub Date"], errors="ignore") \
# #                .assign(upload_date=df["Upload To Hub Date"].values)

# # build new index: "<model>_<date>"
# result.index = [
#     f"{name}_{date.date() if pd.notna(date) else 'NaT'}"
#     for name, date in zip(df["fullname"], df["Upload To Hub Date"])
# ]


# for dataset, d_time in dataset_dates.items():
#     col = f"{dataset}_{d_time.date()}"
#     series = df["Upload To Hub Date"].apply(
#         lambda x: np.nan if pd.isna(x) else (1 if x > d_time else 0)
#     )
#     # assign by position (ignore index) to avoid alignment
#     result[col] = series.to_numpy()

# # result["Upload To Hub Date"]=df["Upload To Hub Date"].to_numpy()
# result.to_csv("model_dataset_date_compare.csv")


# get_everything_benchmark_1_to_2(0,1,2)
