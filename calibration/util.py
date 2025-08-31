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


def get_all_model_meta_info():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_csv(f"{local_path}/data/openllm_all_model_info.csv")

    return df

def get_everything_benchmark():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_parquet(f"{local_path}/benchmark_data_open_llm.parquet")

    return df

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
    train_idtor[train_row_idtor, :] = data_idtor[train_row_idtor, :]
    
    train_idtor[~train_row_idtor, :],_ = random_mask(data_idtor[~train_row_idtor, :],0.1)
    test_idtor = data_idtor - train_idtor
    
    return train_idtor, test_idtor


def get_mask_and_data(data_withnan, is_random_row=False, custom_train_row = None):
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    trial = 0
    valid_condition = False
    while not valid_condition:

        if custom_train_row:

            train_idtor, test_idtor = row_mask(data_idtor, custom_train_row)
        elif is_random_row:
            train_idtor, test_idtor = model_mask(data_idtor.int())
        else:
            train_idtor, test_idtor = random_mask(data_idtor)
        
        valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
        print(f"trial {trial} valid condition: {valid_condition}")
        trial += 1
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool()




def load_old_benchmark(seed, is_adap_testing=False):
    # with open(f"/lfs/skampere1/0/sttruong/reeval/data/resmat.pkl", "rb") as f:
    results = get_HELM_model_benchmark()
        
    print("loaded")
    all_items = list(results.columns)
    cat1 = [i[1] for i in all_items]
    cat2 = [i[2] for i in all_items]
    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.values, dtype=torch.float)
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_adap_testing)
    

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




def get_new_benchmark(seed,filter_method = 'random_mask'):
    torch.manual_seed(seed)
    
    all_benchmark_data = get_official_provider_model_benchmark()
    

    model_names = all_benchmark_data['model_name'].tolist()
    model_meta_info = attatch_meta(all_benchmark_data[['model_name']], get_all_model_meta_info())
    is_random_row = None
    sel_train_row = None
    if filter_method == 'date':
        sel_train_row = uploaded_before(model_meta_info, "2024-10-01")
    elif filter_method == 'size':
        sel_train_row = size_smaller(model_meta_info, 30)
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
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_random_row=is_random_row,custom_train_row=sel_train_row)
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (None,None,model_names)


def get_everything_benchmark(seed, filter_method = 'date'):
    torch.manual_seed(seed)

    results = get_everything_benchmark()
    results_model_name_df = results.reset_index().rename(columns={"index": "model_name"})
    
    model_meta_info = attatch_meta(results_model_name_df[['model_name']], get_all_model_meta_info())
    
    is_random_row = None
    sel_train_row = None
    
    if filter_method == 'date':
        sel_train_row = uploaded_before(model_meta_info, "2025-02-26")
    elif filter_method == 'size':
        sel_train_row = size_smaller(model_meta_info, 14)
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
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_random_row=is_random_row, custom_train_row=sel_train_row)
    

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
    
    breakpoint()
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
