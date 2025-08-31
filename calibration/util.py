import pandas as pd
import numpy as np
import torch
import pickle
from huggingface_hub import snapshot_download

import warnings
warnings.filterwarnings("ignore")

def get_all_model_meta_info():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_csv(f"{local_path}/data/openllm_all_model_info.csv")

    return df

def get_all_model_benchmark():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_parquet(f"{local_path}/benchmark_data_open_llm.parquet")

    return df

def get_official_provider_model_benchmark():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    df = pd.read_parquet(f"{local_path}/data/official_provider_benchmark.csv")

    return df

def get_HELM_model_benchmark():
    local_path = snapshot_download(
        repo_id="stair-lab/reeval_llm_leaderbord", repo_type="dataset"
    )
    with open(f"{local_path}/data/resmat.pkl", "rb") as f:
        results = pickle.load(f)
    
    return results


def get_mask_and_data(data_withnan, is_adap_testing=False):
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    trial = 0
    valid_condition = False

    if is_adap_testing:
        idx_split = int(data_withneg1.shape[0] * 0.8)
        train_idtor = data_idtor[:idx_split,:]
        test_idtor = data_idtor[idx_split:,:]    
    else:
        while not valid_condition:

            train_idtor = torch.bernoulli(data_idtor * 0.8).int()
            test_idtor = data_idtor - train_idtor
            valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
            print(f"trial {trial} valid condition: {valid_condition}")
            trial += 1

    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool()

def load_old_benchmark(seed, is_adap_testing=False):
    with open(f"/lfs/skampere1/0/sttruong/reeval/data/resmat.pkl", "rb") as f:
        results = pickle.load(f)
        
    print("loaded")
    all_items = list(results.columns)
    cat1 = [i[1] for i in all_items]
    cat2 = [i[2] for i in all_items]
    model_names = list(results.index)

    torch.manual_seed(seed)
    data_withnan = torch.tensor(results.values, dtype=torch.float)
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_adap_testing)
    

    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (cat1,cat2,model_names)


def get_new_benchmark(seed, is_adap_testing=False):
    torch.manual_seed(seed)
    print("loading pd")
    all_benchmark_data = pd.read_csv("/lfs/skampere1/0/sttruong/reeval/calibration/grab_dataset/data/all_benchmarks_joined.csv")
    print("loaded pd")

    model_names = all_benchmark_data['model_name'].tolist()
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
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_mask_and_data(data_withnan, is_adap_testing)
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), (None,None,model_names)
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

