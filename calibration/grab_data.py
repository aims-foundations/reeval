import pandas as pd
import numpy as np
import torch
import pickle
from torchmetrics import AUROC
auroc = AUROC(task="binary")

import warnings
warnings.filterwarnings("ignore")


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
    all_benchmark_data = pd.read_csv("/lfs/skampere1/0/sttruong/reeval/calibration/all_benchmarks_joined.csv")
    
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
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool(), None

