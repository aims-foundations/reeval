import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from utils import (
    set_seed, 
    DATASETS,
    item_response_fn_1PL_multi_dim, 
    goodness_of_fit_1PL_multi_dim_plot, 
    error_bar_plot_double,
)

if __name__ == "__main__":
    set_seed(42)
    output_dir = f'../data/mle_multi_dim_amor_theta'
    plot_dir = f'../plot/mle_multi_dim_amor_theta'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    model_id_df = pd.read_csv('configs/model_id_ver1.csv')
    valid_model_names = model_id_df['model_names_reeval'].values
    feat_matrix = model_id_df[['Model Size (B)', 'Pretraining Data Size (T)', 'FLOPs (1E21)']].values
    # feat_matrix = (feat_matrix - feat_matrix.mean(axis=0)) / feat_matrix.std(axis=0)
    
    valid_model_names_test = ['meta_llama-2-70b', 'meta_llama-2-7b', 'meta_llama-2-13b']
    valid_model_indices = [np.where(valid_model_names == name)[0][0] for name in valid_model_names_test]
    feat_matrix_test = feat_matrix[valid_model_indices]
    valid_model_names_train = [name for name in valid_model_names if name not in valid_model_names_test]
    valid_model_indices = [np.where(valid_model_names == name)[0][0] for name in valid_model_names_train]
    feat_matrix_train = feat_matrix[valid_model_indices]
    
    valid_datasets = []
    combined_matrix_train = pd.DataFrame()
    combined_matrix_test = pd.DataFrame()
    for dataset in DATASETS:
        matrix = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
        
        filtered_matrix_train = matrix[matrix.index.isin(valid_model_names_train)]
        # print(f"Dataset: {dataset}, left model num: {filtered_matrix_train.shape[0]}, left models: {filtered_matrix_train.index.tolist()}")
        if not filtered_matrix_train.empty:
            valid_datasets.append(dataset)
            if combined_matrix_train.empty:
                combined_matrix_train = filtered_matrix_train
            else:
                combined_matrix_train = combined_matrix_train.join(filtered_matrix_train, how='outer', rsuffix='_dup')
                
        filtered_matrix_test = matrix[matrix.index.isin(valid_model_names_test)]
        if not filtered_matrix_test.empty:
            if combined_matrix_test.empty:
                combined_matrix_test = filtered_matrix_test
            else:
                combined_matrix_test = combined_matrix_test.join(filtered_matrix_test, how='outer', rsuffix='_dup')
    
    valid_datasets_df = pd.DataFrame(valid_datasets, columns=["dataset"])
    valid_datasets_df.to_csv(f"{output_dir}/valid_datasets.csv", index=False)
    
    combined_matrix_train.fillna(-1, inplace=True)
    print(combined_matrix_train.shape)
    combined_matrix_train.to_csv(f"{output_dir}/combined_matrix_train.csv")
    
    combined_matrix_test.fillna(-1, inplace=True)
    print(combined_matrix_test.shape)
    combined_matrix_test.to_csv(f"{output_dir}/combined_matrix_test.csv")