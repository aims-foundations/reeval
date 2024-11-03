import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    
    
    # valid_datasets_df = pd.DataFrame(valid_datasets, columns=["dataset"])
    # valid_datasets_df.to_csv(f"{output_dir}/valid_datasets.csv", index=False)
    
    # combined_matrix_train.fillna(-1, inplace=True)
    # print(combined_matrix_train.shape)
    # combined_matrix_train.to_csv(f"{output_dir}/combined_matrix_train.csv")
    
    # combined_matrix_test.fillna(-1, inplace=True)
    # print(combined_matrix_test.shape)
    # combined_matrix_test.to_csv(f"{output_dir}/combined_matrix_test.csv")
    
    valid_datasets = pd.read_csv(f"{output_dir}/valid_datasets.csv").values.flatten()
    combined_matrix_train = pd.read_csv(f"{output_dir}/combined_matrix_train.csv", index_col=0)
    combined_matrix_test = pd.read_csv(f"{output_dir}/combined_matrix_test.csv", index_col=0)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].plot(x, y1)
    axs[0, 0].set_title('Sine')

    axs[0, 1].plot(x, y2)
    axs[0, 1].set_title('Cosine')

    axs[0, 2].plot(x, y3)
    axs[0, 2].set_title('Tangent')

    axs[1, 0].plot(x, y4)
    axs[1, 0].set_title('Exponential')

    axs[1, 1].plot(x, y5)
    axs[1, 1].set_title('Logarithm')

    axs[1, 2].plot(x, y6)
    axs[1, 2].set_title('Square Root')