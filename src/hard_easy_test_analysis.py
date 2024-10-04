
import os
import numpy as np
import pandas as pd
import torch
from utils import DATASETS, plot_hard_easy

if __name__ == "__main__":
    for dataset in DATASETS:
        df = pd.read_csv(f'../data/hard_easy_test/{dataset}/hard_easy_test.csv')
        theta_hats_all = df['theta_hat'].values
        y_means_all = df['y_mean'].values

        response_matrix = pd.read_csv(
            f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0
        ).values
        theta = pd.read_csv(
            f'../data/nonamor_calibration/{dataset}/nonamor_theta.csv'
        )["theta"].values
        z = pd.read_csv(
            f'../data/nonamor_calibration/{dataset}/nonamor_z.csv'
        )["z"].values

        # rows in y with more than 500 non -1 values
        valid_rows_mask = (response_matrix != -1).sum(axis=1) > 500
        filtered_theta = theta[valid_rows_mask]
        filtered_response_matrix = response_matrix[valid_rows_mask]
        # theta value closest to zero
        min_index = np.argmin(np.abs(filtered_theta))
        
        y = torch.tensor(filtered_response_matrix[min_index], dtype=torch.float32)
        valid_cols_mask = y != -1
        y = y[valid_cols_mask]
        z = torch.tensor(z[valid_cols_mask], dtype=torch.float32)
        theta = torch.tensor(filtered_theta[min_index], dtype=torch.float32)

        save_dir = f'../plot/hard_easy_test'
        os.makedirs(save_dir, exist_ok=True)
        plot_hard_easy(theta_hats_all, y_means_all, theta, y, f'{save_dir}/easy_hard_{dataset}.png')