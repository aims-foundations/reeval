import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from utils import goodness_of_fit_1PL, error_bar_plot_double, DATASETS
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    input_dir = '../data/plugin_regression'
    plot_dir = f'../plot/plugin_regression'
    os.makedirs(plot_dir, exist_ok=True)
    
    dataset_gof_train_means, dataset_gof_train_stds = [], []
    dataset_gof_test_means, dataset_gof_test_stds = [], []
    dataset_train_mse_means, dataset_train_mse_stds = [], []
    dataset_test_mse_means, dataset_test_mse_stds = [], []
    dataset_baseline_train_mse_means, dataset_baseline_train_mse_stds = [], []
    dataset_baseline_test_mse_means, dataset_baseline_test_mse_stds = [], []
    single_gof_train_means, single_gof_test_means = [], []
    
    for dataset in tqdm(DATASETS):
        gof_train_means, gof_test_means = [], []
        train_mses, test_mses = [], []
        baseline_train_mses, baseline_test_mses = [], []
        
        for i in range(10):
            y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
            df_train = pd.read_csv(f'{input_dir}/{dataset}/train_{i}.csv')
            train_indices = df_train['index'].values
            z_train_true = df_train['z_true'].values
            z_train_pred = df_train['z_pred'].values
            df_test = pd.read_csv(f'{input_dir}/{dataset}/test_{i}.csv')
            test_indices = df_test['index'].values
            z_test_true = df_test['z_true'].values
            z_test_pred = df_test['z_pred'].values
            theta = pd.read_csv(f'../data/nonamor_calibration/{dataset}/nonamor_theta.csv')['theta'].values
            
            gof_train_mean, _ = goodness_of_fit_1PL(
                z=torch.tensor(z_train_pred, dtype=torch.float32),
                theta=torch.tensor(theta, dtype=torch.float32),
                y=torch.tensor(y[:, train_indices], dtype=torch.float32),
            )
            gof_train_means.append(gof_train_mean)
            
            gof_test_mean, _ = goodness_of_fit_1PL(
                z=torch.tensor(z_test_pred, dtype=torch.float32),
                theta=torch.tensor(theta, dtype=torch.float32),
                y=torch.tensor(y[:, test_indices], dtype=torch.float32),
            )
            gof_test_means.append(gof_test_mean)
            
            train_mse = mean_squared_error(z_train_true, z_train_pred)
            test_mse = mean_squared_error(z_test_true, z_test_pred)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            z_baseline_pred = np.mean(z_train_true)
            baseline_train_mse = mean_squared_error(
                z_train_true,
                np.repeat(z_baseline_pred, len(z_train_true))
            )
            baseline_test_mse = mean_squared_error(
                z_test_true,
                np.repeat(z_baseline_pred, len(z_test_true))
            )
            baseline_train_mses.append(baseline_train_mse)
            baseline_test_mses.append(baseline_test_mse)
            
            if i == 0:
                single_gof_train_means.append(gof_train_mean)
                single_gof_test_means.append(gof_test_mean)
                
        dataset_gof_train_means.append(np.mean(gof_train_means))
        dataset_gof_test_means.append(np.mean(gof_test_means))
        dataset_train_mse_means.append(np.mean(train_mses))
        dataset_test_mse_means.append(np.mean(test_mses))
        dataset_baseline_train_mse_means.append(np.mean(baseline_train_mses))
        dataset_baseline_test_mse_means.append(np.mean(baseline_test_mses))
        
        dataset_gof_train_stds.append(np.std(gof_train_means))
        dataset_gof_test_stds.append(np.std(gof_test_means))
        dataset_train_mse_stds.append(np.std(train_mses))
        dataset_test_mse_stds.append(np.std(test_mses))
        dataset_baseline_train_mse_stds.append(np.std(baseline_train_mses))
        dataset_baseline_test_mse_stds.append(np.std(baseline_test_mses))
    
    
    single_df_gof_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': single_gof_train_means,
    })
    single_df_gof_train.to_csv(f'{plot_dir}/plugin_single_gof_train.csv', index=False)
    
    single_df_gof_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': single_gof_test_means,
    })
    single_df_gof_test.to_csv(f'{plot_dir}/plugin_single_gof_test.csv', index=False)
    
    gof_df_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': dataset_gof_train_means,
        'gof_stds': dataset_gof_train_stds
    })
    gof_df_train.to_csv(f'{plot_dir}/plugin_regression_gof_train.csv', index=False)
    
    gof_df_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': dataset_gof_test_means,
        'gof_stds': dataset_gof_test_stds
    })
    gof_df_test.to_csv(f'{plot_dir}/plugin_regression_gof_test.csv', index=False)
    
    mse_train_df = pd.DataFrame({
        'datasets': DATASETS,
        'mse_means': dataset_train_mse_means,
        'mse_stds': dataset_train_mse_stds
    })
    mse_train_df.to_csv(f'{plot_dir}/plugin_regression_mse_train.csv', index=False)
    
    mse_test_df = pd.DataFrame({
        'datasets': DATASETS,
        'mse_means': dataset_test_mse_means,
        'mse_stds': dataset_test_mse_stds
    })
    mse_test_df.to_csv(f'{plot_dir}/plugin_regression_mse_test.csv', index=False)
    
    error_bar_plot_double(
        datasets=DATASETS, 
        means_train=dataset_gof_train_means,
        stds_train=dataset_gof_train_stds,
        means_test=dataset_gof_test_means,
        stds_test=dataset_gof_test_stds,
        plot_path=f"{plot_dir}/plugin_regression_summarize_gof",
        xlabel=r"Goodness of Fit",
    )   
    
    # dataset_train_mse_means = pd.read_csv(f'{plot_dir}/plugin_regression_mse_train.csv')['mse_means'].values
    # dataset_train_mse_stds = pd.read_csv(f'{plot_dir}/plugin_regression_mse_train.csv')['mse_stds'].values
    # dataset_test_mse_means = pd.read_csv(f'{plot_dir}/plugin_regression_mse_test.csv')['mse_means'].values
    # dataset_test_mse_stds = pd.read_csv(f'{plot_dir}/plugin_regression_mse_test.csv')['mse_stds'].values
    
    error_bar_plot_double(
        datasets=DATASETS, 
        means_train=dataset_train_mse_means,
        stds_train=dataset_train_mse_stds,
        means_test=dataset_test_mse_means,
        stds_test=dataset_test_mse_stds,
        plot_path=f"{plot_dir}/plugin_regression_summarize_mse",
        xlabel=r"MSE",
        xlim_upper=10,
        average_line=True,
    )
    
    error_bar_plot_double(
        datasets=DATASETS, 
        means_train=dataset_baseline_train_mse_means,
        stds_train=dataset_baseline_train_mse_stds,
        means_test=dataset_baseline_test_mse_means,
        stds_test=dataset_baseline_test_mse_stds,
        plot_path=f"{plot_dir}/plugin_regression_summarize_baseline_mse",
        xlabel=r"Baseline MSE",
        xlim_upper=10,
    )
    