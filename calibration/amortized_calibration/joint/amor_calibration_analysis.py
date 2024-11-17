import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import (
    goodness_of_fit_1PL, 
    theta_corr_ctt, 
    theta_corr_helm,
    error_bar_plot_single,
    error_bar_plot_double, 
    amorz_corr_nonamorz,
    DATASETS,
)

if __name__ == "__main__":
    input_dir = '../data/amor_calibration'
    plot_dir = f'../plot/amor_calibration'
    os.makedirs(plot_dir, exist_ok=True)
    
    dataset_gof_train_means, dataset_gof_train_stds = [], []
    dataset_gof_test_means, dataset_gof_test_stds = [], []
    dataset_theta_corr_ctt_means, dataset_theta_corr_ctt_stds = [], []
    dataset_theta_corr_helm_means, dataset_theta_corr_helm_stds = [], []
    dataset_z_corr_train_means, dataset_z_corr_train_stds = [], []
    dataset_z_corr_test_means, dataset_z_corr_test_stds = [], []
    single_gof_train_means, single_gof_test_means = [], []
    
    for dataset in tqdm(DATASETS):
        print(f"Processing {dataset}")
        gof_train_means, gof_test_means = [], []
        theta_corr_ctt_means = []
        theta_corr_helm_means = []
        z_corr_train_means, z_corr_test_means = [], []
        
        for i in range(10):
            y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
            theta_train = pd.read_csv(f'{input_dir}/{dataset}/theta_{i}.csv')['theta'].values
            df_z_train = pd.read_csv(f'{input_dir}/{dataset}/z_train_{i}.csv')
            train_indices = df_z_train['index'].values
            z_train = df_z_train['z_pred'].values
            df_z_test = pd.read_csv(f'{input_dir}/{dataset}/z_test_{i}.csv')
            test_indices = df_z_test['index'].values
            z_test = df_z_test['z_pred'].values
            nonamor_z = pd.read_csv(f'../data/nonamor_calibration/{dataset}/nonamor_z.csv')['z'].values
            
            gof_train_mean, _ = goodness_of_fit_1PL(
                z=torch.tensor(z_train, dtype=torch.float32),
                theta=torch.tensor(theta_train, dtype=torch.float32),
                y=torch.tensor(y[:, train_indices], dtype=torch.float32),
            )
            gof_train_means.append(gof_train_mean)
            
            gof_test_mean, _ = goodness_of_fit_1PL(
                z=torch.tensor(z_test, dtype=torch.float32),
                theta=torch.tensor(theta_train, dtype=torch.float32),
                y=torch.tensor(y[:, test_indices], dtype=torch.float32),
            )
            gof_test_means.append(gof_test_mean)
            
            theta_corr_ctt_mean, _, _ = theta_corr_ctt(
                theta=theta_train,
                y=y,
            )
            theta_corr_ctt_means.append(theta_corr_ctt_mean)
            
            if dataset != "airbench":
                theta_corr_helm_mean, _, _ = theta_corr_helm(
                    theta=theta_train,
                    dataset=dataset,
                )
                theta_corr_helm_means.append(theta_corr_helm_mean)
            
            z_corr_train_mean = amorz_corr_nonamorz(
                z_amor=z_train,
                z_nonamor=nonamor_z[train_indices],
            )
            z_corr_train_means.append(z_corr_train_mean)
            
            z_corr_test_mean = amorz_corr_nonamorz(
                z_amor=z_test,
                z_nonamor=nonamor_z[test_indices],
            )
            z_corr_test_means.append(z_corr_test_mean)

            if i == 0:
                single_gof_train_means.append(gof_train_mean)
                single_gof_test_means.append(gof_test_mean)
                
        dataset_gof_train_means.append(np.mean(gof_train_means))
        dataset_gof_test_means.append(np.mean(gof_test_means))
        dataset_theta_corr_ctt_means.append(np.mean(theta_corr_ctt_means))
        if dataset != "airbench":
            dataset_theta_corr_helm_means.append(np.mean(theta_corr_helm_means))
        dataset_z_corr_train_means.append(np.mean(z_corr_train_means))
        dataset_z_corr_test_means.append(np.mean(z_corr_test_means))
        
        dataset_gof_train_stds.append(np.std(gof_train_means))
        dataset_gof_test_stds.append(np.std(gof_test_means))
        dataset_theta_corr_ctt_stds.append(np.std(theta_corr_ctt_means))
        if dataset != "airbench":
            dataset_theta_corr_helm_stds.append(np.std(theta_corr_helm_means))
        dataset_z_corr_train_stds.append(np.std(z_corr_train_means))
        dataset_z_corr_test_stds.append(np.std(z_corr_test_means))
    
    single_df_gof_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': single_gof_train_means,
    })
    single_df_gof_train.to_csv(f'{plot_dir}/amor_single_gof_train.csv', index=False)
    
    single_df_gof_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': single_gof_test_means,
    })
    single_df_gof_test.to_csv(f'{plot_dir}/amor_single_gof_test.csv', index=False)
    
    gof_df_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': dataset_gof_train_means,
        'gof_stds': dataset_gof_train_stds
    })
    gof_df_train.to_csv(f'{plot_dir}/amor_calibration_gof_train.csv', index=False)
    
    gof_df_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': dataset_gof_test_means,
        'gof_stds': dataset_gof_test_stds
    })
    gof_df_test.to_csv(f'{plot_dir}/amor_calibration_gof_test.csv', index=False)
    
    ctt_df = pd.DataFrame({
        'datasets': DATASETS,
        'corr_ctt_means': dataset_theta_corr_ctt_means,
        'corr_ctt_stds': dataset_theta_corr_ctt_stds
    })
    ctt_df.to_csv(f'{plot_dir}/amor_calibration_corr_ctt.csv', index=False)
    
    helm_df = pd.DataFrame({
        'datasets': [d for d in DATASETS if d != "airbench"],
        'corr_helm_means': dataset_theta_corr_helm_means,
        'corr_helm_stds': dataset_theta_corr_helm_stds
    })
    helm_df.to_csv(f'{plot_dir}/amor_calibration_corr_helm.csv', index=False)
    
    error_bar_plot_double(
        datasets=DATASETS, 
        means_train=dataset_gof_train_means,
        stds_train=dataset_gof_train_stds,
        means_test=dataset_gof_test_means,
        stds_test=dataset_gof_test_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_gof",
        xlabel=r"Goodness of Fit",
    )   
    
    error_bar_plot_single(
        datasets=DATASETS,
        means=dataset_theta_corr_ctt_means,
        stds=dataset_theta_corr_ctt_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_theta_corr_ctt",
        xlabel=r"$\theta$ correlation with CTT",
    )
    
    error_bar_plot_single(
        datasets=[d for d in DATASETS if d != "airbench"],
        means=dataset_theta_corr_helm_means,
        stds=dataset_theta_corr_helm_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_theta_corr_helm",
        xlabel=r"$\theta$ correlation with HELM",
    )
    
    error_bar_plot_double(
        datasets=DATASETS, 
        means_train=dataset_z_corr_train_means,
        stds_train=dataset_z_corr_train_stds,
        means_test=dataset_z_corr_test_means,
        stds_test=dataset_z_corr_test_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_z_corr",
        xlabel=r"correlation of $z$",
    )   
    