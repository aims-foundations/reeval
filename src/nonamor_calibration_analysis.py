import os
import torch
import pandas as pd
from tqdm import tqdm
from utils import (
    goodness_of_fit_1PL_plot, 
    theta_corr_ctt_plot, 
    theta_corr_helm_plot,
    error_bar_plot_single,
    DATASETS,
    goodness_of_fit_1PL
)

if __name__ == "__main__":
    input_dir = '../data/nonamor_calibration'
    plot_dir = f'../plot/nonamor_calibration'
    os.makedirs(plot_dir, exist_ok=True)
    
    gof_means, gof_stds = [], []
    corr_ctt_means, corr_ctt_stds = [], []
    corr_helm_means, corr_helm_stds = [], []
    plugin_gof_train_means, plugin_gof_test_means = [], []
    amor_gof_train_means, amor_gof_test_means = [], []
    for dataset in tqdm(DATASETS):
        print(f"Processing {dataset}")
        y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
        theta_hat = pd.read_csv(f'{input_dir}/{dataset}/nonamor_theta.csv')['theta'].values
        z_hat = pd.read_csv(f'{input_dir}/{dataset}/nonamor_z.csv')['z'].values
        
        gof_mean, gof_std = goodness_of_fit_1PL_plot(
            z=torch.tensor(z_hat, dtype=torch.float32),
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            plot_path=f"{plot_dir}/goodness_of_fit_{dataset}",
        )
        gof_means.append(gof_mean)
        gof_stds.append(gof_std)
        
        corr_ctt_mean, corr_ctt_std = theta_corr_ctt_plot(
            theta=theta_hat,
            y=y,
            plot_path=f"{plot_dir}/theta_corr_ctt_{dataset}",
        )
        corr_ctt_means.append(corr_ctt_mean)
        corr_ctt_stds.append(corr_ctt_std)
        
        if dataset != "airbench":
            corr_helm_mean, corr_helm_std = theta_corr_helm_plot(
                theta=theta_hat,
                dataset=dataset,
                plot_path=f"{plot_dir}/theta_corr_helm_{dataset}",
            )
            corr_helm_means.append(corr_helm_mean)
            corr_helm_stds.append(corr_helm_std)

        plugin_train_indices = pd.read_csv(f'../data/plugin_regression/{dataset}/train_0.csv')['index'].values
        plugin_test_indices = pd.read_csv(f'../data/plugin_regression/{dataset}/test_0.csv')['index'].values
        
        plugin_gof_train_mean, _ = goodness_of_fit_1PL(
            z=torch.tensor(z_hat[plugin_train_indices], dtype=torch.float32),
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            y=torch.tensor(y[:, plugin_train_indices], dtype=torch.float32),
        )
        plugin_gof_train_means.append(plugin_gof_train_mean)
        
        plugin_gof_test_mean, _ = goodness_of_fit_1PL(
            z=torch.tensor(z_hat[plugin_test_indices], dtype=torch.float32),
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            y=torch.tensor(y[:, plugin_test_indices], dtype=torch.float32),
        )
        plugin_gof_test_means.append(plugin_gof_test_mean)
        
        amor_train_indices = pd.read_csv(f'../data/amor_calibration/{dataset}/z_train_0.csv')['index'].values
        amor_test_indices = pd.read_csv(f'../data/amor_calibration/{dataset}/z_test_0.csv')['index'].values
        
        amor_gof_train_mean, _ = goodness_of_fit_1PL(
            z=torch.tensor(z_hat[amor_train_indices], dtype=torch.float32),
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            y=torch.tensor(y[:, amor_train_indices], dtype=torch.float32),
        )
        amor_gof_train_means.append(amor_gof_train_mean)
        
        amor_gof_test_mean, _ = goodness_of_fit_1PL(
            z=torch.tensor(z_hat[amor_test_indices], dtype=torch.float32),
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            y=torch.tensor(y[:, amor_test_indices], dtype=torch.float32),
        )
        amor_gof_test_means.append(amor_gof_test_mean)
    
    plugin_gof_df_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': plugin_gof_train_means,
    })
    plugin_gof_df_train.to_csv(f'{plot_dir}/nonamor4plugin_gof_train.csv', index=False)
    
    plugin_gof_df_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': plugin_gof_test_means,
    })
    plugin_gof_df_test.to_csv(f'{plot_dir}/nonamor4plugin_gof_test.csv', index=False)
    
    amor_gof_df_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': amor_gof_train_means,
    })
    amor_gof_df_train.to_csv(f'{plot_dir}/nonamor4amor_gof_train.csv', index=False)
    
    amor_gof_df_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': amor_gof_test_means,
    })
    amor_gof_df_test.to_csv(f'{plot_dir}/nonamor4amor_gof_test.csv', index=False)
    
    gof_df = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': gof_means,
        'gof_stds': gof_stds
    })
    gof_df.to_csv(f'{plot_dir}/nonamor_calibration_gof.csv', index=False)
    
    ctt_df = pd.DataFrame({
        'datasets': DATASETS,
        'corr_ctt_means': corr_ctt_means,
        'corr_ctt_stds': corr_ctt_stds
    })
    ctt_df.to_csv(f'{plot_dir}/nonamor_calibration_corr_ctt.csv', index=False)
    
    helm_df = pd.DataFrame({
        'datasets': [d for d in DATASETS if d != "airbench"],
        'corr_helm_means': corr_helm_means,
        'corr_helm_stds': corr_helm_stds
    })
    helm_df.to_csv(f'{plot_dir}/nonamor_calibration_corr_helm.csv', index=False)

    error_bar_plot_single(
        datasets=DATASETS,
        means=gof_means,
        stds=gof_stds,
        plot_path=f"{plot_dir}/nonamor_calibration_summarize_gof",
        xlabel=r"Goodness of Fit",
    )
    
    error_bar_plot_single(
        datasets=DATASETS,
        means=corr_ctt_means,
        stds=corr_ctt_stds,
        plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_ctt",
        xlabel=r"$\theta$ correlation with CTT",
    )
    
    error_bar_plot_single(
        datasets=[d for d in DATASETS if d != "airbench"],
        means=corr_helm_means,
        stds=corr_helm_stds,
        plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_helm",
        xlabel=r"$\theta$ correlation with HELM",
    )
    