import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import (
    goodness_of_fit_1PL, 
    theta_corr_ctt, 
    error_bar_plot_single,
    error_bar_plot_double, 
    amorz_corr_nonamorz,
)

if __name__ == "__main__":
    input_dir = '../data/amor_calibration'
    # datasets = [f for f in os.listdir(input_dir)]
    datasets = ['synthetic_efficiency', 'wikifact', 'entity_data_imputation', 'commonsense', 'quac', 'imdb', 'bbq', 'math', 'twitter_aae', 'truthful_qa', 'legal_support', 'boolq', 'narrative_qa', 'real_toxicity_prompts', 'bold', 'gsm', 'babi_qa', 'summarization_xsum', 'synthetic_reasoning_natural', 'dyck_language_np3', 'lsat_qa', 'raft', 'code', 'entity_matching', 'synthetic_reasoning', 'mmlu', 'airbench', 'civil_comments']
    
    plot_dir = f'../plot/amor_calibration'
    os.makedirs(plot_dir, exist_ok=True)
    
    dataset_gof_train_means, dataset_gof_train_stds = [], []
    dataset_gof_test_means, dataset_gof_test_stds = [], []
    dataset_theta_corr_ctt_means, dataset_theta_corr_ctt_stds = [], []
    dataset_z_corr_train_means, dataset_z_corr_train_stds = [], []
    dataset_z_corr_test_means, dataset_z_corr_test_stds = [], []
    
    for dataset in tqdm(datasets):
        print(f"Processing {dataset}")
        gof_train_means, gof_test_means = [], []
        theta_corr_ctt_means = []
        z_corr_train_means, z_corr_test_means = [], []
        
        for i in range(2):
            y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
            theta_train = pd.read_csv(f'{input_dir}/{dataset}/theta_{i}.csv')['theta'].values
            df_z_train = pd.read_csv(f'{input_dir}/{dataset}/z_train_{i}.csv')
            train_indices = df_z_train['index'].values
            z_train = df_z_train['z'].values
            df_z_test = pd.read_csv(f'{input_dir}/{dataset}/z_test_{i}.csv')
            test_indices = df_z_test['index'].values
            z_test = df_z_test['z'].values
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

        dataset_gof_train_means.append(np.mean(gof_train_means))
        dataset_gof_test_means.append(np.mean(gof_test_means))
        dataset_theta_corr_ctt_means.append(np.mean(theta_corr_ctt_means))
        dataset_z_corr_train_means.append(np.mean(z_corr_train_means))
        dataset_z_corr_test_means.append(np.mean(z_corr_test_means))
        
        dataset_gof_train_stds.append(np.std(gof_train_means))
        dataset_gof_test_stds.append(np.std(gof_test_means))
        dataset_theta_corr_ctt_stds.append(np.std(theta_corr_ctt_means))
        dataset_z_corr_train_stds.append(np.std(z_corr_train_means))
        dataset_z_corr_test_stds.append(np.std(z_corr_test_means))
    
    error_bar_plot_double(
        datasets=datasets, 
        means_train=dataset_gof_train_means,
        stds_train=dataset_gof_train_stds,
        means_test=dataset_gof_test_means,
        stds_test=dataset_gof_test_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_gof",
        ylabel=r"Goodness of Fit",
    )   
    
    error_bar_plot_single(
        datasets=datasets,
        means=dataset_theta_corr_ctt_means,
        stds=dataset_theta_corr_ctt_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_theta_corr_ctt",
        ylabel=r"$\theta$ correlation with CTT",
    )
    
    error_bar_plot_double(
        datasets=datasets, 
        means_train=dataset_z_corr_train_means,
        stds_train=dataset_z_corr_train_stds,
        means_test=dataset_z_corr_test_means,
        stds_test=dataset_z_corr_test_stds,
        plot_path=f"{plot_dir}/amor_calibration_summarize_z_corr",
        ylabel=r"correlation of $z$",
    )   
    