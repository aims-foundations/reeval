import os
import torch
import pandas as pd
from tqdm import tqdm
from utils import goodness_of_fit_1PL_plot, theta_corr_ctt_plot, error_bar_plot

if __name__ == "__main__":
    plot_dir = f'../plot/nonamor_calibration'
    os.makedirs(plot_dir, exist_ok=True)
    
    input_dir = '../data/nonamor_calibration/'
    datasets = [f for f in os.listdir(input_dir)]
    # datasets = ['synthetic_efficiency', 'wikifact', 'entity_data_imputation', 'commonsense', 'quac', 'imdb', 'bbq', 'math', 'twitter_aae', 'truthful_qa', 'legal_support', 'boolq', 'narrative_qa', 'real_toxicity_prompts', 'bold', 'gsm', 'babi_qa', 'summarization_xsum', 'synthetic_reasoning_natural', 'dyck_language_np3', 'lsat_qa', 'raft', 'code', 'entity_matching', 'synthetic_reasoning', 'mmlu', 'airbench']

    gof_means, gof_stds = [], []
    corr_ctt_means, corr_ctt_stds = [], []
    for dataset in tqdm(datasets):
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

    error_bar_plot(
        datasets=datasets,
        means=gof_means,
        stds=gof_stds,
        plot_path=f"{plot_dir}/summarize_goodness_of_fit",
    )
    
    error_bar_plot(
        datasets=datasets,
        means=corr_ctt_means,
        stds=corr_ctt_stds,
        plot_path=f"{plot_dir}/summarize_theta_corr_ctt",
    )
    
